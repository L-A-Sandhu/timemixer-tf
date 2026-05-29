"""Past Decomposable Mixing: multi-scale season and trend mixing blocks."""
import tensorflow as tf

from timemixer_tf.layers.decomposition import DFT_series_decomp, series_decomp


class MultiScaleSeasonMixing(tf.keras.layers.Layer):
    """Bottom-up mixing of seasonal patterns from fine to coarse scales."""

    def __init__(self, down_sampling_layers: int, seq_len: int,
                 down_sampling_window: int, **kwargs):
        super().__init__(**kwargs)
        self.down_sampling_layers = down_sampling_layers
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_layers = []

    def build(self, input_shape):
        for i in range(self.down_sampling_layers):
            in_len = self.seq_len // (self.down_sampling_window ** i)
            out_len = self.seq_len // (self.down_sampling_window ** (i + 1))
            self.down_layers.append([
                tf.keras.layers.Dense(out_len, name=f"down_linear1_{i}"),
                tf.keras.layers.Dense(out_len, name=f"down_linear2_{i}"),
            ])
        super().build(input_shape)

    def call(self, season_list):
        # season_list: list of [B, C, T] tensors at different scales
        out_high = season_list[0]  # finest scale
        out_low = season_list[1]   # next scale
        out_season_list = [tf.transpose(out_high, [0, 2, 1])]  # [B, T, C]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_layers[i][0](out_high)
            out_low_res = tf.nn.gelu(out_low_res)
            out_low_res = self.down_layers[i][1](out_low_res)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(tf.transpose(out_high, [0, 2, 1]))

        return out_season_list

    def get_config(self):
        config = super().get_config()
        config.update({
            "down_sampling_layers": self.down_sampling_layers,
            "seq_len": self.seq_len,
            "down_sampling_window": self.down_sampling_window,
        })
        return config


class MultiScaleTrendMixing(tf.keras.layers.Layer):
    """Top-down mixing of trend patterns from coarse to fine scales."""

    def __init__(self, down_sampling_layers: int, seq_len: int,
                 down_sampling_window: int, **kwargs):
        super().__init__(**kwargs)
        self.down_sampling_layers = down_sampling_layers
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.up_layers = []

    def build(self, input_shape):
        for i in reversed(range(self.down_sampling_layers)):
            in_len = self.seq_len // (self.down_sampling_window ** (i + 1))
            out_len = self.seq_len // (self.down_sampling_window ** i)
            self.up_layers.append([
                tf.keras.layers.Dense(out_len, name=f"up_linear1_{i}"),
                tf.keras.layers.Dense(out_len, name=f"up_linear2_{i}"),
            ])
        super().build(input_shape)

    def call(self, trend_list):
        trend_list_reverse = list(reversed(trend_list))
        out_low = trend_list_reverse[0]      # coarsest
        out_high = trend_list_reverse[1]     # next
        out_trend_list = [tf.transpose(out_low, [0, 2, 1])]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_layers[i][0](out_low)
            out_high_res = tf.nn.gelu(out_high_res)
            out_high_res = self.up_layers[i][1](out_high_res)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(tf.transpose(out_low, [0, 2, 1]))

        out_trend_list = list(reversed(out_trend_list))
        return out_trend_list

    def get_config(self):
        config = super().get_config()
        config.update({
            "down_sampling_layers": self.down_sampling_layers,
            "seq_len": self.seq_len,
            "down_sampling_window": self.down_sampling_window,
        })
        return config


class PastDecomposableMixing(tf.keras.layers.Layer):
    """PDM block: decomposes → mixes season (bottom-up) and trend (top-down).

    This is the core building block of TimeMixer. It takes multi-scale input,
    decomposes each into seasonal and trend components, mixes them separately
    across scales, then recombines them.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.down_sampling_window = config.down_sampling_window
        self.channel_independence = config.channel_independence

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.dropout = tf.keras.layers.Dropout(config.dropout)

        if config.decomp_method == "moving_avg":
            self.decomposition = series_decomp(config.moving_avg)
        elif config.decomp_method == "dft_decomp":
            self.decomposition = DFT_series_decomp(config.top_k)
        else:
            raise ValueError(f"Unknown decomp_method: {config.decomp_method}")

        if config.channel_independence == 0:
            self.cross_layer = tf.keras.Sequential([
                tf.keras.layers.Dense(config.d_ff, activation="gelu"),
                tf.keras.layers.Dense(config.d_model),
            ], name="cross_layer")

        self.season_mixing = MultiScaleSeasonMixing(
            config.down_sampling_layers, config.seq_len, config.down_sampling_window)
        self.trend_mixing = MultiScaleTrendMixing(
            config.down_sampling_layers, config.seq_len, config.down_sampling_window)

        self.out_cross_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(config.d_ff, activation="gelu"),
            tf.keras.layers.Dense(config.d_model),
        ], name="out_cross_layer")

    def call(self, x_list, training=False):
        # x_list: list of [B, T, d_model]
        length_list = [tf.shape(x)[1] for x in x_list]

        # 1. Decompose
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season, training=training)
                trend = self.cross_layer(trend, training=training)
            # [B, T, C] → [B, C, T] for mixing
            season_list.append(tf.transpose(season, [0, 2, 1]))
            trend_list.append(tf.transpose(trend, [0, 2, 1]))

        # 2. Mix
        out_season_list = self.season_mixing(season_list)
        out_trend_list = self.trend_mixing(trend_list)

        # 3. Recombine
        out_list = []
        for ori, out_season, out_trend, length in zip(
                x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out, training=training)
            out_list.append(out[:, :length, :])

        return out_list

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config})
        return config
