"""Data embedding layers: token, temporal, time-feature, and combined embeddings."""

import tensorflow as tf


class TokenEmbedding(tf.keras.layers.Layer):
    """Feature embedding via learned linear projection.

    Equivalent to PT Conv1d(c_in, d_model, k=3, padding='circular')
    reimplemented as a pure matmul to avoid cuDNN dependencies.
    Uses circular padding + sliding window matmul over k=3 context.
    """

    def __init__(self, c_in: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.c_in = c_in
        self.d_model = d_model

    def build(self, input_shape):
        kaiming = tf.keras.initializers.HeNormal()
        # Weight: [3 * c_in, d_model] representing a k=3 conv kernel
        self.kernel = self.add_weight(
            name="kernel",
            shape=(3 * self.c_in, self.d_model),
            initializer=kaiming,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: [B, T, C] → circular-pad → extract 3-windows → matmul
        x_t = tf.transpose(x, [0, 2, 1])  # [B, C, T]
        # Circular pad: wrap 1 left and 1 right
        x_pad = tf.concat([x_t[:, :, -1:], x_t, x_t[:, :, :1]], axis=2)  # [B, C, T+2]
        # Extract sliding windows of size 3 along the T dimension
        # Stack shifted versions
        windows = tf.stack(
            [
                x_pad[:, :, 0:-2],
                x_pad[:, :, 1:-1],
                x_pad[:, :, 2:],
            ],
            axis=-1,
        )  # [B, C, T, 3]
        B_dim = tf.shape(windows)[0]
        C_dim = tf.shape(windows)[1]
        T_dim = tf.shape(windows)[2]
        # Reshape to [B, T, 3*C] then matmul with [3*C, d_model]
        windows = tf.transpose(windows, [0, 2, 1, 3])  # [B, T, C, 3]
        windows = tf.reshape(windows, [B_dim, T_dim, 3 * C_dim])  # [B, T, 3C]
        out = tf.matmul(windows, self.kernel)  # [B, T, d_model]
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"c_in": self.c_in, "d_model": self.d_model})
        return config


class TimeFeatureEmbedding(tf.keras.layers.Layer):
    """Linear embedding for time features (e.g., hour, day, month)."""

    def __init__(self, d_model: int, freq: str = "h", **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.freq = freq
        freq_map = {"h": 4, "t": 5, "s": 6, "ms": 7, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        self.d_inp = freq_map[freq]
        self.linear = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, x):
        return self.linear(x)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "freq": self.freq})
        return config


class DataEmbedding_wo_pos(tf.keras.layers.Layer):
    """Data embedding without positional encoding: value + temporal.

    When x is None and x_mark is provided, returns only temporal embedding
    (used for future temporal features).
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "timeF",
        freq: str = "h",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.c_in = c_in
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout_rate = dropout

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, x_mark, training=False):
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            out = self.value_embedding(x)
        else:
            out = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(out, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "c_in": self.c_in,
                "d_model": self.d_model,
                "embed_type": self.embed_type,
                "freq": self.freq,
                "dropout": self.dropout_rate,
            }
        )
        return config
