"""TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting.

This is a complete TensorFlow 2.x reimplementation of the ICLR 2024 paper
"TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting".

Supports: long-term forecasting, short-term forecasting, imputation,
          anomaly detection, and classification.
"""
import tensorflow as tf

from timemixer_tf.config import TimeMixerConfig
from timemixer_tf.layers import (
    DataEmbedding_wo_pos,
    Normalize,
    PastDecomposableMixing,
    series_decomp,
)


class TimeMixer(tf.keras.Model):
    """TimeMixer: fully MLP-based multiscale time series model.

    Args:
        config: TimeMixerConfig dataclass with model hyperparameters.

    The model operates on multi-scale representations of the input,
    decomposing each scale into seasonal and trend components, mixing
    them across scales via PDM blocks, and producing predictions via
    Future Multipredictor Mixing (FMM).
    """

    def __init__(self, config: TimeMixerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Core PDM encoder blocks
        self.pdm_blocks = [
            PastDecomposableMixing(config)
            for _ in range(config.e_layers)
        ]

        # Preprocessing decomposition
        self.preprocess = series_decomp(config.moving_avg)

        # Embedding
        if config.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(
                1, config.d_model, config.embed, config.freq, config.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                config.enc_in, config.d_model, config.embed, config.freq, config.dropout)

        # Normalization per scale
        self.normalize_layers = [
            Normalize(config.enc_in, affine=True,
                      non_norm=(config.use_norm == 0))
            for _ in range(config.down_sampling_layers + 1)
        ]

        # Prediction heads per scale (forecasting tasks)
        if config.task_name in ("long_term_forecast", "short_term_forecast"):
            self.predict_layers = [
                tf.keras.layers.Dense(config.pred_len)
                for _ in range(config.down_sampling_layers + 1)
            ]

            if config.channel_independence == 1:
                self.projection_layer = tf.keras.layers.Dense(1, use_bias=True)
            else:
                self.projection_layer = tf.keras.layers.Dense(config.c_out, use_bias=True)
                self.out_res_layers = [
                    tf.keras.layers.Dense(
                        config.seq_len // (config.down_sampling_window ** i))
                    for i in range(config.down_sampling_layers + 1)
                ]
                self.regression_layers = [
                    tf.keras.layers.Dense(config.pred_len)
                    for _ in range(config.down_sampling_layers + 1)
                ]

        # Heads for imputation and anomaly detection
        if config.task_name in ("imputation", "anomaly_detection"):
            if config.channel_independence == 1:
                self.projection_layer = tf.keras.layers.Dense(1, use_bias=True)
            else:
                self.projection_layer = tf.keras.layers.Dense(config.c_out, use_bias=True)

        # Head for classification
        if config.task_name == "classification":
            self.projection = tf.keras.layers.Dense(config.num_class)

    # ------------------------------------------------------------------
    # Multi-scale down-sampling
    # ------------------------------------------------------------------
    def _multi_scale_process_inputs(self, x_enc, x_mark_enc):
        """Create multi-scale versions of the input via down-sampling."""
        method = self.config.down_sampling_method
        window = self.config.down_sampling_window
        B = tf.shape(x_enc)[0]

        x_enc = tf.transpose(x_enc, [0, 2, 1])  # [B, C, T]
        x_enc_ori = x_enc
        x_mark_ori = x_mark_enc

        x_enc_sampling_list = [tf.transpose(x_enc, [0, 2, 1])]  # scale 0
        x_mark_sampling_list = [x_mark_enc]

        for _ in range(self.config.down_sampling_layers):
            if method == "max":
                x_enc_sampling = tf.nn.pool(
                    x_enc_ori, [window], "MAX", strides=[window], padding="VALID")
            elif method == "avg":
                # Manual avg pool without cuDNN: reshape [B, C, T] → [B, C, T//w, w] → mean
                B = tf.shape(x_enc_ori)[0]
                C = tf.shape(x_enc_ori)[1]
                T_orig = tf.shape(x_enc_ori)[2]
                T_new = T_orig // window
                x_trunc = x_enc_ori[:, :, :T_new * window]
                x_reshaped = tf.reshape(x_trunc, [B, C, T_new, window])
                x_enc_sampling = tf.reduce_mean(x_reshaped, axis=-1)
            elif method == "conv":
                # Manual conv-like down-sampling to avoid cuDNN
                B = tf.shape(x_enc_ori)[0]
                C = tf.shape(x_enc_ori)[1]
                T_orig = tf.shape(x_enc_ori)[2]
                T_new = T_orig // window
                x_trunc = x_enc_ori[:, :, :T_new * window]
                x_reshaped = tf.reshape(x_trunc, [B, C, T_new, window])
                x_enc_sampling = x_reshaped[:, :, :, 0]  # take first element (strided)
            else:
                return [x_enc], [x_mark_enc]

            x_enc_sampling_list.append(tf.transpose(x_enc_sampling, [0, 2, 1]))
            x_enc_ori = x_enc_sampling

            if x_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_ori[:, ::window, :])
                x_mark_ori = x_mark_ori[:, ::window, :]

        if x_mark_ori is None:
            x_mark_sampling_list = None

        return x_enc_sampling_list, x_mark_sampling_list

    # ------------------------------------------------------------------
    # Pre-encoder: optional decomposition of input
    # ------------------------------------------------------------------
    def _pre_enc(self, x_list):
        if self.config.channel_independence == 1:
            return (x_list, None)
        out1_list = []
        out2_list = []
        for x in x_list:
            x1, x2 = self.preprocess(x)
            out1_list.append(x1)
            out2_list.append(x2)
        return (out1_list, out2_list)

    # ------------------------------------------------------------------
    # Output projection
    # ------------------------------------------------------------------
    def _out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = tf.transpose(out_res, [0, 2, 1])
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res)
        out_res = tf.transpose(out_res, [0, 2, 1])
        return dec_out + out_res

    # ------------------------------------------------------------------
    # Future Multipredictor Mixing (FMM)
    # ------------------------------------------------------------------
    def _future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.config.channel_independence == 1:
            for i, enc_out in enumerate(enc_out_list):
                dec_out = tf.transpose(
                    self.predict_layers[i](tf.transpose(enc_out, [0, 2, 1])),
                    [0, 2, 1])
                dec_out = self.projection_layer(dec_out)
                c_out = self.config.c_out
                pred_len = self.config.pred_len
                dec_out = tf.reshape(dec_out, [B, c_out, pred_len])
                dec_out = tf.transpose(dec_out, [0, 2, 1])
                dec_out_list.append(dec_out)
        else:
            for i, (enc_out, out_res) in enumerate(zip(enc_out_list, x_list[1])):
                dec_out = tf.transpose(
                    self.predict_layers[i](tf.transpose(enc_out, [0, 2, 1])),
                    [0, 2, 1])
                dec_out = self._out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        return dec_out_list

    # ------------------------------------------------------------------
    # Task: long/short-term forecasting
    # ------------------------------------------------------------------
    def _forecast(self, x_enc, x_mark_enc, training=False):
        B = tf.shape(x_enc)[0]
        x_enc_scales, x_mark_scales = self._multi_scale_process_inputs(x_enc, x_mark_enc)

        # Normalize and reshape per scale
        x_list = []
        for i, x in enumerate(x_enc_scales):
            x_norm = self.normalize_layers[i](x, mode="norm")
            if self.config.channel_independence == 1:
                _, T, N = tf.unstack(tf.shape(x_norm))
                x_norm = tf.reshape(tf.transpose(x_norm, [0, 2, 1]), [B * N, T, 1])
            x_list.append(x_norm)

        # Embed
        x_parts = self._pre_enc(x_list)
        enc_out_list = []
        for i, x in enumerate(x_parts[0]):
            enc_out = self.enc_embedding(x, None, training=training)
            enc_out_list.append(enc_out)

        # PDM blocks
        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list, training=training)

        # FMM
        dec_out_list = self._future_multi_mixing(B, enc_out_list, x_parts)
        dec_out = tf.reduce_sum(tf.stack(dec_out_list, axis=-1), axis=-1)
        dec_out = self.normalize_layers[0](dec_out, mode="denorm")
        return dec_out

    # ------------------------------------------------------------------
    # Task: imputation
    # ------------------------------------------------------------------
    def _imputation(self, x_enc, x_mark_enc, mask, training=False):
        B = tf.shape(x_enc)[0]
        T = tf.shape(x_enc)[1]
        N = tf.shape(x_enc)[2]

        # Instance normalization with mask
        mask_sum = tf.reduce_sum(tf.cast(mask == 1, x_enc.dtype), axis=1)
        means = tf.reduce_sum(x_enc, axis=1) / (mask_sum + 1e-5)
        means = tf.stop_gradient(tf.expand_dims(means, axis=1))
        x_enc = x_enc - means
        x_enc = tf.where(mask == 0, tf.zeros_like(x_enc), x_enc)
        stdev = tf.sqrt(
            tf.reduce_sum(x_enc * x_enc, axis=1) / (mask_sum + 1e-5) + 1e-5)
        stdev = tf.stop_gradient(tf.expand_dims(stdev, axis=1))
        x_enc = x_enc / stdev

        x_enc_scales, x_mark_scales = self._multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        for x in x_enc_scales:
            if self.config.channel_independence == 1:
                B_s, T_s, N_s = tf.unstack(tf.shape(x))
                x = tf.reshape(tf.transpose(x, [0, 2, 1]), [B_s * N_s, T_s, 1])
            x_list.append(x)

        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None, training=training)
            enc_out_list.append(enc_out)

        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list, training=training)

        dec_out = self.projection_layer(enc_out_list[0])
        c_out = self.config.c_out
        dec_out = tf.reshape(dec_out, [B, c_out, -1])
        dec_out = tf.transpose(dec_out, [0, 2, 1])

        # Denormalize
        broadcast_stdev = tf.tile(tf.expand_dims(stdev[:, 0, :], 1), [1, T, 1])
        broadcast_means = tf.tile(tf.expand_dims(means[:, 0, :], 1), [1, T, 1])
        dec_out = dec_out * broadcast_stdev + broadcast_means
        return dec_out

    # ------------------------------------------------------------------
    # Task: anomaly detection
    # ------------------------------------------------------------------
    def _anomaly_detection(self, x_enc, training=False):
        B = tf.shape(x_enc)[0]
        N = tf.shape(x_enc)[2]

        x_enc_scales, _ = self._multi_scale_process_inputs(x_enc, None)

        x_list = []
        for i, x in enumerate(x_enc_scales):
            x_norm = self.normalize_layers[i](x, mode="norm")
            if self.config.channel_independence == 1:
                B_s, T_s, N_s = tf.unstack(tf.shape(x_norm))
                x_norm = tf.reshape(tf.transpose(x_norm, [0, 2, 1]), [B_s * N_s, T_s, 1])
            x_list.append(x_norm)

        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None, training=training)
            enc_out_list.append(enc_out)

        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list, training=training)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = tf.reshape(dec_out, [B, self.config.c_out, -1])
        dec_out = tf.transpose(dec_out, [0, 2, 1])
        dec_out = self.normalize_layers[0](dec_out, mode="denorm")
        return dec_out

    # ------------------------------------------------------------------
    # Task: classification
    # ------------------------------------------------------------------
    def _classification(self, x_enc, x_mark_enc, training=False):
        x_enc_scales, _ = self._multi_scale_process_inputs(x_enc, None)
        enc_out_list = []
        for x in x_enc_scales:
            if self.config.channel_independence == 1:
                B, T, N = tf.unstack(tf.shape(x))
                x = tf.reshape(tf.transpose(x, [0, 2, 1]), [B * N, T, 1])
            enc_out = self.enc_embedding(x, None, training=training)
            enc_out_list.append(enc_out)

        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list, training=training)

        enc_out = enc_out_list[0]
        enc_out = tf.nn.gelu(enc_out)
        enc_out = tf.keras.layers.Dropout(self.config.dropout)(enc_out, training=training)
        if x_mark_enc is not None:
            if len(x_mark_enc.shape) == 2:
                x_mark_enc = tf.expand_dims(x_mark_enc, -1)
            if self.config.channel_independence == 1:
                N = self.config.enc_in
                x_mark_enc = tf.repeat(x_mark_enc, N, axis=0)
            enc_out = enc_out * x_mark_enc
        enc_out = tf.reshape(enc_out, [tf.shape(enc_out)[0], -1])
        return self.projection(enc_out)

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------
    def call(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
             mask=None, training=False):
        task = self.config.task_name
        if task in ("long_term_forecast", "short_term_forecast"):
            return self._forecast(x_enc, x_mark_enc, training=training)
        elif task == "imputation":
            return self._imputation(x_enc, x_mark_enc, mask, training=training)
        elif task == "anomaly_detection":
            return self._anomaly_detection(x_enc, training=training)
        elif task == "classification":
            return self._classification(x_enc, x_mark_enc, training=training)
        raise ValueError(f"Unknown task_name: {task}")

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def train_step(self, data):
        x_enc, x_mark_enc, x_dec, x_mark_dec, *rest = data
        mask = rest[0] if rest else None

        with tf.GradientTape() as tape:
            pred = self(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, training=True)
            # Slice to prediction horizon for forecasting
            if self.config.task_name in ("long_term_forecast", "short_term_forecast"):
                pred = pred[:, -self.config.pred_len:, :]
                # Get targets (y is the second element in typical data setup)
                # For now, compatibility placeholder
                true = x_dec if x_dec is not None else x_enc[:, -self.config.pred_len:, :]
                f_dim = -1  # 'M' task
                pred = pred[..., f_dim:]
                true = true[..., f_dim:]
                loss = tf.reduce_mean(tf.square(pred - true))
            elif self.config.task_name == "imputation":
                # Masked MSE
                loss = tf.reduce_mean(tf.square(pred[mask == 0] - x_enc[mask == 0]))
            else:
                loss = tf.reduce_mean(tf.square(pred - x_enc))

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": loss}

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config})
        return config
