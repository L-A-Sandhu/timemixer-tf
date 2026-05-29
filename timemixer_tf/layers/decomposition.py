"""Series decomposition layers: moving average and DFT-based."""

import tensorflow as tf


class moving_avg(tf.keras.layers.Layer):
    """Moving average block to extract trend component of time series.

    Implemented as depthwise 1D convolution with uniform weights,
    matching the PyTorch AvgPool1d behavior exactly.
    """

    def __init__(self, kernel_size: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_size = (kernel_size - 1) // 2

    def build(self, input_shape):
        channels = input_shape[-1]
        # Depthwise conv: one filter per channel, each weight = 1/kernel_size
        kernel_init = tf.constant_initializer(1.0 / self.kernel_size)
        self.conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="valid",
            use_bias=False,
            depthwise_initializer=kernel_init,
            trainable=False,
            name="moving_avg_conv",
        )
        # Build to create weights
        self.conv.build([None, None, channels])
        super().build(input_shape)

    def call(self, x):
        # x: [B, T, C] — mirror left/right padding then depthwise conv
        front = tf.repeat(x[:, :1, :], self.pad_size, axis=1)
        end = tf.repeat(x[:, -1:, :], self.pad_size, axis=1)
        x_padded = tf.concat([front, x, end], axis=1)
        return self.conv(x_padded)

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size, "stride": self.stride})
        return config


class series_decomp(tf.keras.layers.Layer):
    """Series decomposition into residual (seasonal) and moving average (trend)."""

    def __init__(self, kernel_size: int, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def call(self, x):
        trend = self.moving_avg(x)
        season = x - trend
        return season, trend

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class DFT_series_decomp(tf.keras.layers.Layer):
    """DFT-based series decomposition: uses FFT to separate season and trend.

    Keeps only the top_k frequency components as seasonal, rest as trend.
    """

    def __init__(self, top_k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    def call(self, x):
        # x: [B, T, C]  —  rfft along the time axis (dim=1)
        x = tf.cast(x, tf.complex64)
        xf = tf.signal.rfft(tf.cast(x, tf.float32))

        # rfft shape: [B, T//2+1, C]
        freq = tf.abs(xf)
        mask_zero = tf.tile(tf.constant([[[0.0]]], dtype=freq.dtype), [1, 1, tf.shape(freq)[2]])
        zero_row = tf.tile(mask_zero, [tf.shape(freq)[0], 1, 1])
        # zero out DC component
        freq_dc_zero = tf.tensor_scatter_nd_update(
            freq,
            tf.stack(
                [
                    tf.range(tf.shape(freq)[0]),
                    tf.zeros(tf.shape(freq)[0], dtype=tf.int32),
                    tf.zeros(tf.shape(freq)[0], dtype=tf.int32),
                ],
                axis=1,
            ),
            tf.zeros(tf.shape(freq)[0]),
        )

        # Find threshold: top_k smallest among non-zero frequencies
        top_k_values = tf.sort(freq_dc_zero, axis=1, direction="DESCENDING")
        threshold = top_k_values[:, self.top_k : self.top_k + 1, :]

        # Zero out frequencies below threshold
        xf_filtered = tf.where(freq < threshold, tf.zeros_like(xf), xf)
        x_season = tf.signal.irfft(xf_filtered)
        # Trim to original time length if needed
        orig_T = tf.shape(x)[1]
        x_season = x_season[:, :orig_T, :]
        x_trend = x - x_season
        return x_season, x_trend

    def get_config(self):
        config = super().get_config()
        config.update({"top_k": self.top_k})
        return config
