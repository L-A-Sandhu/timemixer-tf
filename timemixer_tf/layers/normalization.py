"""RevIN: Reversible Instance Normalization for Time Series."""

import tensorflow as tf


class Normalize(tf.keras.layers.Layer):
    """Reversible Instance Normalization (RevIN).

    Normalizes input per-instance, storing statistics for later denormalization.
    Used as a preprocessing step in TimeMixer to handle distribution shift.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        non_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.non_norm = non_norm
        self._mean = None
        self._stdev = None

    def build(self, input_shape):
        if self.affine:
            self.affine_weight = self.add_weight(
                name="affine_weight", shape=(self.num_features,), initializer="ones", trainable=True
            )
            self.affine_bias = self.add_weight(
                name="affine_bias", shape=(self.num_features,), initializer="zeros", trainable=True
            )
        super().build(input_shape)

    def _get_statistics(self, x):
        axes = tuple(range(1, len(x.shape) - 1))  # reduce over time dims, keep batch and feature
        self._mean = tf.stop_gradient(tf.reduce_mean(x, axis=axes, keepdims=True))
        self._stdev = tf.stop_gradient(
            tf.sqrt(tf.math.reduce_variance(x, axis=axes, keepdims=True) + self.eps)
        )

    def _normalize(self, x):
        if self.non_norm:
            return x
        x = x - self._mean
        x = x / self._stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self._stdev
        x = x + self._mean
        return x

    def call(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "eps": self.eps,
                "affine": self.affine,
                "non_norm": self.non_norm,
            }
        )
        return config
