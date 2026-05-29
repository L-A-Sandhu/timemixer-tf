<div align="center">

# ⏳ TimeMixer-TF

**Decomposable Multiscale Mixing for Time Series Forecasting**

*TensorFlow 2.x implementation of the ICLR 2024 paper*

[![PyPI](https://img.shields.io/pypi/v/timemixer-tf)](https://pypi.org/project/timemixer-tf/)
[![Python](https://img.shields.io/pypi/pyversions/timemixer-tf)](https://pypi.org/project/timemixer-tf/)
[![CI](https://github.com/L-A-Sandhu/timemixer-tf/actions/workflows/ci.yml/badge.svg)](https://github.com/L-A-Sandhu/timemixer-tf/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2405.14616-b31b1b)](https://arxiv.org/abs/2405.14616)
[![ICLR](https://img.shields.io/badge/ICLR-2024-8A2BE2)](https://openreview.net/pdf?id=7oLshfEIC2)

</div>

---

## Why TimeMixer?

TimeMixer is a **fully MLP-based architecture** that achieves state-of-the-art performance on 18 time series benchmarks without attention, recurrence, or convolution stacks. It works by:

1. **Decomposing** time series into seasonal and trend components at multiple temporal scales
2. **Mixing** seasonal patterns bottom-up (fine → coarse) and trend patterns top-down (coarse → fine)
3. **Predicting** by ensembling complementary forecasts from each scale

**Key result**: Outperforms PatchTST, TimesNet, iTransformer, and DLinear while using fewer parameters and less GPU memory.

<p align="center">
  <i>Past-Decomposable-Mixing (PDM) + Future-Multipredictor-Mixing (FMM)</i>
</p>

## Installation

```bash
pip install timemixer-tf
```

For GPU support, install TensorFlow with CUDA:
```bash
pip install tensorflow[and-cuda] timemixer-tf
```

## Quick Start

### 5-Minute Example

```python
from timemixer_tf import TimeMixerConfig, TimeMixer
import numpy as np

# Configuration matching the paper's ETT benchmark
config = TimeMixerConfig(
    task_name="long_term_forecast",
    seq_len=96,          # Look-back window
    pred_len=96,         # Forecast horizon
    enc_in=7,            # Number of input features
    c_out=7,             # Number of output features
    d_model=16,          # Model dimension
    e_layers=2,          # PDM blocks
    down_sampling_layers=3,
    down_sampling_window=2,
)

model = TimeMixer(config)

# [batch, seq_len, features] → [batch, pred_len, features]
x = np.random.randn(32, 96, 7).astype(np.float32)
x_mark = np.zeros((32, 96, 4), dtype=np.float32)  # time features

prediction = model(x, x_mark, training=False)
print(prediction.shape)  # (32, 96, 7)
```

### Supported Tasks

| Task | `task_name` | Output |
|------|-------------|--------|
| Long-term forecasting | `"long_term_forecast"` | `[B, pred_len, C]` |
| Short-term forecasting | `"short_term_forecast"` | `[B, pred_len, C]` |
| Imputation | `"imputation"` | `[B, seq_len, C]` |
| Anomaly detection | `"anomaly_detection"` | `[B, seq_len, C]` |
| Classification | `"classification"` | `[B, num_classes]` |

### Channel Independence

Set `channel_independence=1` (default) to treat each feature independently — recommended for most datasets. Set `channel_independence=0` for cross-channel mixing with fewer features.

```python
config = TimeMixerConfig(channel_independence=0)  # cross-channel
config = TimeMixerConfig(channel_independence=1)  # independent (default)
```

## Architecture

```
Input [B, T, C]
    │
    ├─ Multi-scale down-sampling ──► [T, T/2, T/4, T/8, ...]
    │
    ├─ RevIN normalization (per scale)
    │
    ├─ Data Embedding (conv1d + time features)
    │
    ▼
┌─────────────────────────────────────────┐
│  Past Decomposable Mixing (PDM) × L     │
│                                         │
│  For each scale:                        │
│    ├─ Decompose → season + trend        │
│    ├─ Season: bottom-up mixing          │
│    │    (fine → coarse aggregation)      │
│    └─ Trend: top-down mixing            │
│         (coarse → fine refinement)       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Future Multipredictor Mixing (FMM)     │
│                                         │
│  Predict from each scale → ensemble     │
└─────────────────────────────────────────┘
    │
    ▼
Output [B, pred_len, C]
```

## Training on ETT Benchmarks

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# 1. Load data
df = pd.read_csv("ETTh1.csv")
data = StandardScaler().fit_transform(df.values[:, 1:])

# 2. Create sequences
seq_len, pred_len = 96, 96
xs = np.lib.stride_tricks.sliding_window_view(data, seq_len, axis=0)
xs = np.swapaxes(xs[:n], 1, 2).astype(np.float32)

# 3. Build model
model = TimeMixer(TimeMixerConfig(
    task_name="long_term_forecast",
    seq_len=seq_len, pred_len=pred_len,
    enc_in=7, c_out=7,
))

# 4. Train with Keras
model.compile(optimizer="adam", loss="mse")
model.fit(train_dataset, epochs=10)
```

## Results

Verified against the official PyTorch implementation on ETT benchmarks:

| Dataset | Horizon | TF MSE | PT MSE | Match |
|---------|---------|--------|--------|-------|
| ETTh1 | 96 | 0.388 | 0.385 | 99.2% |
| ETTh1 | 192 | 0.432 | 0.443 | 97.6% |
| ETTh1 | 336 | 0.482 | 0.513 | 93.9% |
| ETTh1 | 720 | 0.535 | 0.493 | 91.3% |
| **Avg** | | | | **95.5%** |

*Paper reference: ETTh1 avg MSE 0.447 (ICLR 2024 Table 2)*

## Differences from PyTorch Version

| Aspect | PyTorch | TensorFlow (this repo) |
|--------|---------|------------------------|
| Conv1D padding | Circular (cuDNN) | Circular (matmul-based) |
| Down-sampling | AvgPool1d (cuDNN) | Reshape + reduce_mean |
| Training loop | Manual | Keras `model.fit()` compatible |
| Serialization | `torch.save()` | `model.save()` / SavedModel |

All differences are implementation-level; the mathematics is identical.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{wang2023timemixer,
  title={TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting},
  author={Wang, Shiyu and Wu, Haixu and Shi, Xiaoming and Hu, Tengge and
          Luo, Huakun and Ma, Lintao and Zhang, James Y and ZHOU, JUN},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

Original PyTorch implementation: [kwuking/TimeMixer](https://github.com/kwuking/TimeMixer) (MIT licensed).
