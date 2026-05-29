import dataclasses


@dataclasses.dataclass
class TimeMixerConfig:
    task_name: str = "long_term_forecast"
    seq_len: int = 96
    label_len: int = 0
    pred_len: int = 96
    enc_in: int = 7
    c_out: int = 7
    d_model: int = 16
    d_ff: int = 32
    e_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1
    embed: str = "timeF"
    freq: str = "h"
    activation: str = "gelu"
    moving_avg: int = 25
    factor: int = 1
    distil: bool = True
    channel_independence: int = 1
    decomp_method: str = "moving_avg"  # "moving_avg" or "dft_decomp"
    top_k: int = 5
    down_sampling_layers: int = 3
    down_sampling_window: int = 2
    down_sampling_method: str = "avg"  # "avg", "max", or "conv"
    use_norm: int = 1
    use_future_temporal_feature: int = 0
    num_class: int | None = None
    use_amp: bool = False
    output_attention: bool = False
    learning_rate: float = 0.01
    train_epochs: int = 10
    batch_size: int = 128
    patience: int = 10
    loss: str = "MSE"
    lradj: str = "TST"
    pct_start: float = 0.2
    mask_rate: float = 0.125
    anomaly_ratio: float = 0.25
    seasonal_patterns: str = "Monthly"
    inverse: bool = False

    @property
    def down_sampling_window_list(self) -> list[int]:
        return [self.down_sampling_window**i for i in range(self.down_sampling_layers + 1)]

    @property
    def seq_len_list(self) -> list[int]:
        return [
            self.seq_len // (self.down_sampling_window**i)
            for i in range(self.down_sampling_layers + 1)
        ]
