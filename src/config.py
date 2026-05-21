from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    n_steps: int = 100
    batch_size: int = 64
    lr: float = 2e-4
    log_every: int = 25


@dataclass
class EvalConfig:
    n_samples: int = 256
    sample_batch: int = 64


@dataclass
class SmokeConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0
