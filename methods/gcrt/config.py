"""Development defaults; final evaluation uses its frozen protocol manifest."""

from dataclasses import dataclass

from benchmark.protocol import RESULT_ROOT


@dataclass(frozen=True)
class ModelConfig:
    steps: int = 1500
    learning_rate: float = 0.0003
    support_counts: tuple[int, ...] = (4, 8, 16)
    query_count: int = 32
    validation_interval: int = 100
    auxiliary_weight: float = 0.1
    target_weight: float = 0.25
    seed: int = 1103


CONFIG = ModelConfig()
MODEL_ROOT = RESULT_ROOT / "models"
