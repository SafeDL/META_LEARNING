"""Fixed configuration for the highway-env DIVA-Mine MVP."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """The deliberately small experiment described in the MVP design."""

    num_anchors: int = 128
    prior_rank: int = 2
    support_budget: int = 4
    total_budget: int = 20
    random_support_repeats: int = 20
    seed: int = 20260912

    @property
    def mining_budget(self) -> int:
        return self.total_budget - self.support_budget

    def validate(self) -> None:
        if self.prior_rank < 1:
            raise ValueError("prior_rank must be positive")
        if not 0 < self.support_budget < self.total_budget:
            raise ValueError("support_budget must be in (0, total_budget)")
        if self.total_budget > self.num_anchors:
            raise ValueError("total_budget cannot exceed num_anchors")
