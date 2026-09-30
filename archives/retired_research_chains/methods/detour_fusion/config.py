"""Fixed configuration owned by the isolated Mining-Detour fusion chain."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionExperimentConfig:
    """Shared fixed-budget configuration for target-hidden failure mining."""

    num_anchors: int = 240
    prior_rank: int = 2
    support_budget: int = 10
    total_budget: int = 50
    random_support_repeats: int = 20
    seed: int = 20260914
    detour_hierarchy_weight: float = 0.10

    @property
    def mining_budget(self) -> int:
        return self.total_budget - self.support_budget

    def validate(self) -> None:
        """Validate the fixed multi-function mining protocol."""
        if self.prior_rank < 1:
            raise ValueError("prior_rank must be positive")
        if not 0 < self.support_budget < self.total_budget <= self.num_anchors:
            raise ValueError("require 0 < support_budget < total_budget <= num_anchors")
        if self.random_support_repeats < 1:
            raise ValueError("random_support_repeats must be positive")
        if not 0.0 <= self.detour_hierarchy_weight <= 1.0:
            raise ValueError("detour_hierarchy_weight must be in [0, 1]")
