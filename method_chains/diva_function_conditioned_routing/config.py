"""Frozen protocol shared with the DIVA-DETOUR multi-function benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingExperimentConfig:
    """Fixed-budget, target-hidden evaluation settings."""

    num_anchors: int = 240
    prior_rank: int = 2
    support_budget: int = 10
    total_budget: int = 50
    random_repeats: int = 20
    seed: int = 20260914
    local_shrinkage: float = 0.5
    gate_error_scale: float = 0.30
    hierarchy_weight: float = 0.10

    def validate(self) -> None:
        if self.num_anchors < 1:
            raise ValueError("num_anchors must be positive")
        if self.prior_rank < 1:
            raise ValueError("prior_rank must be positive")
        if not 0 < self.support_budget < self.total_budget <= self.num_anchors:
            raise ValueError("require 0 < support_budget < total_budget <= num_anchors")
        if self.random_repeats < 1:
            raise ValueError("random_repeats must be positive")
        if self.local_shrinkage < 0 or self.gate_error_scale <= 0:
            raise ValueError("routing regularizers must be nonnegative and scales positive")
        if self.hierarchy_weight < 0:
            raise ValueError("hierarchy_weight must be nonnegative")
