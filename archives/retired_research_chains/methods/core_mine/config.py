"""Frozen experimental constants for the CoRe-Mine cache study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BUDGETS = (10, 20, 30, 50)
TOTAL_BUDGET = 50
SUPPORT_BUDGET = 10
B_REF = 50
LAMBDA = 0.10
COVERAGE_LENGTH = 0.20
EVENT_THRESHOLD = 0.375
COLLISION_THRESHOLD = 0.875
SOURCE_COUNT = 6
DEVELOPMENT_BANKS = (20261103, 20261117)
VALIDATION_BANKS = (20261201, 20261213, 20261229)
ROOT = Path("results/method_chains/core_mine")
BANK_DIR = Path("results/method_chains/function_posterior_search/confirmation/banks")


@dataclass(frozen=True)
class CoreMineConfig:
    residual_length: float = 0.30
    residual_amplitude: float = 0.25
    observation_noise: float = 0.05
    lambda_: float = LAMBDA
    include_null: bool = True
    compositional: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "residual_length": self.residual_length,
            "residual_amplitude": self.residual_amplitude,
            "observation_noise": self.observation_noise,
            "lambda": self.lambda_,
            "include_null": self.include_null,
            "compositional": self.compositional,
            "coverage_length": COVERAGE_LENGTH,
            "B_ref": B_REF,
        }
