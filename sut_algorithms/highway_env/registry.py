"""Registry for retained Highway-env systems under test."""

from __future__ import annotations

from pathlib import Path

from .base import Policy
from .idm_mobil import IDMMobilPolicy
from .mcts_cv import MCTSCVPolicy
from .ppo_ece import PPOPolicy
from .value_iteration import ValueIterationPolicy

RETAINED_SUTS = ("idm_mobil", "vi_ttc", "mcts_cv", "ppo_ece")


def policy_factory(sut: str, assets_root: Path) -> Policy:
    if sut == "idm_mobil":
        return IDMMobilPolicy()
    if sut == "vi_ttc":
        return ValueIterationPolicy()
    if sut == "mcts_cv":
        return MCTSCVPolicy()
    if sut == "ppo_ece":
        checkpoint = assets_root / "ppo_ece" / "vd_1_5_trial_1.zip"
        return PPOPolicy(checkpoint)
    raise KeyError(f"Unsupported SUT: {sut}")

