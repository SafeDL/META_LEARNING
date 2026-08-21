"""Stage guards prevent accidental simultaneous-from-scratch optimization."""
from __future__ import annotations

from enum import Enum


class TrainingStage(str, Enum):
    INNER_PRETRAIN = "inner_pretrain"
    POSTERIOR = "posterior"
    OUTER = "outer"
    JOINT = "joint"


def trainable_components(stage: TrainingStage) -> frozenset[str]:
    return {
        TrainingStage.INNER_PRETRAIN: frozenset({"inner"}),
        TrainingStage.POSTERIOR: frozenset({"trajectory", "posterior", "outcome_decoder"}),
        TrainingStage.OUTER: frozenset({"outer"}),
        TrainingStage.JOINT: frozenset({"outer", "inner", "posterior", "trajectory", "outcome_decoder"}),
    }[stage]
