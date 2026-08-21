"""Stage guards prevent accidental simultaneous-from-scratch optimization."""
from __future__ import annotations

from enum import Enum


class TrainingStage(str, Enum):
    INNER_PRETRAIN = "inner_pretrain"
    POSTERIOR = "posterior"
    OUTER = "outer"
    LIGHT_JOINT = "light_joint"


def trainable_components(stage: TrainingStage) -> frozenset[str]:
    return {
        TrainingStage.INNER_PRETRAIN: frozenset({"map_encoder", "shared_feature_encoder", "option_embedding", "inner_sac"}),
        TrainingStage.POSTERIOR: frozenset({"episode_token_builder", "posterior", "outcome_decoder"}),
        TrainingStage.OUTER: frozenset({"scene_policies"}),
        TrainingStage.LIGHT_JOINT: frozenset({"scene_policies", "inner_sac", "posterior"}),
    }[stage]
