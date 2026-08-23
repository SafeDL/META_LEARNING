"""Stage guards prevent accidental simultaneous-from-scratch optimization."""
from __future__ import annotations

from enum import Enum


class TrainingStage(str, Enum):
    INNER_PRETRAIN = "inner_pretrain"
    INNER_LATENT_CALIBRATION = "inner_latent_calibration"
    POSTERIOR = "posterior"
    OUTER = "outer"
    LIGHT_JOINT = "light_joint"


CANONICAL_TRAINING_STAGES = (
    TrainingStage.INNER_PRETRAIN,
    TrainingStage.POSTERIOR,
    TrainingStage.INNER_LATENT_CALIBRATION,
    TrainingStage.OUTER,
)


def validate_stage_transition(stage: TrainingStage, previous: TrainingStage | None) -> None:
    index = CANONICAL_TRAINING_STAGES.index(stage)
    required = None if index == 0 else CANONICAL_TRAINING_STAGES[index - 1]
    if previous != required:
        expected = "no checkpoint" if required is None else f"an {required.value} checkpoint"
        actual = "no checkpoint" if previous is None else f"an {previous.value} checkpoint"
        raise RuntimeError(f"{stage.value} requires {expected}, got {actual}")


def trainable_components(stage: TrainingStage) -> frozenset[str]:
    return {
        TrainingStage.INNER_PRETRAIN: frozenset({"map_encoder", "shared_feature_encoder", "option_embedding", "inner_sac"}),
        TrainingStage.INNER_LATENT_CALIBRATION: frozenset({"shared_feature_encoder", "option_embedding", "inner_sac"}),
        TrainingStage.POSTERIOR: frozenset({"episode_token_builder", "posterior", "outcome_decoder"}),
        TrainingStage.OUTER: frozenset({"scene_policies"}),
        TrainingStage.LIGHT_JOINT: frozenset({"scene_policies", "inner_sac", "posterior"}),
    }[stage]
