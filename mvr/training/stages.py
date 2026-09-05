"""Stage guards prevent accidental simultaneous-from-scratch optimization."""
from __future__ import annotations

from enum import Enum


class TrainingStage(str, Enum):
    INTERACTION_PRIOR = "interaction_prior"
    CONTEXT_META = "context_meta"
    OUTER = "outer"


CANONICAL_TRAINING_STAGES = (
    TrainingStage.INTERACTION_PRIOR,
    TrainingStage.CONTEXT_META,
    TrainingStage.OUTER,
)


def validate_stage_transition(stage: TrainingStage, previous: TrainingStage | None) -> None:
    index = CANONICAL_TRAINING_STAGES.index(stage)
    required = None if index == 0 else CANONICAL_TRAINING_STAGES[index - 1]
    if previous != required:
        expected = "no checkpoint" if required is None else f"an {required.value} checkpoint"
        actual = "no checkpoint" if previous is None else f"an {previous.value} checkpoint"
        raise RuntimeError(f"{stage.value} requires {expected}, got {actual}")


def trainable_components(
    stage: TrainingStage,
    *,
    freeze_static_representation: bool = False,
) -> frozenset[str]:
    if stage is TrainingStage.INTERACTION_PRIOR and freeze_static_representation:
        return frozenset({
            "task_structure_encoder",
            "shared_feature_encoder",
            "inner_sac",
        })
    return {
        TrainingStage.INTERACTION_PRIOR: frozenset({"map_encoder", "interaction_encoder", "task_structure_encoder", "shared_feature_encoder", "inner_sac"}),
        TrainingStage.CONTEXT_META: frozenset({"episode_token_builder", "context_encoder", "outcome_decoder", "task_structure_encoder", "shared_feature_encoder", "inner_sac"}),
        TrainingStage.OUTER: frozenset({"task_structure_encoder", "universal_scene_policy"}),
    }[stage]
