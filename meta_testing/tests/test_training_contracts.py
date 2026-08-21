from __future__ import annotations

import pytest
from torch import nn

from meta_testing.training.replay import InnerReplay, InnerTransition
from meta_testing.training.stages import TrainingStage, trainable_components
from meta_testing.training.workflow import StagedWorkflow


def test_inner_replay_excludes_context_episodes() -> None:
    replay = InnerReplay()
    replay.add(InnerTransition("support", 1, 1, 1.0, 2, False))
    replay.add(InnerTransition("query", 1, 1, 1.0, 2, False))
    assert replay.sample(1, excluded_episode_ids={"support"})[0].episode_id == "query"
    with pytest.raises(ValueError):
        replay.sample(1, excluded_episode_ids={"support", "query"})


def test_stages_do_not_start_outer_and_inner_together() -> None:
    assert trainable_components(TrainingStage.INNER_PRETRAIN) == {"inner"}
    assert "outer" not in trainable_components(TrainingStage.POSTERIOR)


def test_workflow_freezes_components_by_stage() -> None:
    modules = {name: nn.Linear(1, 1) for name in ("inner", "outer", "trajectory", "posterior", "outcome_decoder")}
    workflow = StagedWorkflow(modules)
    workflow.activate(TrainingStage.OUTER)
    assert all(parameter.requires_grad for parameter in modules["outer"].parameters())
    assert not any(parameter.requires_grad for parameter in modules["inner"].parameters())
