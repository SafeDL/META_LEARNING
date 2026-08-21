from __future__ import annotations

import pytest
import torch
from torch import nn

from meta_testing.training.replay import InnerReplay, InnerTransition
from meta_testing.training.stages import TrainingStage, trainable_components
from meta_testing.training.workflow import StagedWorkflow


def test_inner_replay_excludes_context_episodes() -> None:
    replay = InnerReplay()
    context = (None, torch.zeros(1), torch.zeros((), dtype=torch.long), torch.zeros(1))
    replay.add(InnerTransition("support", 1, 1, 1.0, 2, False, *context))
    replay.add(InnerTransition("query", 1, 1, 1.0, 2, False, *context))
    assert replay.sample(1, excluded_episode_ids={"support"})[0].episode_id == "query"
    with pytest.raises(ValueError):
        replay.sample(1, excluded_episode_ids={"support", "query"})


def test_stages_do_not_start_outer_and_inner_together() -> None:
    assert trainable_components(TrainingStage.INNER_PRETRAIN) == {"map_encoder", "shared_feature_encoder", "option_embedding", "inner_sac"}
    assert "scene_policies" not in trainable_components(TrainingStage.POSTERIOR)


def test_workflow_freezes_components_by_stage() -> None:
    modules = {name: nn.Linear(1, 1) for name in ("map_encoder", "shared_feature_encoder", "option_embedding", "inner_sac", "episode_token_builder", "posterior", "outcome_decoder", "scene_policies")}
    workflow = StagedWorkflow(modules)
    workflow.activate(TrainingStage.OUTER)
    assert all(parameter.requires_grad for parameter in modules["scene_policies"].parameters())
    assert not any(parameter.requires_grad for parameter in modules["inner_sac"].parameters())
