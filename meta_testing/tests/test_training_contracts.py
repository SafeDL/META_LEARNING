from __future__ import annotations

import pytest
import torch
from torch import nn

from meta_testing.model import HierarchicalMetaTester
from meta_testing.scenario.catalog import mvr_parameter_spaces
from meta_testing.scripts.training_cli import _save
from meta_testing.training.checkpoint import HierarchicalCheckpoint
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
    assert trainable_components(TrainingStage.INNER_CALIBRATION) == {"shared_feature_encoder", "option_embedding", "inner_sac"}
    assert "scene_policies" not in trainable_components(TrainingStage.POSTERIOR)


def test_workflow_freezes_components_by_stage() -> None:
    modules = {name: nn.Linear(1, 1) for name in ("map_encoder", "shared_feature_encoder", "option_embedding", "inner_sac", "episode_token_builder", "posterior", "outcome_decoder", "scene_policies")}
    workflow = StagedWorkflow(modules)
    workflow.activate(TrainingStage.OUTER)
    assert all(parameter.requires_grad for parameter in modules["scene_policies"].parameters())
    assert not any(parameter.requires_grad for parameter in modules["inner_sac"].parameters())
    workflow.activate(TrainingStage.INNER_CALIBRATION)
    assert not any(parameter.requires_grad for parameter in modules["map_encoder"].parameters())
    assert all(parameter.requires_grad for parameter in modules["inner_sac"].parameters())


def test_inner_replay_is_persisted_with_training_checkpoint(tmp_path) -> None:
    model = HierarchicalMetaTester({"merge_v1": mvr_parameter_spaces()["merge_v1"]}, state_dim=10, map_dim=16)
    replay = InnerReplay(rows=[InnerTransition("episode", 1, 1, 1.0, 2, True, None, torch.zeros(16), torch.zeros((), dtype=torch.long), torch.zeros(4))])
    path = tmp_path / "inner.pt"
    _save(path, stage=TrainingStage.INNER_PRETRAIN.value, config={"seed": 1}, model=model, optimizer=None, progress={"episodes": 1}, metrics={}, device=torch.device("cpu"), inner_replay=replay)
    checkpoint = HierarchicalCheckpoint.load(path)
    assert len(checkpoint.state["inner_replay"]) == 1
