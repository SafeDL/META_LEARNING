from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from meta_testing.model import HierarchicalMetaTester
from meta_testing.scenario.catalog import mvr_parameter_spaces
from meta_testing.scenario.task_spec import MetaTestTaskSpec
from meta_testing.scripts import training_cli
from meta_testing.scripts.training_cli import _save, checkpoint_config_hash
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


def test_checkpoint_compatibility_excludes_evaluation_and_keeps_provenance(tmp_path) -> None:
    config = {
        "seed": 11,
        "schema": "mvr_v1",
        "taskbook": "tasks.json",
        "model": {"state_dim": 10},
        "map": {"embedding_dim": 16},
        "inner": {"updates_per_episode": 8},
        "posterior": {"episode_budget": 20},
        "inner_calibration": {"updates_per_episode": 8},
        "outer": {"episode_budget": 20},
        "training": {"step_budget": 240},
        "evaluation": {"seeds": [11]},
    }
    evaluation_changed = {**config, "evaluation": {"seeds": [11, 22, 33]}}
    model_changed = {**config, "model": {"state_dim": 12}}
    assert checkpoint_config_hash(config) == checkpoint_config_hash(evaluation_changed)
    assert checkpoint_config_hash(config) != checkpoint_config_hash(model_changed)
    model = HierarchicalMetaTester({"merge_v1": mvr_parameter_spaces()["merge_v1"]}, state_dim=10, map_dim=16)
    path = tmp_path / "checkpoint.pt"
    _save(path, stage=TrainingStage.INNER_PRETRAIN.value, config=config, model=model, optimizer=None, progress={}, metrics={}, device=torch.device("cpu"))
    checkpoint = HierarchicalCheckpoint.load(path, expected_config_hash=checkpoint_config_hash(evaluation_changed))
    assert checkpoint.state["provenance_config"] == config


def test_default_inner_updates_per_episode_are_conservative() -> None:
    config = yaml.safe_load(Path("meta_testing/configs/mvr.yaml").read_text(encoding="utf-8"))
    assert config["inner"]["updates_per_episode"] == 8
    assert config["inner_calibration"]["updates_per_episode"] == 8


def test_posterior_collection_stays_at_the_prior(monkeypatch) -> None:
    calls = []

    class Online:
        def run(self, task, budget, **kwargs):
            calls.append((task, budget, kwargs))
            return SimpleNamespace(episodes=[])

    class PosteriorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def posterior_loss(self, batch) -> torch.Tensor:
            return self.weight.square()

    monkeypatch.setattr(training_cli, "_online", lambda *args: Online())
    monkeypatch.setattr(training_cli, "posterior_batch_from_episodes", lambda *args: object())
    task = MetaTestTaskSpec("task", "meta_train", "idm_cautious", "merge", "map", "a" * 64, "template", "merge_v1", 1)
    model = PosteriorModel()
    training_cli._train_posterior(
        model,
        [task],
        {"posterior": {"episode_budget": 2, "support_episodes": [1]}, "training": {"step_budget": 1}},
        torch.optim.Adam(model.parameters(), lr=1e-3),
    )
    assert calls[0][1:] == (2, {"posterior_support_limit": 0})
