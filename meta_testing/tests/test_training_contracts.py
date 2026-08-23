from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from meta_testing.model import HierarchicalMetaTester
from meta_testing.scenario.catalog import mvr_parameter_spaces
from meta_testing.scenario.task_spec import MetaTestTaskSpec
from meta_testing.training.checkpoint import HierarchicalCheckpoint
from meta_testing.training.pipeline import (
    MVRTrainingPipeline,
    checkpoint_config_hash,
    taskbook_hash,
)
from meta_testing.training.replay import InnerReplay, InnerTransition
from meta_testing.training.stages import TrainingStage, trainable_components, validate_stage_transition
from meta_testing.training.trainers import train_posterior
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
    assert trainable_components(TrainingStage.INNER_PRETRAIN) == {
        "map_encoder", "shared_feature_encoder", "option_embedding", "inner_sac"
    }
    assert trainable_components(TrainingStage.INNER_LATENT_CALIBRATION) == {
        "shared_feature_encoder", "option_embedding", "inner_sac"
    }
    assert "scene_policies" not in trainable_components(TrainingStage.POSTERIOR)


def test_workflow_freezes_components_by_stage() -> None:
    modules = {
        name: nn.Linear(1, 1)
        for name in (
            "map_encoder", "shared_feature_encoder", "option_embedding", "inner_sac",
            "episode_token_builder", "posterior", "outcome_decoder", "scene_policies",
        )
    }
    workflow = StagedWorkflow(modules)
    workflow.activate(TrainingStage.OUTER)
    assert all(parameter.requires_grad for parameter in modules["scene_policies"].parameters())
    assert not any(parameter.requires_grad for parameter in modules["inner_sac"].parameters())
    workflow.activate(TrainingStage.INNER_LATENT_CALIBRATION)
    assert not any(parameter.requires_grad for parameter in modules["map_encoder"].parameters())
    assert all(parameter.requires_grad for parameter in modules["inner_sac"].parameters())


def test_stage_transition_requires_preceding_checkpoint() -> None:
    with pytest.raises(RuntimeError, match="inner_pretrain"):
        validate_stage_transition(TrainingStage.POSTERIOR, None)
    with pytest.raises(RuntimeError, match="inner_latent_calibration"):
        validate_stage_transition(TrainingStage.OUTER, TrainingStage.POSTERIOR)
    validate_stage_transition(TrainingStage.INNER_LATENT_CALIBRATION, TrainingStage.POSTERIOR)


def test_checkpoint_compatibility_excludes_evaluation_and_keeps_provenance(tmp_path) -> None:
    config = {
        "seed": 11,
        "schema": "mvr_v1",
        "taskbook": "tasks.json",
        "model": {"state_dim": 10},
        "map": {"embedding_dim": 16},
        "failure": {"severity_thresholds": {"ttc_s": 5.0, "distance_m": 10.0, "closing_speed_mps": 20.0}, "severity_bins": 5},
        "inner": {"updates_per_episode": 8},
        "posterior": {"episode_budget": 20},
        "inner_latent_calibration": {"updates_per_episode": 8},
        "outer": {"episodes_per_task": 20},
        "training": {"step_budget": 240},
        "evaluation": {"seeds": [11]},
    }
    evaluation_changed = {**config, "evaluation": {"seeds": [11, 22, 33]}}
    assert checkpoint_config_hash(config) == checkpoint_config_hash(evaluation_changed)
    model = HierarchicalMetaTester({"merge_v1": mvr_parameter_spaces()["merge_v1"]}, state_dim=10, map_dim=16)
    checkpoint = HierarchicalCheckpoint(
        HierarchicalCheckpoint.SCHEMA,
        TrainingStage.INNER_PRETRAIN.value,
        checkpoint_config_hash(config),
        {"model": model.state_dict(), "provenance_config": config},
    )
    path = tmp_path / "checkpoint.pt"
    checkpoint.save(path)
    restored = HierarchicalCheckpoint.load(path, expected_config_hash=checkpoint_config_hash(evaluation_changed))
    assert restored.state["provenance_config"] == config


def test_default_inner_updates_and_calibration_budget_are_explicit() -> None:
    config = yaml.safe_load(Path("meta_testing/configs/mvr.yaml").read_text(encoding="utf-8"))
    assert config["inner"]["updates_per_episode"] == 8
    assert config["inner_latent_calibration"]["updates_per_episode"] == 8
    assert config["inner_latent_calibration"]["simulator_episode_budget"] == 16
    assert config["outer"]["episodes_per_task"] == 20


def test_posterior_collection_stays_at_the_prior(monkeypatch) -> None:
    calls = []

    class Online:
        def run(self, task, budget, **kwargs):
            calls.append((task, budget, kwargs))
            return type("Result", (), {"episodes": []})()

    class PosteriorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def posterior_loss(self, batch) -> torch.Tensor:
            return self.weight.square()

    monkeypatch.setattr("meta_testing.training.trainers.build_online", lambda *args: Online())
    monkeypatch.setattr("meta_testing.training.trainers.posterior_batch_from_episodes", lambda *args: object())
    task = MetaTestTaskSpec("task", "meta_train", "idm_cautious", "merge", "map", "a" * 64, "template", "merge_v1", 1)
    model = PosteriorModel()
    from meta_testing.failure.criteria import DEFAULT_FAILURE_CRITERIA

    train_posterior(
        model,
        [task],
        {"posterior": {"episode_budget": 2, "support_episodes": [1]}, "training": {"step_budget": 1}},
        DEFAULT_FAILURE_CRITERIA,
        torch.optim.Adam(model.parameters(), lr=1e-3),
    )
    assert calls[0][1:] == (2, {"posterior_support_limit": 0})


def test_pipeline_records_stage_order(tmp_path, monkeypatch) -> None:
    config = yaml.safe_load(Path("meta_testing/configs/mvr.yaml").read_text(encoding="utf-8"))
    task = MetaTestTaskSpec("task", "meta_train", "idm_cautious", "merge", "map", "a" * 64, "template", "merge_v1", 1)
    pipeline = MVRTrainingPipeline(config, Path("meta_testing/configs/idm_taskbook.json"), torch.device("cpu"))
    monkeypatch.setattr("meta_testing.training.pipeline.selected_tasks", lambda *args: [task])
    monkeypatch.setattr(pipeline, "_train_stage", lambda stage, tasks, optimizer: {"simulator_episodes_consumed": 0})
    manifest = pipeline.run(tmp_path)
    stages = yaml.safe_load(manifest.read_text(encoding="utf-8"))["stages"]
    assert [record["stage"] for record in stages] == [stage.value for stage in (
        TrainingStage.INNER_PRETRAIN,
        TrainingStage.POSTERIOR,
        TrainingStage.INNER_LATENT_CALIBRATION,
        TrainingStage.OUTER,
    )]
    checkpoint = HierarchicalCheckpoint.load(tmp_path / "outer.pt")
    assert checkpoint.state["taskbook_hash"] == taskbook_hash("meta_testing/configs/idm_taskbook.json")
