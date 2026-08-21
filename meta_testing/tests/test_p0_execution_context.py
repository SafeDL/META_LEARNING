from __future__ import annotations

import numpy as np
import pytest
import torch

from meta_testing.context.outcome_schema import encode_outcome
from meta_testing.context.set_posterior import PosteriorTrainingBatch
from meta_testing.failure.signature import FailureSignatureBuilder
from meta_testing.map import tokenize_road_network
from meta_testing.model import HierarchicalMetaTester
from meta_testing.scenario.adapters.cutin import CutInScenarioAdapter
from meta_testing.scenario.adapters.merge import MergeScenarioAdapter
from meta_testing.scenario.adapters.roundabout import RoundaboutScenarioAdapter
from meta_testing.scenario.catalog import mvr_parameter_spaces
from meta_testing.scenario.executor import ScenarioExecutor
from meta_testing.scenario.option import AdversarialOption
from meta_testing.scenario.parameter_space import NormalizedScenarioAction
from meta_testing.scenario.task_spec import MetaTestTaskSpec
from meta_testing.training.runner import HierarchicalRunner


ADAPTERS = {"merge": MergeScenarioAdapter(), "cutin": CutInScenarioAdapter(), "roundabout": RoundaboutScenarioAdapter()}


def _task_with_runtime_hash(family: str, space_id: str, adapter: object) -> MetaTestTaskSpec:
    provisional = MetaTestTaskSpec(f"{family}-physical", "meta_train", "idm_cautious", family, f"{family}-map", "0" * 64, "template", space_id, 1)
    env = adapter.build_env(provisional, {})
    try:
        env.reset()
        actual_hash = tokenize_road_network(env.current_map.road_network).map_hash
    finally:
        env.close()
    return MetaTestTaskSpec(f"{family}-physical", "meta_train", "idm_cautious", family, f"{family}-map", actual_hash, "template", space_id, 1)


@pytest.mark.parametrize(("family", "space_id", "candidate"), (("merge", "merge_v1", 0), ("cutin", "cutin_v1", 1), ("roundabout", "roundabout_v1", 2)))
def test_executor_applies_outer_values_to_real_vehicles(family: str, space_id: str, candidate: int) -> None:
    spaces = mvr_parameter_spaces()
    task = _task_with_runtime_hash(family, space_id, ADAPTERS[family])
    action = NormalizedScenarioAction(candidate, (-0.7, -0.6, -0.5, -0.4), AdversarialOption.GAP_CLOSE)
    episode = ScenarioExecutor(ADAPTERS, spaces).reset(task, action)
    try:
        expected = spaces[space_id].decode(action)
        applied = episode.applied_scenario
        assert applied.selected_candidate == expected["route_or_conflict_candidate"]
        assert applied.selected_option == expected["option"]
        assert applied.adversary_spawn_m == expected["adversary_spawn_m"]
        assert applied.sut_spawn_m == expected["sut_spawn_m"]
        assert applied.adversary_speed_mps == expected["adversary_initial_speed_mps"]
        assert applied.sut_speed_mps == expected["sut_initial_speed_mps"]
        assert episode.sut_adapter.metadata(episode.sut_profile)["profile_is_model_input"] is False
    finally:
        episode.env.close()


def test_executor_rejects_runtime_map_hash_mismatch() -> None:
    task = MetaTestTaskSpec("bad-hash", "meta_train", "idm_cautious", "merge", "map", "0" * 64, "template", "merge_v1", 1)
    with pytest.raises(RuntimeError, match="map hash mismatch"):
        ScenarioExecutor(ADAPTERS, mvr_parameter_spaces()).reset(task, NormalizedScenarioAction(0, (-0.7,) * 4, AdversarialOption.GAP_CLOSE))


def test_idm_profile_uses_distance_and_time_headway_directly() -> None:
    task = _task_with_runtime_hash("merge", "merge_v1", ADAPTERS["merge"])
    episode = ScenarioExecutor(ADAPTERS, mvr_parameter_spaces()).reset(task, NormalizedScenarioAction(0, (-0.7,) * 4, AdversarialOption.GAP_CLOSE))
    try:
        policy = episode.env.engine.get_policy(episode.sut.id)
        assert policy.DISTANCE_WANTED == episode.sut_profile.distance_wanted_m
        assert policy.TIME_WANTED == episode.sut_profile.time_headway_s
    finally:
        episode.env.close()


def test_runner_collects_real_trajectory_features() -> None:
    task = _task_with_runtime_hash("merge", "merge_v1", ADAPTERS["merge"])
    episode = ScenarioExecutor(ADAPTERS, mvr_parameter_spaces()).reset(task, NormalizedScenarioAction(0, (-0.7,) * 4, AdversarialOption.GAP_CLOSE))
    try:
        signature = FailureSignatureBuilder().from_outcome({}, "merge", None)
        rollout = HierarchicalRunner(max_steps=2).rollout(episode, {}, "gap_close", lambda _: np.zeros(2, dtype=np.float32), lambda _: ({}, signature))
        assert rollout.trajectory.shape == (2, 12)
        assert torch.isfinite(rollout.trajectory).all()
        assert all("sut_evidence" in transition and "trajectory_features" in transition for transition in rollout.transitions)
    finally:
        episode.env.close()


def test_mixed_outcome_loss_and_leakage_guard() -> None:
    target = encode_outcome({"target_collision": True, "min_ttc": 1.0, "min_distance": 2.0, "max_closing_speed": 8.0})
    assert target.shape == (5,)
    batch = PosteriorTrainingBatch(torch.randn(1, 2, 128), torch.tensor([[True, False]]), (("support",),), ("target",), torch.randn(1, 16), torch.randn(1, 4), torch.zeros(1, dtype=torch.long), target.unsqueeze(0))
    batch.validate()
    leaking = PosteriorTrainingBatch(batch.support_tokens, batch.support_mask, (("target",),), ("target",), batch.target_map, batch.target_config, batch.target_option, batch.target_outcome)
    with pytest.raises(ValueError, match="must not appear"):
        leaking.validate()
    model = HierarchicalMetaTester({"merge_v1": mvr_parameter_spaces()["merge_v1"]}, state_dim=3, map_dim=16)
    loss = model.posterior_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_model_uses_posterior_in_outer_and_inner_paths() -> None:
    space = mvr_parameter_spaces()["merge_v1"]
    model = HierarchicalMetaTester({"merge_v1": space}, state_dim=3, map_dim=16)
    support = torch.randn(1, 2, 128)
    mean, _ = model.infer_posterior(support, torch.tensor([[True, True]]))
    prior, _ = model.posterior.prior(1)
    map_embedding = torch.randn(1, 16)
    scene_prior = model.select_scene("merge_v1", map_embedding, prior, deterministic=True)
    scene_posterior = model.select_scene("merge_v1", map_embedding, mean, deterministic=True)
    features_prior = model.inner_features(torch.randn(1, 3), map_embedding, prior, torch.zeros(1, dtype=torch.long), torch.zeros(1, 4))
    features_posterior = model.inner_features(torch.randn(1, 3), map_embedding, mean, torch.zeros(1, dtype=torch.long), torch.zeros(1, 4))
    assert not torch.allclose(mean, prior)
    assert not torch.allclose(scene_prior.value, scene_posterior.value)
    assert not torch.allclose(features_prior, features_posterior)
