from __future__ import annotations

import pytest
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy

from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.option import AdversarialOption
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.scenario.taskbook import load_taskbook
from mvr.scenario.registry import load_adapters
from mvr.training.runner import HierarchicalRunner


def test_initial_speed_bounds_follow_family_traffic_contract() -> None:
    spaces = mvr_parameter_spaces()
    assert spaces["merge"].bounds["adversary_initial_speed_mps"][1] == 18.0
    assert spaces["cutin"].bounds["adversary_initial_speed_mps"][1] == 20.0
    assert spaces["roundabout"].bounds["adversary_initial_speed_mps"][1] == 6.5
    for space in spaces.values():
        assert space.bounds["sut_initial_speed_mps"][1] == space.bounds[
            "adversary_initial_speed_mps"
        ][1]


@pytest.mark.parametrize("family", ("merge", "cutin", "roundabout"))
def test_runtime_geometry_hash_and_sut_attachment(family: str) -> None:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.functional_scenario == family and row.geometry_split == "train"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task, NormalizedScenarioAction(0, (0.0,) * 4, AdversarialOption.GAP_CLOSE), episode_seed=999
    )
    try:
        assert episode.map_tokens.map_hash == task.geometry_hash
        assert episode.episode_seed == 999
        assert episode.sut_adapter.metadata(episode.sut_profile)["profile_is_model_input"] is False
        policy = episode.env.engine.get_policy(episode.sut.id)
        assert isinstance(policy, IDMPolicy)
        assert not policy.enable_lane_change
    finally:
        episode.env.close()


@pytest.mark.parametrize("family", ("merge", "cutin", "roundabout"))
def test_idm_sut_stays_on_its_declared_route_centerline(family: str) -> None:
    """Lane-stable native SUT routes must not oscillate under a zero residual."""
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.functional_scenario == family and row.geometry_split == "train"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(0, (0.0,) * 4, AdversarialOption.GAP_CLOSE),
        episode_seed=999,
    )
    lateral_errors: list[float] = []

    def record_sut_tracking(current, _step, _info) -> None:
        projection = current.sut_route.projection(
            current.sut.position, current.sut.heading_theta
        )
        lateral_errors.append(abs(projection.lateral_m))

    try:
        HierarchicalRunner(max_steps=80).rollout(
            episode,
            family,
            AdversarialOption.GAP_CLOSE.value,
            lambda _state: np.zeros(3, dtype=np.float32),
            step_callback=record_sut_tracking,
        )
        assert lateral_errors
        assert max(lateral_errors) <= 0.60, (
            f"{family} SUT lateral errors: {np.round(lateral_errors, 3).tolist()}"
        )
    finally:
        episode.env.close()


def test_runtime_hash_mismatch_fails_fast() -> None:
    task = load_taskbook("mvr/configs/taskbook.json")[0]
    broken = type(task)(**{**task.to_dict(), "geometry_hash": "0" * 64})
    with pytest.raises(RuntimeError, match="map hash mismatch"):
        ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
            broken, NormalizedScenarioAction(0, (0.0,) * 4, AdversarialOption.GAP_CLOSE)
        )


def test_static_geometry_cache_avoids_rebuilding_layout_env(monkeypatch) -> None:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.functional_scenario == "merge" and row.geometry_split == "train"
    )
    adapter = load_adapters()["merge"]
    original_build_env = adapter.build_env
    layouts = []

    def counted_build_env(task, config, layout=None):
        layouts.append(layout)
        return original_build_env(task, config, layout)

    monkeypatch.setattr(adapter, "build_env", counted_build_env)
    executor = ScenarioExecutor({"merge": adapter}, mvr_parameter_spaces())
    action = NormalizedScenarioAction(0, (0.0,) * 4, AdversarialOption.GAP_CLOSE)

    executor.enumerate_interactions(task)
    first = executor.reset(task, action)
    cached_tokens = first.map_tokens
    first.env.close()
    second = executor.reset(task, action, episode_seed=999)
    try:
        assert layouts[0] is None
        assert all(layout is not None for layout in layouts[1:])
        assert len(layouts) == 3
        assert cached_tokens is second.map_tokens
    finally:
        second.env.close()


@pytest.mark.parametrize(
    ("candidate_index", "entry", "exit_", "minimum_segments"),
    ((0, 0, 1, 6), (1, 1, 2, 7), (2, 2, 0, 9)),
)
def test_roundabout_candidate_binds_idm_to_complete_entry_exit_route(
    candidate_index: int,
    entry: int,
    exit_: int,
    minimum_segments: int,
) -> None:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.task_id == "roundabout-g04-fast_small_gap"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(candidate_index, (0.0,) * 4, AdversarialOption.GAP_CLOSE),
        episode_seed=304 + candidate_index,
    )
    try:
        route = episode.layout.sut_route
        checkpoints = tuple([route[0][0], *(lane[1] for lane in route)])
        assert len(route) >= minimum_segments
        assert any(lane[1] == f"1O{entry}_0_" for lane in route)
        assert route[-1][0] == f"1O{exit_}_2_"
        assert tuple(episode.sut.navigation.checkpoints) == checkpoints
        assert episode.layout.native_navigation.sut_checkpoints == checkpoints
        assert episode.layout.native_navigation.sut_lane_stable
        assert isinstance(episode.env.engine.get_policy(episode.sut.id), IDMPolicy)
        assert not episode.env.engine.get_policy(episode.sut.id).enable_lane_change
    finally:
        episode.env.close()
