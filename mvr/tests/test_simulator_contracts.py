from __future__ import annotations

from dataclasses import replace

import pytest
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy

from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.scenario.semantics import ScenarioSemanticMonitor
from mvr.scenario.taskbook import load_taskbook
from mvr.scenario.registry import load_adapters
from mvr.training.runner import HierarchicalRunner
from mvr.sut.idm import LaneStableNativeIDMPolicy


def test_initial_speed_bounds_follow_family_traffic_contract() -> None:
    spaces = mvr_parameter_spaces()
    assert spaces["merge"].bounds["adversary_initial_speed_mps"][1] == 18.0
    assert spaces["cutin"].bounds["sut_initial_speed_mps"] == (7.0, 13.0)
    assert spaces["cutin"].bounds["relative_speed_mps"] == (-3.0, 1.0)
    assert spaces["roundabout"].bounds["adversary_initial_speed_mps"][1] == 6.5
    for family, space in spaces.items():
        if family == "cutin":
            continue
        assert space.bounds["sut_initial_speed_mps"][1] == space.bounds[
            "adversary_initial_speed_mps"
        ][1]


@pytest.mark.parametrize("family", ("merge", "cutin", "roundabout"))
def test_runtime_geometry_hash_and_sut_attachment(family: str) -> None:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.functional_scenario == family and row.geometry_split == "train"
    )
    space = mvr_parameter_spaces()[family]
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task, NormalizedScenarioAction(0, (0.0,) * space.continuous_dim), episode_seed=999
    )
    try:
        assert episode.map_tokens.map_hash == task.geometry_hash
        assert episode.episode_seed == 999
        assert episode.sut_adapter.metadata(episode.sut_profile)["profile_is_model_input"] is False
        policy = episode.env.engine.get_policy(episode.sut.id)
        assert isinstance(policy, LaneStableNativeIDMPolicy)
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
    space = mvr_parameter_spaces()[family]
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(0, (0.0,) * space.continuous_dim),
        episode_seed=999,
    )
    lateral_errors: list[float] = []
    lane_statuses: list[dict[str, object]] = []

    def record_sut_tracking(current, _step, info) -> None:
        projection = current.sut_route.projection(
            current.sut.position, current.sut.heading_theta
        )
        lateral_errors.append(abs(projection.lateral_m))
        lane_statuses.append(dict(info))

    try:
        HierarchicalRunner(max_steps=80).rollout(
            episode,
            family,
            lambda _state: np.zeros(4, dtype=np.float32),
            step_callback=record_sut_tracking,
        )
        assert lateral_errors
        assert np.sqrt(np.mean(np.square(lateral_errors))) <= 0.30
        assert max(lateral_errors) <= 0.60, (
            f"{family} SUT lateral errors: {np.round(lateral_errors, 3).tolist()}"
        )
        assert all(
            row["sut_current_lane"][2] == row["sut_expected_lane_number"]
            and row["sut_routing_target_lane"][2] == row["sut_expected_lane_number"]
            for row in lane_statuses
        )
    finally:
        episode.env.close()


def test_runtime_hash_mismatch_fails_fast() -> None:
    task = load_taskbook("mvr/configs/taskbook.json")[0]
    broken = type(task)(**{**task.to_dict(), "geometry_hash": "0" * 64})
    with pytest.raises(RuntimeError, match="map hash mismatch"):
        ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
                broken, NormalizedScenarioAction(0, (0.0,) * 5)
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
    action = NormalizedScenarioAction(0, (0.0,) * 5)

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


@pytest.mark.parametrize("candidate_index", (0, 1))
def test_merge_always_assigns_branch_entry_to_the_adversary(candidate_index: int) -> None:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.task_id == "merge-g04-fast_small_gap-interaction_core"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
            NormalizedScenarioAction(candidate_index, (0.0,) * 5),
        episode_seed=204 + candidate_index,
    )
    try:
        layout = episode.layout
        contract = layout.traffic_contract
        network = episode.env.current_map.road_network
        adversary_source_width = len(network.graph[layout.adversary_lane[0]][layout.adversary_lane[1]])
        sut_source_width = len(network.graph[layout.sut_lane[0]][layout.sut_lane[1]])
        assert contract.adversary_intent == "merge_from_branch"
        assert contract.sut_role == "lane_stable_mainline"
        assert adversary_source_width == 1
        assert sut_source_width >= 2
        assert layout.adversary_route[1] == layout.sut_route[1]
        assert len(layout.adversary_route) == len(layout.sut_route) + 1
        common_lane = network.get_lane(layout.adversary_route[1])
        assert network.get_lane(layout.adversary_lane).is_previous_lane_of(common_lane)
        assert network.get_lane(layout.sut_lane).is_previous_lane_of(common_lane)
        assert contract.completion_condition == "sut_route_destination"
        assert contract.terminate_on_target_collision
    finally:
        episode.env.close()


@pytest.mark.parametrize("candidate_index", (0, 1))
def test_merge_adversary_physically_enters_the_sut_downstream_lane(
    candidate_index: int,
) -> None:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.task_id == "merge-g04-fast_small_gap-interaction_core"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(candidate_index, (0.0,) * 5),
        episode_seed=204 + candidate_index,
    )
    shared_lane = tuple(episode.layout.adversary_route[1])
    shared_runtime_lane = episode.env.current_map.road_network.get_lane(shared_lane)
    adversary_enters_shared_lane = False
    both_overlap_shared_corridor = False
    semantic_challenge_seen = False

    def record_merge_runtime(current, _step, info) -> None:
        nonlocal adversary_enters_shared_lane, both_overlap_shared_corridor, semantic_challenge_seen
        adversary_enters_shared_lane |= tuple(current.adversary.lane.index) == shared_lane
        overlaps = (
            ScenarioSemanticMonitor._vehicle_overlaps_lane_corridor(
                current.adversary, shared_runtime_lane
            )
            and ScenarioSemanticMonitor._vehicle_overlaps_lane_corridor(
                current.sut, shared_runtime_lane
            )
        )
        both_overlap_shared_corridor |= overlaps
        semantic_challenge_seen |= bool(info["semantic_challenge_phase_active"])
        if overlaps:
            assert info["semantic_challenge_phase_active"]

    try:
        rollout = HierarchicalRunner(max_steps=80).rollout(
            episode,
            "merge",
            lambda _state: np.zeros(4, dtype=np.float32),
            step_callback=record_merge_runtime,
        )
        assert adversary_enters_shared_lane
        assert both_overlap_shared_corridor
        assert semantic_challenge_seen
        assert rollout.outcome["termination_reason"] in {
            "sut_route_completed", "runner_step_budget"
        }
    finally:
        episode.env.close()


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
        if row.task_id == "roundabout-g04-fast_small_gap-interaction_core"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
            NormalizedScenarioAction(candidate_index, (0.0,) * 5),
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
        assert isinstance(episode.env.engine.get_policy(episode.sut.id), LaneStableNativeIDMPolicy)
        assert not episode.env.engine.get_policy(episode.sut.id).enable_lane_change
    finally:
        episode.env.close()


def test_roundabout_entry_probe_preserves_physical_route_tracking() -> None:
    """The common tracker must not manufacture a semantic conflict."""
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.task_id == "roundabout-g04-fast_small_gap-interaction_core"
    )
    task = replace(
        task,
        logical_domain_id="semantic_probe",
        logical_domain_bounds={name: (-1.0, 1.0) for name in task.logical_domain_bounds},
    )
    action = NormalizedScenarioAction(
        0,
        (
            0.9514361619949341,
            -0.7125153541564941,
            -0.6000000238418579,
            -0.6000000238418579,
            0.0,
        ),
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task, action, episode_seed=task.geometry_seed + 1
    )
    try:
        rollout = HierarchicalRunner().rollout(
            episode,
            "roundabout",
            lambda _state: np.zeros(4, dtype=np.float32),
        )
        errors = [
            abs(float(row["info"]["maneuver_reference_lateral_error_m"]))
            for row in rollout.transitions
        ]
        assert np.sqrt(np.mean(np.square(errors))) <= 0.35
        assert not rollout.outcome["adversary_traffic_violation"]
        if rollout.outcome["event_kind"] is not None:
            assert rollout.outcome["event_semantic_valid"]
    finally:
        episode.env.close()
