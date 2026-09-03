from __future__ import annotations

import numpy as np
import pytest

from mvr.control import FrenetSACAdversaryController
from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.scenario.registry import load_adapters
from mvr.scenario.semantics import (
    ScenarioActionAdapter,
    ScenarioSemanticMonitor,
    quintic_smoothstep,
    quintic_smoothstep_derivative,
    quintic_smoothstep_second_derivative,
)
from mvr.state import INNER_STATE_FIELDS, PhysicalStateExtractor
from mvr.scenario.taskbook import load_taskbook
from mvr.safety.dynamics import CUTIN_VEHICLE_CONFIG
from mvr.safety import TrafficActionShield
from mvr.training.runner import HierarchicalRunner


def _episode(candidate_index: int = 0):
    task = next(
        task
        for task in load_taskbook("mvr/configs/taskbook.json")
        if task.task_id == "cutin-g04-fast_small_gap-cutin_interaction_core"
    )
    return ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(
            candidate_index,
            (0.0,) * 5,
        ),
        episode_seed=204 + candidate_index,
    )


@pytest.mark.parametrize("candidate_index", (0, 1))
def test_cutin_contract_uses_one_long_legal_corridor(candidate_index: int) -> None:
    episode = _episode(candidate_index)
    try:
        contract = episode.layout.traffic_contract
        merge_start, merge_end = contract.merge_window_m
        assert episode.adversary_route.length_m >= 200.0
        assert episode.sut_route.length_m >= 200.0
        assert abs(contract.target_lane_number - contract.source_lane_number) == 1
        assert contract.crossing_boundary == "broken"
        assert merge_start >= 0.0
        assert merge_end <= episode.adversary_route.length_m
        assert merge_end - merge_start >= 60.0
    finally:
        episode.env.close()


def test_route_block_has_no_adversarial_option() -> None:
    assert not hasattr(mvr_parameter_spaces()["cutin"], "options")


def test_frenet_sac_action_uses_full_jerk_limited_longitudinal_control() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule.update()
        controller = FrenetSACAdversaryController(episode, "cutin", schedule)
        try:
            target = np.asarray((0.25, -0.5, 0.75, -0.5), dtype=np.float32)
            planner_action, action = controller.action(target)
        finally:
            controller.destroy()
        np.testing.assert_allclose(planner_action[:3], 0.0)
        assert action[0] == pytest.approx(0.0)
        assert action[1] == pytest.approx(-0.15 / 6.0)
    finally:
        episode.env.close()


def test_cutin_agent_force_limits_match_the_physical_contract() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        shield = TrafficActionShield(episode, schedule)
        acceleration, deceleration = shield._longitudinal_limits()
        assert episode.adversary.config["max_engine_force"] == CUTIN_VEHICLE_CONFIG["max_engine_force"]
        assert episode.sut.config["max_engine_force"] == CUTIN_VEHICLE_CONFIG["max_engine_force"]
        assert episode.adversary.config["max_brake_force"] == CUTIN_VEHICLE_CONFIG["max_brake_force"]
        assert episode.sut.config["max_brake_force"] == CUTIN_VEHICLE_CONFIG["max_brake_force"]
        assert getattr(episode.env.engine.get_policy(episode.sut.id), "action_projector") is not None
        assert acceleration == pytest.approx(3.0)
        assert deceleration == pytest.approx(6.0)
    finally:
        episode.env.close()


def test_before_onset_shield_releases_no_lateral_action() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule.update()
        shielded = TrafficActionShield(episode, schedule).project(
            np.asarray((1.0, -1.0), dtype=np.float32)
        )
        assert shielded.action[0] == pytest.approx(0.0)
        assert shielded.action[1] == pytest.approx(-1.0)
        assert shielded.rejection_reason == "before_cutin_onset"
    finally:
        episode.env.close()


def test_replay_action_is_the_declared_sac_actuator_target() -> None:
    episode = _episode()
    try:
        rollout = HierarchicalRunner(max_steps=1).rollout(
            episode,
            "cutin",
            lambda _: np.asarray((1.0, -1.0, 1.0, -1.0), dtype=np.float32),
        )
        transition = rollout.transitions[0]
        np.testing.assert_allclose(
            transition["raw_policy_action"], (1.0, -1.0, 1.0, -1.0)
        )
        assert transition["planner_action"].shape == (4,)
        assert transition["executed_vehicle_action"].shape == (2,)
    finally:
        episode.env.close()


def test_maximum_direct_braking_stays_within_physical_envelope() -> None:
    episode = _episode()
    try:
        rollout = HierarchicalRunner(max_steps=130).rollout(
            episode,
            "cutin",
            lambda _: np.asarray((0.0, 0.0, 0.0, -1.0), dtype=np.float32),
        )
        telemetry = rollout.transitions[-1]["info"]
        assert telemetry["traffic_violation_counts"] == {}
        assert telemetry["traffic_max_abs_acceleration_mps2"] <= 6.0 + 1e-3
        assert telemetry["traffic_max_abs_jerk_mps3"] <= 6.0 + 1e-3
    finally:
        episode.env.close()


def test_direct_cutin_steering_stays_inside_road_corridor() -> None:
    episode = _episode()
    try:
        rollout = HierarchicalRunner(max_steps=130).rollout(
            episode,
            "cutin",
            lambda _: np.asarray((0.5, 1.0, -1.0, 0.0), dtype=np.float32),
        )
        telemetry = rollout.transitions[-1]["info"]
        assert not telemetry["adversary_out_of_road"]
        assert "out_of_road" not in telemetry["traffic_violation_counts"]
        assert telemetry["traffic_cutin_lateral_corridor_m"][0] < telemetry[
            "traffic_cutin_lateral_m"
        ] < telemetry["traffic_cutin_lateral_corridor_m"][1]
    finally:
        episode.env.close()


def test_cutin_onset_is_fixed_logical_scenario_parameter() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule._elapsed_seconds = lambda: 0.0
        schedule.update()
        assert not schedule.state.maneuver_latched
        schedule._elapsed_seconds = lambda: 100.0
        schedule.update()
        assert schedule.state.maneuver_latched
        monitor = ScenarioSemanticMonitor(episode, "cutin", schedule)
        assert not monitor.info()["event_semantic_valid"]
        extractor = PhysicalStateExtractor()
        extractor.reset(
            episode.env,
            episode.layout,
            episode.adversary_route,
            episode.sut_route,
        )
        state = extractor(episode.adversary, episode.sut, schedule)
        assert state.shape == (PhysicalStateExtractor.dimension,)
        assert state[INNER_STATE_FIELDS.index("maneuver_started")] == 1.0
    finally:
        episode.env.close()


def test_cutin_reference_previews_curve_speed_before_spatial_onset() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule._elapsed_seconds = lambda: 100.0
        schedule.update()

        reference = schedule.maneuver_reference()

        assert reference.progress == pytest.approx(0.0)
        assert reference.speed_limit_mps < episode.layout.traffic_contract.speed_limit_mps
    finally:
        episode.env.close()


def test_quintic_reference_has_zero_endpoint_slope_and_acceleration() -> None:
    assert float(quintic_smoothstep(0.0)) == pytest.approx(0.0)
    assert float(quintic_smoothstep(1.0)) == pytest.approx(1.0)
    for endpoint in (0.0, 1.0):
        assert float(quintic_smoothstep_derivative(endpoint)) == pytest.approx(0.0)
        assert float(quintic_smoothstep_second_derivative(endpoint)) == pytest.approx(0.0)


def test_reference_path_tracks_with_bounded_direct_longitudinal_command() -> None:
    episode = _episode()
    observed_actions = []
    try:
        rollout = HierarchicalRunner(max_steps=120).rollout(
            episode,
            "cutin",
            lambda _: np.asarray((0.0, 0.0, 0.0, 0.35), dtype=np.float32),
            step_callback=lambda _episode, _step, info: observed_actions.append(
                (
                    np.asarray(info["traffic_requested_action"], dtype=float),
                    np.asarray(info["traffic_executed_action"], dtype=float),
                    info.get("traffic_shield_rejection_reason"),
                )
            ),
        )
    finally:
        episode.env.close()
    # The declared spatial reference, not native IDM, supplies legal lateral
    # geometry. The direct longitudinal command keeps the adversary moving;
    # it may steer only after onset.
    assert observed_actions
    assert all(np.isclose(requested[0], 0.0) for requested, _, _ in observed_actions[:15])
    assert any(not np.isclose(requested[0], 0.0) for requested, _, _ in observed_actions[40:])
    assert all(
        abs(float(row["info"].get("maneuver_reference_lateral_error_m", 0.0))) < 3.5
        for row in rollout.transitions
    )
    assert rollout.transitions[-1]["info"]["semantic_maneuver_completed"]


def test_cutin_intrusion_requires_vehicle_footprint_overlap() -> None:
    class StraightLane:
        width = 3.5
        length = 100.0

        @staticmethod
        def local_coordinates(position):
            return float(position[0]), float(position[1])

        @staticmethod
        def heading_theta_at(_longitudinal):
            return 0.0

    class Vehicle:
        LENGTH = 4.5
        WIDTH = 2.0
        heading_theta = 0.0

        def __init__(self, lateral: float) -> None:
            self.position = np.asarray((50.0, lateral), dtype=float)

    lane = StraightLane()
    assert ScenarioSemanticMonitor._vehicle_overlaps_lane_corridor(Vehicle(2.74), lane)
    assert not ScenarioSemanticMonitor._vehicle_overlaps_lane_corridor(Vehicle(2.76), lane)
