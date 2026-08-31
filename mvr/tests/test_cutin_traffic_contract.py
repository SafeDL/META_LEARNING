from __future__ import annotations

import numpy as np
import pytest

from mvr.control import DirectSACAdversaryController
from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.scenario.registry import load_adapters
from mvr.scenario.semantics import ScenarioActionAdapter, ScenarioSemanticMonitor
from mvr.state import PhysicalStateExtractor
from mvr.scenario.taskbook import load_taskbook
from mvr.safety.dynamics import CUTIN_VEHICLE_CONFIG
from mvr.safety import TrafficActionShield
from mvr.training.runner import HierarchicalRunner


def _episode(candidate_index: int = 0):
    task = next(
        task
        for task in load_taskbook("mvr/configs/taskbook.json")
        if task.task_id == "cutin-g04-fast_small_gap-interaction_core"
    )
    return ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(
            candidate_index,
            (0.0, 0.0, 0.0, 0.0),
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
        assert merge_start >= 40.0
        assert episode.adversary_route.length_m - merge_end >= 40.0
        assert merge_end - merge_start >= 20.0
    finally:
        episode.env.close()


def test_route_block_has_no_adversarial_option() -> None:
    assert not hasattr(mvr_parameter_spaces()["cutin"], "options")


def test_direct_sac_action_is_not_composed_with_a_nominal_controller() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule.update()
        controller = DirectSACAdversaryController(episode, "cutin", schedule)
        try:
            requested = np.asarray((0.25, -0.5), dtype=np.float32)
            action = controller.action(requested)
        finally:
            controller.destroy()
        assert np.allclose(action, requested)
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
        assert shielded.action[1] == pytest.approx(-2.0 / 60.0)
        assert shielded.rejection_reason == "before_cutin_onset"
    finally:
        episode.env.close()


def test_maximum_direct_braking_stays_within_physical_envelope() -> None:
    episode = _episode()
    try:
        rollout = HierarchicalRunner(max_steps=60).rollout(
            episode,
            "cutin",
            lambda _: np.asarray((0.0, -1.0), dtype=np.float32),
        )
        telemetry = rollout.transitions[-1]["info"]
        assert telemetry["traffic_violation_counts"] == {}
        assert telemetry["traffic_max_abs_acceleration_mps2"] <= 6.0 + 1e-3
        assert telemetry["traffic_max_abs_jerk_mps3"] <= 4.0 + 1e-3
    finally:
        episode.env.close()


def test_direct_cutin_steering_stays_inside_road_corridor() -> None:
    episode = _episode()
    try:
        rollout = HierarchicalRunner(max_steps=130).rollout(
            episode,
            "cutin",
            lambda _: np.asarray((0.5, 0.0), dtype=np.float32),
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
        state = extractor(episode.adversary, episode.sut, schedule.state)
        assert state.shape == (PhysicalStateExtractor.dimension,)
        assert state[-1] == 1.0
    finally:
        episode.env.close()


def test_zero_direct_action_does_not_create_an_implicit_cutin() -> None:
    episode = _episode()
    observed_actions = []
    try:
        rollout = HierarchicalRunner(max_steps=60).rollout(
            episode,
            "cutin",
            lambda _: np.asarray((0.0, 0.0), dtype=np.float32),
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
    # Direct control makes a zero SAC action a true coasting/no-steering
    # command.  In particular, no nominal IDM lane-change may be hidden
    # underneath the policy output.
    assert observed_actions
    for requested, executed, reason in observed_actions:
        np.testing.assert_allclose(requested, (0.0, 0.0), atol=1e-7)
        # A non-zero steering command is permitted only when the physical
        # shield is actively preventing a lane-corridor excursion; it cannot
        # come from a hidden nominal IDM controller.
        if not np.isclose(executed[0], 0.0, atol=1e-7):
            assert reason in {"lateral_corridor", "steering_rate"}
    np.testing.assert_allclose(
        rollout.transitions[0]["action"],
        np.asarray((0.0, 0.0), dtype=np.float32),
    )
    # A fixed SUT can still enter the geometric near-miss band while the
    # adversary coasts; that evidence must not be mistaken for an implicit
    # Cut-in maneuver.
    assert rollout.outcome["event_kind"] in {None, "near_miss"}
    assert not rollout.outcome["event_semantic_valid"]


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
