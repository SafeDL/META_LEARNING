from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from metadrive_benchmark.control import FrenetSACAdversaryController
from metadrive_benchmark.physical_limits import CUTIN_LATERAL_ACCELERATION_LIMIT_MPS2
from metadrive_benchmark.scenario.catalog import (
    CUTIN_REFERENCE_LATERAL_ACCELERATION_MPS2,
    mvr_parameter_spaces,
)
from metadrive_benchmark.scenario.executor import ScenarioExecutor
from metadrive_benchmark.scenario.parameter_space import NormalizedScenarioAction
from metadrive_benchmark.scenario.registry import load_adapters
from metadrive_benchmark.scenario.semantics import (
    ScenarioActionAdapter,
    ScenarioSemanticMonitor,
    SemanticState,
    quintic_smoothstep,
    quintic_smoothstep_derivative,
    quintic_smoothstep_second_derivative,
)
from metadrive_benchmark.scenario.frenet import REFERENCE_LATERAL_ACCELERATION_MPS2
from metadrive_benchmark.state import INNER_STATE_FIELDS, PhysicalStateExtractor
from metadrive_benchmark.scenario.taskbook import load_taskbook
from metadrive_benchmark.safety.dynamics import (
    CUTIN_MAX_LATERAL_ACCELERATION_MPS2,
    CUTIN_VEHICLE_CONFIG,
)
from metadrive_benchmark.safety import TrafficActionShield
from metadrive_benchmark.training.runner import HierarchicalRunner


def _episode(candidate_index: int = 0, *, environment_overrides=None):
    task = next(task for task in load_taskbook("metadrive_benchmark/configs/taskbook.json")
                if task.task_id == "cutin-g04-fast_small_gap-cutin_interaction_core")
    return ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(
            candidate_index,
            (0.0, ) * 5,
        ),
        episode_seed=204 + candidate_index,
        environment_overrides=environment_overrides,
    )


def test_cutin_lateral_acceleration_limit_is_consistent_across_layers() -> None:
    assert CUTIN_LATERAL_ACCELERATION_LIMIT_MPS2 == pytest.approx(0.6 * 9.80665)
    assert CUTIN_REFERENCE_LATERAL_ACCELERATION_MPS2 == pytest.approx(
        CUTIN_LATERAL_ACCELERATION_LIMIT_MPS2)
    assert REFERENCE_LATERAL_ACCELERATION_MPS2 == pytest.approx(
        CUTIN_REFERENCE_LATERAL_ACCELERATION_MPS2)
    assert CUTIN_MAX_LATERAL_ACCELERATION_MPS2 == pytest.approx(
        CUTIN_REFERENCE_LATERAL_ACCELERATION_MPS2)


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
            control = controller.action(target)
        finally:
            controller.destroy()
        np.testing.assert_allclose(control.planner_action[:3], 0.0)
        assert control.raw_vehicle_action[0] == pytest.approx(0.0)
        assert control.projected_vehicle_action[1] == pytest.approx(-0.15 / 6.0)
    finally:
        episode.env.close()


def test_cutin_agent_force_limits_match_the_physical_contract() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        shield = TrafficActionShield(episode, schedule)
        acceleration, deceleration = shield._longitudinal_limits()
        assert episode.adversary.config["max_engine_force"] == CUTIN_VEHICLE_CONFIG[
            "max_engine_force"]
        assert episode.sut.config["max_engine_force"] == CUTIN_VEHICLE_CONFIG["max_engine_force"]
        assert episode.adversary.config["max_brake_force"] == CUTIN_VEHICLE_CONFIG[
            "max_brake_force"]
        assert episode.sut.config["max_brake_force"] == CUTIN_VEHICLE_CONFIG["max_brake_force"]
        assert getattr(episode.env.engine.get_policy(episode.sut.id),
                       "action_projector") is not None
        assert acceleration == pytest.approx(3.0)
        assert deceleration == pytest.approx(6.0)
        assert shield.max_lateral_acceleration_mps2 == pytest.approx(
            CUTIN_MAX_LATERAL_ACCELERATION_MPS2)
    finally:
        episode.env.close()


def test_before_onset_shield_releases_no_lateral_action() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule.update()
        shielded = TrafficActionShield(episode,
                                       schedule).project(np.asarray((1.0, -1.0), dtype=np.float32))
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
        np.testing.assert_allclose(transition["raw_policy_action"], (1.0, -1.0, 1.0, -1.0))
        assert transition["planner_action"].shape == (4, )
        assert transition["executed_vehicle_action"].shape == (2, )
    finally:
        episode.env.close()


def test_inner_policy_action_is_held_between_planner_decisions() -> None:
    episode = _episode()
    calls = 0

    def policy(_state):
        nonlocal calls
        calls += 1
        return np.asarray((0.05 * calls, 0.0, 0.0, 0.0), dtype=np.float32)

    try:
        rollout = HierarchicalRunner(max_steps=60).rollout(episode, "cutin", policy)
    finally:
        episode.env.close()

    decisions = [
        index for index, row in enumerate(rollout.transitions)
        if row["info"]["inner_policy_decision"]
    ]
    assert calls == len(decisions)
    assert decisions[0] == 0
    for previous, row in zip(rollout.transitions, rollout.transitions[1:]):
        if not row["info"]["inner_policy_decision"]:
            np.testing.assert_allclose(row["raw_policy_action"], previous["raw_policy_action"])

    inactive_decisions = [
        index for index in decisions if not rollout.transitions[index]["info"]["planner_active"]
    ]
    active_decisions = [index for index in decisions if index not in inactive_decisions]
    assert all(index % 5 == 0 for index in inactive_decisions)
    assert active_decisions
    assert all(right - left == 5 for left, right in zip(active_decisions, active_decisions[1:]))


def test_step_action_is_called_each_tick_without_changing_inner_action_holding() -> None:
    episode = _episode()
    phases = []

    def policy(phase):
        phases.append(phase)
        return np.asarray((0.2, 0.0, 0.0, 0.0), dtype=np.float32)

    try:
        rollout = HierarchicalRunner(max_steps=12).rollout(episode, "cutin", step_action=policy)
    finally:
        episode.env.close()

    assert len(phases) == len(rollout.transitions)
    assert all(row["info"]["inner_policy_decision"] for row in rollout.transitions)
    assert all(row["raw_policy_action"].shape == (4, ) for row in rollout.transitions)


def test_runner_rejects_missing_or_ambiguous_action_provider() -> None:
    runner = HierarchicalRunner()
    with pytest.raises(ValueError, match="exactly one"):
        runner.rollout(None, "cutin")
    with pytest.raises(ValueError, match="exactly one"):
        runner.rollout(None, "cutin", lambda _: np.zeros(4), step_action=lambda _: np.zeros(4))


def test_simulator_truncation_precedes_runner_step_budget() -> None:
    episode = _episode(environment_overrides={"horizon": 1})
    try:
        rollout = HierarchicalRunner(max_steps=300).rollout(
            episode,
            "cutin",
            lambda _state: np.zeros(4, dtype=np.float32),
        )
    finally:
        episode.env.close()

    assert len(rollout.transitions) == 1
    assert rollout.transitions[-1]["info"]["termination_reason"] == ("simulator_truncated")


def test_maximum_direct_braking_stays_within_physical_envelope() -> None:
    episode = _episode()
    try:
        rollout = HierarchicalRunner(max_steps=130).rollout(
            episode,
            "cutin",
            lambda _: np.asarray((0.0, 0.0, 0.0, -1.0), dtype=np.float32),
        )
        telemetry = rollout.transitions[-1]["info"]
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
        assert telemetry["traffic_cutin_lateral_corridor_m"][0] < telemetry[
            "traffic_cutin_lateral_m"] < telemetry["traffic_cutin_lateral_corridor_m"][1]
    finally:
        episode.env.close()


def test_cutin_onset_is_fixed_spatial_scenario_parameter() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule._elapsed_seconds = lambda: 0.0
        schedule.update()
        assert not schedule.state.maneuver_latched
        schedule._elapsed_seconds = lambda: 100.0
        schedule.update()
        assert not schedule.state.maneuver_latched
        monitor = ScenarioSemanticMonitor(episode, "cutin", schedule)
        assert not monitor.info()["event_semantic_valid"]
        extractor = PhysicalStateExtractor()
        extractor.reset(
            episode.env,
            episode.layout,
            episode.adversary_route,
            episode.sut_route,
        )
        state = extractor(
            episode.adversary,
            episode.sut,
            schedule,
            valid_near_miss_seen=False,
        )
        assert state.shape == (PhysicalStateExtractor.dimension, )
        assert state[INNER_STATE_FIELDS.index("maneuver_started")] == 0.0
    finally:
        episode.env.close()


def test_invalid_near_miss_does_not_occupy_the_event_latch() -> None:
    monitor = ScenarioSemanticMonitor(SimpleNamespace(), "cutin", SimpleNamespace())

    assert not monitor.capture_event("near_miss", {})
    assert monitor.info()["event_kind"] is None
    assert not monitor.info()["event_just_captured"]

    monitor._state = SemanticState(True, True, False, True, True, True)
    assert monitor.capture_event("near_miss", {})
    assert monitor.info()["event_semantic_valid"]
    assert monitor.info()["event_execution_valid"]
    assert monitor.valid_near_miss_seen
    assert not monitor.capture_event("near_miss", {})
    assert monitor.capture_event("collision", {})
    assert monitor.info()["event_kind"] == "collision"


def test_execution_invalid_near_miss_does_not_occupy_the_event_latch() -> None:
    monitor = ScenarioSemanticMonitor(SimpleNamespace(), "cutin", SimpleNamespace())
    monitor._state = SemanticState(True, True, False, True, True, True)

    assert not monitor.capture_event("near_miss", {"wrong_route": True})
    assert monitor.info()["event_kind"] is None
    assert not monitor.valid_near_miss_seen


def test_runner_next_state_matches_the_following_actor_state() -> None:
    episode = _episode()
    try:
        rollout = HierarchicalRunner(max_steps=60).rollout(
            episode,
            "cutin",
            lambda _state: np.zeros(4, dtype=np.float32),
        )
    finally:
        episode.env.close()

    assert len(rollout.transitions) > 1
    started = INNER_STATE_FIELDS.index("maneuver_started")
    assert any(row["state"][started] == 0.0 for row in rollout.transitions)
    assert any(row["state"][started] == 1.0 for row in rollout.transitions)
    for previous, following in zip(rollout.transitions, rollout.transitions[1:]):
        np.testing.assert_allclose(previous["next_state"], following["state"])
    assert all("raw_near_miss_candidate" in row["info"] for row in rollout.transitions)


def test_inner_state_includes_projector_acceleration_and_vehicle_steering() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule.update()
        extractor = PhysicalStateExtractor()
        extractor.reset(episode.env, episode.layout, episode.adversary_route, episode.sut_route)
        controller = FrenetSACAdversaryController(episode, "cutin", schedule)
        try:
            initial = extractor(
                episode.adversary,
                episode.sut,
                schedule,
                controller.actuator_state(),
                valid_near_miss_seen=False,
            )
            controller.action(np.asarray((0.0, 0.0, 0.0, -1.0), dtype=np.float32))
            updated = extractor(
                episode.adversary,
                episode.sut,
                schedule,
                controller.actuator_state(),
                valid_near_miss_seen=True,
            )
        finally:
            controller.destroy()

        acceleration = INNER_STATE_FIELDS.index("executed_longitudinal_acceleration")
        steering = INNER_STATE_FIELDS.index("executed_steering")
        near_miss_seen = INNER_STATE_FIELDS.index("valid_near_miss_seen")
        assert initial.shape == (31, )
        assert initial[acceleration] == pytest.approx(0.0)
        assert initial[near_miss_seen] == pytest.approx(0.0)
        assert updated[acceleration] == pytest.approx(-0.15 / 6.0)
        assert updated[steering] == pytest.approx(float(episode.adversary.steering))
        assert updated[near_miss_seen] == pytest.approx(1.0)

        rollout = HierarchicalRunner(max_steps=1).rollout(
            episode,
            "cutin",
            lambda _: np.asarray((0.0, 0.0, 0.0, -1.0), dtype=np.float32),
        )
        assert rollout.transitions[0]["state"][acceleration] == pytest.approx(0.0)
        assert rollout.transitions[0]["next_state"][acceleration] == pytest.approx(-0.15 / 6.0)
        assert rollout.transitions[0]["next_state"][steering] == pytest.approx(0.0)
    finally:
        episode.env.close()


def test_cutin_reference_uses_road_speed_before_spatial_onset() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule._elapsed_seconds = lambda: 100.0
        schedule.update()

        reference = schedule.maneuver_reference()

        assert reference.progress == pytest.approx(0.0)
        assert reference.speed_limit_mps == episode.layout.traffic_contract.speed_limit_mps
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
                {
                    "requested": np.asarray(info["traffic_requested_action"], dtype=float),
                    "start_remaining_m": float(info["maneuver_start_remaining_m"]),
                }),
        )
    finally:
        episode.env.close()
    # The declared spatial reference, not native IDM, supplies legal lateral
    # geometry. The direct longitudinal command keeps the adversary moving;
    # it may steer only after onset.
    assert observed_actions
    assert all(
        np.isclose(row["requested"][0], 0.0) for row in observed_actions
        if row["start_remaining_m"] > 0.0)
    assert any(not np.isclose(row["requested"][0], 0.0) for row in observed_actions
               if row["start_remaining_m"] <= 0.0)
    assert all(
        abs(float(row["info"].get("maneuver_reference_lateral_error_m", 0.0))) < 3.5
        for row in rollout.transitions)
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
