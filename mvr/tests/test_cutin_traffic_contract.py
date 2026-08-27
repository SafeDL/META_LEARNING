from __future__ import annotations

import numpy as np
import pytest

from mvr.control import NativeAdversaryBaseController
from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.option import AdversarialOption
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.scenario.registry import load_adapters
from mvr.scenario.semantics import ScenarioActionAdapter, ScenarioSemanticMonitor
from mvr.state import PhysicalStateExtractor
from mvr.scenario.taskbook import load_taskbook
from mvr.safety import TrafficActionShield
from mvr.training.runner import HierarchicalRunner


def _episode(candidate_index: int = 0):
    task = next(
        task
        for task in load_taskbook("mvr/configs/taskbook.json")
        if task.task_id == "cutin-g04-fast_small_gap"
    )
    return ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(
            candidate_index,
            (0.0, 0.0, 0.0, 0.0),
            AdversarialOption.APPROACH_CONFLICT,
        ),
        episode_seed=204 + candidate_index,
    )


@pytest.mark.parametrize("candidate_index", (0, 1))
def test_cutin_contract_uses_one_long_legal_corridor(candidate_index: int) -> None:
    episode = _episode(candidate_index)
    try:
        contract = episode.layout.traffic_contract
        merge_start, merge_end = contract.merge_window_s
        assert episode.adversary_route.length_m >= 200.0
        assert episode.sut_route.length_m >= 200.0
        assert abs(contract.target_lane_number - contract.source_lane_number) == 1
        assert contract.crossing_boundary == "broken"
        assert merge_start >= 40.0
        assert episode.adversary_route.length_m - merge_end >= 40.0
        assert merge_end - merge_start >= 20.0
    finally:
        episode.env.close()


def test_route_block_is_not_an_adversarial_option() -> None:
    assert {option.value for option in AdversarialOption} == {
        "approach_conflict",
        "yield_then_press",
        "gap_close",
    }


def test_zero_residual_preserves_native_nominal_action() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule.update(0.0, 0)
        controller = NativeAdversaryBaseController(episode, "cutin", schedule)
        try:
            base, candidate = controller.action(np.zeros(3, dtype=np.float32))
            shielded = TrafficActionShield(episode, schedule).project(base, candidate)
        finally:
            controller.destroy()
        assert np.allclose(candidate, base)
        assert np.allclose(shielded.action, base)
        assert shielded.rejection_reason is None
    finally:
        episode.env.close()


def test_options_change_nominal_longitudinal_intent_without_changing_route() -> None:
    episode = _episode()
    try:
        targets = []
        for option in AdversarialOption:
            schedule = ScenarioActionAdapter(episode, "cutin")
            controller = NativeAdversaryBaseController(episode, "cutin", schedule, option.value)
            try:
                controller.action(np.zeros(3, dtype=np.float32))
                targets.append(controller.policy.NORMAL_SPEED)
            finally:
                controller.destroy()
        assert targets[1] < targets[0] < targets[2]
        assert episode.adversary.navigation.current_lane.index[2] == (
            episode.layout.traffic_contract.source_lane_number
        )
    finally:
        episode.env.close()


def test_cutin_schedule_is_low_passed_and_latched() -> None:
    episode = _episode()
    try:
        schedule = ScenarioActionAdapter(episode, "cutin")
        schedule.update(1.0, 0)
        assert schedule.state.maneuver_update_mask
        assert 0.0 < schedule.state.timing_reference < 1.0
        schedule.update(-1.0, 1)
        assert not schedule.state.maneuver_update_mask
        assert schedule.state.timing_reference > 0.0
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
        assert state[-1] > 0.0
    finally:
        episode.env.close()


@pytest.mark.parametrize("family,task_id,continuous", (
    ("cutin", "cutin-g04-fast_small_gap", (0.0, 0.0, 0.0, 0.0)),
    ("merge", "merge-g04-fast_small_gap", (0.0, 0.0, 0.0, 0.0)),
    ("roundabout", "roundabout-g04-fast_small_gap", (0.0, 0.0, -1.0, -1.0)),
))
def test_timing_reference_orders_native_conflict_arrival_targets(
    family: str,
    task_id: str,
    continuous: tuple[float, float, float, float],
) -> None:
    task = next(task for task in load_taskbook("mvr/configs/taskbook.json") if task.task_id == task_id)
    targets = []
    for command in (1.0, 0.0, -1.0):
        episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
            task,
            NormalizedScenarioAction(0, continuous, AdversarialOption.APPROACH_CONFLICT),
            episode_seed=204,
        )
        try:
            schedule = ScenarioActionAdapter(episode, family)
            schedule.update(command, 0)
            controller = NativeAdversaryBaseController(episode, family, schedule)
            try:
                controller.action(np.zeros(3, dtype=np.float32))
                targets.append(controller.policy.NORMAL_SPEED)
            finally:
                controller.destroy()
        finally:
            episode.env.close()
    assert targets[0] > targets[1] > targets[2]


def test_cutin_event_requires_physical_target_lane_intrusion() -> None:
    episode = _episode()
    observed_intrusion = []
    try:
        rollout = HierarchicalRunner(max_steps=60).rollout(
            episode,
            "cutin",
            AdversarialOption.APPROACH_CONFLICT.value,
            lambda _: np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
            step_callback=lambda _episode, _step, info: observed_intrusion.append(
                bool(info["semantic_target_lane_intrusion"])
            ),
        )
    finally:
        episode.env.close()
    assert any(observed_intrusion)
    # SUT speed is now scene-relative, so this fixed timing no longer promises
    # a collision.  The contract under test is physical target-lane intrusion.
    if rollout.outcome["event_kind"] is not None:
        assert rollout.outcome["event_semantic_valid"]
        assert rollout.outcome["event_traffic_valid"]
