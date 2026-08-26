from __future__ import annotations

import pytest

from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.option import AdversarialOption
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.scenario.registry import load_adapters
from mvr.scenario.taskbook import load_taskbook
from mvr.safety import TrafficActionShield


@pytest.mark.parametrize("candidate_index", (0, 1))
def test_cutin_contract_uses_one_long_legal_corridor(candidate_index: int) -> None:
    task = next(
        task
        for task in load_taskbook("mvr/configs/taskbook.json")
        if task.task_id == "cutin-g04-fast_small_gap"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(
            candidate_index,
            (0.0, 0.0, 0.0, 0.0),
            AdversarialOption.APPROACH_CONFLICT,
        ),
        episode_seed=204 + candidate_index,
    )
    try:
        contract = episode.layout.traffic_contract
        merge_start, merge_end = contract.merge_window_s
        assert episode.adversary_route.length_m >= 200.0
        assert episode.sut_route.length_m >= 200.0
        assert contract.target_lane_number == 1
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


def test_shield_rejects_a_lane_change_before_the_legal_window() -> None:
    task = next(
        task
        for task in load_taskbook("mvr/configs/taskbook.json")
        if task.task_id == "cutin-g04-fast_small_gap"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(
            0,
            (0.0, 0.0, 0.0, 0.0),
            AdversarialOption.APPROACH_CONFLICT,
        ),
        episode_seed=204,
    )
    try:
        shielded = TrafficActionShield(episode).project((1.0, 0.0))
        assert shielded.rejection_reason == "outside_merge_window"
    finally:
        episode.env.close()


def test_shield_projects_instead_of_replacing_sac_controls() -> None:
    task = next(
        task
        for task in load_taskbook("mvr/configs/taskbook.json")
        if task.task_id == "cutin-g04-fast_small_gap"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(
            0,
            (0.0, 0.0, 0.0, 0.0),
            AdversarialOption.APPROACH_CONFLICT,
        ),
        episode_seed=204,
    )
    try:
        shielded = TrafficActionShield(episode).project((0.3, 1.0))
        assert shielded.rejection_reason == "outside_merge_window"
        assert 0.0 < shielded.action[0] < 0.3
        assert shielded.action[1] > 0.0
    finally:
        episode.env.close()
