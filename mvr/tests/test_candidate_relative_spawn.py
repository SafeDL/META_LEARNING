from __future__ import annotations

import pytest

from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.scenario.registry import load_adapters
from mvr.scenario.taskbook import load_taskbook


def test_candidate_relative_spawn_round_trips_the_outer_distance_controls() -> None:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.task_id == "roundabout-g04-fast_small_gap-interaction_core"
    )
    space = mvr_parameter_spaces()[task.functional_scenario]
    action = NormalizedScenarioAction(2, (-0.4, 0.6, 0.0, 0.0, 0.0))
    executor = ScenarioExecutor(load_adapters(), mvr_parameter_spaces())
    episode = executor.reset(task, action)
    try:
        scenario = episode.applied_scenario
        _, candidates = executor.enumerate_interactions(task)
        candidate = candidates[action.candidate_index]
        assert scenario.normalized_continuous == action.continuous
        assert scenario.adversary_distance_to_conflict_m == pytest.approx(
            candidate.adversary_distance_min_m + 0.3 * (
                candidate.adversary_distance_available_m - candidate.adversary_distance_min_m
            )
        )
        assert scenario.sut_distance_to_conflict_m == pytest.approx(
            candidate.sut_distance_min_m + 0.8 * (
                candidate.sut_distance_available_m - candidate.sut_distance_min_m
            )
        )
    finally:
        episode.env.close()
