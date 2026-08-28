from __future__ import annotations

import pytest

from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.option import AdversarialOption
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.scenario.taskbook import load_taskbook
from mvr.scenario.registry import load_adapters


@pytest.mark.parametrize("geometry_id", ("merge-g01", "merge-g02"))
def test_distance_to_conflict_is_preserved_on_distinct_geometries(geometry_id: str) -> None:
    task = next(task for task in load_taskbook("mvr/configs/taskbook.json") if task.geometry_id == geometry_id)
    action = NormalizedScenarioAction(0, (0.0, 0.0, 0.0, 0.0, 0.0), AdversarialOption.GAP_CLOSE)
    executor = ScenarioExecutor(load_adapters(), mvr_parameter_spaces())
    episode = executor.reset(task, action)
    try:
        _, candidates = executor.enumerate_interactions(task)
        candidate = candidates[0]
        assert episode.applied_scenario.adversary_distance_to_conflict_m == pytest.approx(
            0.5 * (candidate.adversary_distance_min_m + candidate.adversary_distance_available_m)
        )
        assert episode.applied_scenario.sut_distance_to_conflict_m == pytest.approx(
            0.5 * (candidate.sut_distance_min_m + candidate.sut_distance_available_m)
        )
        assert episode.applied_scenario.normalized_continuous == action.continuous
        assert episode.map_tokens.map_hash == task.geometry_hash
    finally:
        episode.env.close()
