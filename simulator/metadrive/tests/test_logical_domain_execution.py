from __future__ import annotations

import pytest

from simulator.metadrive.scenario.catalog import mvr_parameter_spaces
from simulator.metadrive.scenario.executor import ScenarioExecutor
from simulator.metadrive.scenario.parameter_space import NormalizedScenarioAction
from simulator.metadrive.scenario.registry import load_adapters
from simulator.metadrive.scenario.taskbook import load_taskbook


def _executor() -> ScenarioExecutor:
    return ScenarioExecutor(load_adapters(), mvr_parameter_spaces())


def test_executor_rejects_action_outside_task_logical_domain_before_simulation() -> None:
    task = next(
        task
        for task in load_taskbook("simulator/metadrive/configs/taskbook.json")
        if task.logical_split == "validation"
    )
    with pytest.raises(ValueError, match="outside task Logical domain"):
        _executor().reset(task, NormalizedScenarioAction(0, (0.0,) * 5))


@pytest.mark.parametrize("family", ("merge", "roundabout"))
def test_executor_rejects_inactive_maneuver_onset(family: str) -> None:
    task = next(
        task
        for task in load_taskbook("simulator/metadrive/configs/taskbook.json")
        if task.functional_scenario == family and task.logical_split == "train"
    )
    with pytest.raises(ValueError, match="inactive Logical parameter"):
        _executor().reset(task, NormalizedScenarioAction(0, (0.0, 0.0, 0.0, 0.0, 0.1)))
