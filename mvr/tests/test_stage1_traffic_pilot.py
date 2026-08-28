from __future__ import annotations

import numpy as np
import pytest

from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.option import AdversarialOption
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.scenario.registry import load_adapters
from mvr.scenario.taskbook import load_taskbook
from mvr.training.runner import HierarchicalRunner


@pytest.mark.parametrize(
    ("task_id", "continuous"),
    (
            ("merge-g04-fast_small_gap", (0.0, 0.0, 0.0, 0.0, 0.0)),
            ("roundabout-g04-fast_small_gap", (0.0, 0.0, -1.0, -1.0, 0.0)),
    ),
)
def test_non_cutin_stage1_pilots_remain_lawful(
    task_id: str,
    continuous: tuple[float, float, float, float, float],
) -> None:
    task = next(task for task in load_taskbook("mvr/configs/taskbook.json") if task.task_id == task_id)
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(0, continuous, AdversarialOption.APPROACH_CONFLICT),
        episode_seed=204,
    )
    try:
        rollout = HierarchicalRunner(max_steps=60).rollout(
            episode,
            task.functional_scenario,
            AdversarialOption.APPROACH_CONFLICT.value,
            lambda _: np.zeros(2, dtype=np.float32),
        )
    finally:
        episode.env.close()
    assert rollout.outcome["is_valid_episode"]
    assert not rollout.outcome["adversary_traffic_violation"]
