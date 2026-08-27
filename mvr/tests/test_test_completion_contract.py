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


@pytest.mark.parametrize("family", ("merge", "cutin"))
def test_non_collision_rollout_ends_only_after_sut_route_completion(family: str) -> None:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.functional_scenario == family and row.geometry_split == "validation"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        # Keep the nominal adversary behind the SUT at reset; this is a
        # completion-contract test, not an adversarial-success test.
        NormalizedScenarioAction(0, (1.0, -1.0, -1.0, -1.0), AdversarialOption.GAP_CLOSE),
        episode_seed=711,
        environment_overrides={"horizon": 480},
    )
    try:
        rollout = HierarchicalRunner(max_steps=480).rollout(
            episode,
            family,
            AdversarialOption.GAP_CLOSE.value,
            lambda _state: np.asarray((-1.0, 0.0, 0.0), dtype=np.float32),
        )
    finally:
        episode.env.close()
    assert not rollout.outcome["target_collision"]
    assert rollout.outcome["test_completion_condition"] == "sut_route_destination"
    assert rollout.outcome["sut_arrived_destination"], rollout.outcome
    assert rollout.outcome["test_process_completed"]
    assert rollout.outcome["termination_reason"] == "sut_route_completed"


@pytest.mark.parametrize("family", ("merge", "cutin", "roundabout"))
def test_every_family_declares_sut_route_completion_as_the_test_endpoint(family: str) -> None:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.functional_scenario == family and row.geometry_split == "validation"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        NormalizedScenarioAction(0, (0.0,) * 4, AdversarialOption.GAP_CLOSE),
        episode_seed=712,
    )
    try:
        contract = episode.layout.traffic_contract
        assert contract.completion_condition == "sut_route_destination"
        assert contract.terminate_on_target_collision
        assert contract.min_completion_steps >= 180
    finally:
        episode.env.close()
