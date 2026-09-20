from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from diva_metadrive.scenario.catalog import mvr_parameter_spaces
from diva_metadrive.scenario.executor import ScenarioExecutor
from diva_metadrive.scenario.parameter_space import NormalizedScenarioAction
from diva_metadrive.scenario.registry import load_adapters
from diva_metadrive.scenario.taskbook import load_taskbook
from diva_metadrive.training.runner import HierarchicalRunner


@pytest.mark.parametrize("family", ("merge", "cutin"))
def test_non_collision_rollout_ends_only_after_sut_route_completion(family: str) -> None:
    task = next(
        row for row in load_taskbook("diva_metadrive/configs/taskbook.json")
        if row.functional_scenario == family and row.geometry_split == "validation"
    )
    task = replace(
        task,
        logical_domain_id="completion_probe",
        logical_domain_bounds={name: (-1.0, 1.0) for name in task.logical_domain_bounds},
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        # Keep the nominal adversary behind the SUT at reset; this is a
        # completion-contract test, not an adversarial-success test.
            NormalizedScenarioAction(
                0,
                (0.0, 0.0, 0.0, 0.0, 0.0)
                if family == "cutin"
                else (1.0, -1.0, -1.0, -1.0, 0.0),
            ),
        episode_seed=711,
        environment_overrides={"horizon": 900},
    )
    try:
        rollout = HierarchicalRunner(max_steps=900).rollout(
            episode,
            family,
            lambda _state: np.asarray((0.0, 0.0, 0.0, -1.0), dtype=np.float32),
        )
    finally:
        episode.env.close()
    assert not rollout.outcome["target_collision"]
    expected_condition = (
        "sut_cutin_follow_through"
        if family == "cutin" else "sut_route_destination"
    )
    assert rollout.outcome["test_completion_condition"] == expected_condition
    if family != "cutin":
        assert rollout.outcome["sut_arrived_destination"], rollout.outcome
    assert rollout.outcome["test_process_completed"]
    assert rollout.outcome["termination_reason"] == "sut_route_completed"


@pytest.mark.parametrize("family", ("merge", "cutin", "roundabout"))
def test_every_family_declares_sut_route_completion_as_the_test_endpoint(family: str) -> None:
    task = next(
        row for row in load_taskbook("diva_metadrive/configs/taskbook.json")
        if row.functional_scenario == family and row.geometry_split == "validation"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
            NormalizedScenarioAction(
                0, (0.0,) * mvr_parameter_spaces()[family].continuous_dim,
            ),
        episode_seed=712,
    )
    try:
        contract = episode.layout.traffic_contract
        expected_condition = (
            "sut_cutin_follow_through"
            if family == "cutin" else "sut_route_destination"
        )
        assert contract.completion_condition == expected_condition
        assert contract.terminate_on_target_collision
        assert contract.min_completion_steps >= 180
    finally:
        episode.env.close()
