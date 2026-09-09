from __future__ import annotations

import pytest

from mvr.diva.episode_executor import DivaEpisodeExecutor
from mvr.diva.types import DivaCutInDesign
from mvr.scenario.catalog import (
    minimum_cutin_path_length_m,
    mvr_parameter_spaces,
    valid_cutin_initial_state,
)
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.registry import load_adapters
from mvr.scenario.taskbook import load_taskbook
from mvr.training.runner import HierarchicalRunner


def test_diva_cutin_design_runs_through_existing_closed_loop_contract() -> None:
    task = next(
        task for task in load_taskbook("mvr/configs/taskbook.json")
        if task.task_id == "cutin-g01-cautious-cutin_interaction_core"
    )
    executor = DivaEpisodeExecutor(
        ScenarioExecutor(load_adapters(), mvr_parameter_spaces()),
        HierarchicalRunner(max_steps=2),
        environment_horizon=2,
    )
    observation = executor.run(
        task, DivaCutInDesign(0, (0.0,) * 5), episode_seed=11
    )
    assert observation.status == "censored"
    assert not observation.posterior_eligible
    assert observation.concrete_scenario["candidate_id"] == "left_target_lane"
    assert observation.outcome["cutin_actual_onset"] is None


def test_diva_fixed_initial_speed_completes_the_spatial_cutin() -> None:
    task = next(
        task for task in load_taskbook("mvr/configs/taskbook.json")
        if task.task_id == "cutin-g01-cautious-cutin_interaction_core"
    )
    executor = DivaEpisodeExecutor(
        ScenarioExecutor(load_adapters(), mvr_parameter_spaces()),
        HierarchicalRunner(max_steps=180),
        environment_horizon=180,
    )
    observation = executor.run(
        task,
        DivaCutInDesign(0, (0.0,) * 5),
        episode_seed=10011,
    )
    assert observation.outcome["cutin_actual_onset"] is not None
    assert observation.outcome["maneuver_completed"]
    assert observation.outcome["prescribed_adversary_speed_mps"] == pytest.approx(10.0)
    initial_state = observation.concrete_scenario["initial_state"]
    assert initial_state["initial_gap_m"] == pytest.approx(11.5)
    assert initial_state["declared_initial_center_gap_m"] == pytest.approx(16.0)
    assert initial_state["actual_initial_gap_m"] == pytest.approx(11.5, abs=0.35)
    assert initial_state["actual_initial_center_gap_m"] == pytest.approx(16.0, abs=0.35)
    assert initial_state["relative_speed_mps"] == pytest.approx(0.0)
    assert initial_state["cutin_path_length_m"] == pytest.approx(80.0)


def test_cutin_reset_rejects_incoherent_speed_or_initial_gap() -> None:
    assert valid_cutin_initial_state(10.0, -2.0, 7.0, 80.0)
    assert valid_cutin_initial_state(10.0, 4.0, 7.0, 120.0)
    assert not valid_cutin_initial_state(10.0, 0.0, 1.9, 80.0)
    assert not valid_cutin_initial_state(10.0, -10.1, 7.0, 120.0)
    assert valid_cutin_initial_state(13.0, 0.0, 7.0, 30.0)
    assert not valid_cutin_initial_state(13.0, 0.0, 7.0, 25.0)
    assert minimum_cutin_path_length_m(13.0) == pytest.approx(25.29, abs=0.01)
