"""Acceptance tests for clocks, fresh objects, admission and the new collector route."""

from dataclasses import asdict

import pytest

from controllers.highway.catalog import (
    FORMAL_HISTORY_SUTS,
    FORMAL_TARGET_SUTS,
    INDEPENDENT_SUTS,
)
from controllers.highway.control import Control
from controllers.highway.factory import create
from experiments.executions import _execute_task
from experiments.formal_scenarios import candidate_payload
from experiments.history_transfer_experiment import BUDGET_METHODS, register
from experiments.qualification import COURSES, DEVELOPMENT_COURSES
from experiments.scenarios import FORMAL_SCENARIOS, features
from experiments.schema import Scenario
from experiments.storage import read_json
from simulator.highway.collision_env import CollisionWorld
from simulator.highway.timing import DECISION_DT, PHYSICS_DT
from simulator.highway.training_env import DrivingEnv


def test_native_timing_cannot_be_overridden_in_constructor_or_reconfiguration():
    for config in ({"simulation_frequency": 15}, {"policy_frequency": 1}):
        with pytest.raises(ValueError, match="20 Hz"):
            CollisionWorld(config=config)
        world = CollisionWorld()
        try:
            with pytest.raises(ValueError, match="20 Hz"):
                world.configure(config)
            world.advance(Control(0, 0))
            assert world.tick == 4 and world.decisions == 1
            assert world.trace[-1]["time_s"] == DECISION_DT
            assert world.trace[1]["time_s"] == PHYSICS_DT
        finally:
            world.close()


def test_policy_training_uses_the_same_four_native_ticks_per_action():
    env = DrivingEnv()
    try:
        env.reset(seed=101)
        env.step(1)
        assert env.world.tick == 4 and env.world.decisions == 1
    finally:
        env.close()


def test_formal_registration_has_new_configurations_and_consistent_budgets(tmp_path):
    protocol = register(tmp_path)
    assert protocol["base_physical_executions"] == 80000
    assert protocol["scenarios"] == list(FORMAL_SCENARIOS)
    assert sum(protocol["history_counts_per_ads"].values()) * 8 == 16000
    assert protocol["active_experiments"] == ["prediction"]
    assert "budget" not in protocol
    assert set(protocol["history_counts_per_ads"].values()) == {500}
    assert "pooled_history" not in BUDGET_METHODS
    assert protocol["k_shot"]["methods"] == ["target_only", "pooled_history", "full"]
    assert protocol["feedback_seeds"] == [11]
    assert protocol["k_shot"]["query_count"] == 200
    assert protocol["k_shot"]["evaluation_count"] == 1800
    assert protocol["k_shot"]["checkpoints"] == [0, 5, 10, 20, 50, 100, 200]
    assert len(FORMAL_TARGET_SUTS) == len(FORMAL_HISTORY_SUTS) == 8
    assert {s.sut_id for s in FORMAL_TARGET_SUTS}.isdisjoint(s.sut_id for s in INDEPENDENT_SUTS)
    for target, old in zip(FORMAL_TARGET_SUTS, INDEPENDENT_SUTS):
        assert tuple(target.supported_scenarios) == FORMAL_SCENARIOS
        if target.family == "learning":
            assert target.training_seed != old.training_seed
            assert target.training_run != old.training_run
        else:
            assert target.parameters != old.parameters
            assert create(target.sut_id).policy.parameters == target.parameters
    assert register(tmp_path) == protocol


def test_restricted_cut_in_is_a_valid_candidate_pool_with_both_sides():
    scenes = [Scenario(**row["scenario"]) for row in candidate_payload("cut_in_onset", 8, "history")["records"]]
    assert {s.categories["side"] for s in scenes} == {"left", "right"}
    assert features(scenes).shape[0] == 8


def test_cut_in_collection_calls_the_unified_executor(tmp_path):
    scene = Scenario(**candidate_payload("cut_in_onset", 1, "history")["records"][0]["scenario"])
    paid = _execute_task("target_idm_mobil", scene, 81001, tmp_path / "attempt.json", 1)
    assert paid.feedback.scenario_id == scene.scenario_id
    assert paid.trajectory[1]["time_s"] == PHYSICS_DT
    assert {a["actor_id"] for a in paid.trajectory[0]["actors"]} == {
        "ego",
        "cut_in",
        "guard_front",
        "guard_rear",
    }
    assert read_json(tmp_path / "attempt.json")["status"] == "complete"


def test_checkpoint_selection_coordinates_differ_from_acceptance_courses():
    for dev, accepted in zip(DEVELOPMENT_COURSES, COURSES):
        assert dev.scenario.coordinate() != accepted.scenario.coordinate()
        assert asdict(dev.scenario) != asdict(accepted.scenario)
