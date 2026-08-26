from __future__ import annotations

from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.concrete import ConcreteScenario
from mvr.scenario.taskbook import load_taskbook


def test_concrete_scenario_reconstructs_the_outer_action() -> None:
    task = load_taskbook("mvr/configs/taskbook.json")[0]
    scenario = ConcreteScenario(
        task.geometry_id, task.geometry_hash, task.geometry_seed, "main_conflict", "merge:zone",
        "gap_close", {
            "adversary_distance_to_conflict_m": 2.0,
            "sut_distance_to_conflict_m": 3.0,
            "adversary_initial_speed_mps": 8.0,
            "sut_initial_speed_mps": 9.0,
        }, "policy-hash",
    )
    action = scenario.replay_action(mvr_parameter_spaces()["merge"])
    assert action.candidate_index == 0 and scenario.to_dict()["geometry_hash"] == task.geometry_hash


def test_concrete_scenario_replay_clamps_geometry_applied_distance() -> None:
    task = load_taskbook("mvr/configs/taskbook.json")[0]
    scenario = ConcreteScenario(
        task.geometry_id,
        task.geometry_hash,
        task.geometry_seed,
        "main_conflict",
        "merge:zone",
        "gap_close",
        {
            "adversary_distance_to_conflict_m": 99.0,
            "sut_distance_to_conflict_m": 2.0,
            "adversary_initial_speed_mps": 11.0,
            "sut_initial_speed_mps": 9.0,
        },
        "policy-hash",
    )
    space = mvr_parameter_spaces()["merge"]
    action = scenario.replay_action(space)
    decoded = space.decode(action)
    assert decoded["adversary_distance_to_conflict_m"] == space.bounds[
        "adversary_distance_to_conflict_m"
    ][1]


def test_concrete_manifest_keeps_inner_condition_and_episode_seed() -> None:
    scenario = ConcreteScenario(
        "merge-g01", "a" * 64, 101, "main_conflict", "merge:zone", "gap_close",
        {"adversary_distance_to_conflict_m": 2.0}, "policy-hash", (0.1, -0.2), 107,
    )
    payload = scenario.to_dict()
    assert payload["latent"] == (0.1, -0.2)
    assert payload["episode_seed"] == 107
