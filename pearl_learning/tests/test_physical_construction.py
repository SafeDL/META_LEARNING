from __future__ import annotations

from types import SimpleNamespace

import pytest

from pearl_learning.scripts.audit_physical_task_policy_heterogeneity import physical_heterogeneity_gate_report
from pearl_learning.scripts.audit_physical_task_policy_heterogeneity_v2 import physical_heterogeneity_gate_report_v2
from pearl_learning.src.casebook_v2 import solve_interaction_boundary_case
from pearl_learning.src.physical_construction import construction_grid, select_construction_pair, task_action_conflict_report


def _probe_matrix(first: float, second: float) -> dict[str, dict[str, float]]:
    return {
        "longitudinal_-0.75": {"valid_critical_strict_rate": first},
        "longitudinal_-0.35": {"valid_critical_strict_rate": second},
        "longitudinal_+0.00": {"valid_critical_strict_rate": 0.0},
        "longitudinal_+0.35": {"valid_critical_strict_rate": 0.0},
        "longitudinal_+0.75": {"valid_critical_strict_rate": 0.0},
    }


def test_interaction_solver_solves_both_spawns_and_rejects_infeasible_gap() -> None:
    task = SimpleNamespace(spawn_regions={"adversary": [0.0, 100.0], "sut": [0.0, 100.0]})

    def measure(case):
        adv = (100.0 - case["adversary_spawn_m"]) / case["adversary_initial_speed_mps"]
        sut = (100.0 - case["sut_spawn_m"]) / case["sut_initial_speed_mps"]
        return {
            "adversary_time_s": adv, "sut_time_s": sut,
            "adversary_distance_m": 100.0 - case["adversary_spawn_m"],
            "sut_distance_m": 100.0 - case["sut_spawn_m"],
            "adversary_signed_distance_m": 1.0, "sut_signed_distance_m": 1.0,
            "initial_target_overlap": False,
        }

    base = {"adversary_initial_speed_mps": 10.0, "sut_initial_speed_mps": 10.0}
    solved, measured = solve_interaction_boundary_case(task, base, 1.0, measure)
    assert measured["adversary_time_s"] - measured["sut_time_s"] == pytest.approx(1.0)
    assert solved["adversary_spawn_m"] != 50.0 and solved["sut_spawn_m"] != 50.0
    with pytest.raises(ValueError, match="infeasible"):
        solve_interaction_boundary_case(task, base, 20.0, measure)


def test_construction_grid_is_the_frozen_63_cell_lattice() -> None:
    grid = construction_grid({"physical_construction": {
        "eta_gap_grid_s": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        "sut_speed_grid_mps": [11.0, 12.0, 13.0],
        "adversary_speed_grid_mps": [11.0, 12.5, 14.0],
    }})
    assert len(grid) == 63
    assert grid[0]["target_initial_arrival_gap_s"] == -1.5
    assert grid[-1]["adversary_initial_speed_mps"] == 14.0


def test_task_action_conflict_requires_reachability_unique_different_and_bilateral_advantage() -> None:
    passing = task_action_conflict_report({"a": _probe_matrix(0.75, 0.25), "b": _probe_matrix(0.25, 0.75)}, ["a", "b"])
    assert passing["status"] == "pass"
    assert task_action_conflict_report({"a": _probe_matrix(0.40, 0.25), "b": _probe_matrix(0.25, 0.75)}, ["a", "b"])["status"] == "fail"
    assert task_action_conflict_report({"a": _probe_matrix(0.75, 0.75), "b": _probe_matrix(0.25, 0.75)}, ["a", "b"])["status"] == "fail"
    assert task_action_conflict_report({"a": _probe_matrix(0.75, 0.50), "b": _probe_matrix(0.25, 0.75)}, ["a", "b"])["status"] == "pass"
    assert task_action_conflict_report({"a": _probe_matrix(0.75, 0.55), "b": _probe_matrix(0.25, 0.75)}, ["a", "b"])["status"] == "fail"


def test_selection_uses_predeclared_topology_priority_after_score_ties() -> None:
    candidates = [
        {"task_id": "y", "geometry_id": "y_merge_24", "logical_type": "y_merge"},
        {"task_id": "b", "geometry_id": "bottleneck_32", "logical_type": "bottleneck_merge"},
        {"task_id": "r", "geometry_id": "on_ramp_srs", "logical_type": "on_ramp_merge"},
    ]
    reports = {
        ("b", "y"): task_action_conflict_report({"b": _probe_matrix(0.25, 0.75), "y": _probe_matrix(0.75, 0.25)}, ["b", "y"]),
        ("b", "r"): task_action_conflict_report({"b": _probe_matrix(0.25, 0.75), "r": _probe_matrix(0.75, 0.25)}, ["b", "r"]),
    }
    selection = select_construction_pair(candidates, reports)
    assert selection["status"] == "pass"
    assert selection["selected_pair"]["geometry_ids"] == ["y_merge_24", "bottleneck_32"]


def test_gate_v2_adds_own_task_hard_conditions_without_changing_v1() -> None:
    matrix = {
        "a": {"a": {"valid_critical_strict_rate": 0.0}, "b": {"valid_critical_strict_rate": 0.0}},
        "b": {"a": {"valid_critical_strict_rate": 0.0}, "b": {"valid_critical_strict_rate": 0.0}},
    }
    assert physical_heterogeneity_gate_report(matrix, ["a", "b"], minimum_advantage=0.25)["status"] == "fail"
    assert physical_heterogeneity_gate_report_v2(matrix, ["a", "b"])["status"] == "fail"
    passed = {
        "a": {"a": {"valid_critical_strict_rate": 0.50}, "b": {"valid_critical_strict_rate": 0.25}},
        "b": {"a": {"valid_critical_strict_rate": 0.25}, "b": {"valid_critical_strict_rate": 0.50}},
    }
    assert physical_heterogeneity_gate_report_v2(passed, ["a", "b"])["status"] == "pass"
    own_fail = {
        "a": {"a": {"valid_critical_strict_rate": 0.0}, "b": {"valid_critical_strict_rate": 0.0}},
        "b": {"a": {"valid_critical_strict_rate": -0.25}, "b": {"valid_critical_strict_rate": 0.0}},
    }
    assert physical_heterogeneity_gate_report_v2(own_fail, ["a", "b"])["status"] == "fail"
