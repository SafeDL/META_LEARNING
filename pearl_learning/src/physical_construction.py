"""Deterministic zero-training screening for physical task construction."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

import numpy as np

from .io import content_hash


CONSTRUCTION_SCHEMA = "physical_task_construction_screen_v1"
PROBE_LONGITUDINAL = (-0.75, -0.35, 0.0, 0.35, 0.75)


def probe_name(value: float) -> str:
    return f"longitudinal_{value:+.2f}"


def construction_grid(config: Mapping[str, Any]) -> list[dict[str, float]]:
    source = dict(config["physical_construction"])
    gaps = [float(value) for value in source["eta_gap_grid_s"]]
    sut_speeds = [float(value) for value in source["sut_speed_grid_mps"]]
    adversary_speeds = [float(value) for value in source["adversary_speed_grid_mps"]]
    if len(gaps) != 7 or len(sut_speeds) != 3 or len(adversary_speeds) != 3:
        raise ValueError("construction grid must be 7 ETA values x 3 SUT speeds x 3 adversary speeds")
    if len(set(gaps)) != len(gaps) or any(speed <= 0.0 for speed in sut_speeds + adversary_speeds):
        raise ValueError("construction grid values must be unique and speeds positive")
    return [
        {"target_initial_arrival_gap_s": gap, "sut_initial_speed_mps": sut, "adversary_initial_speed_mps": adv}
        for gap in gaps for sut in sut_speeds for adv in adversary_speeds
    ]


def construction_case_seed(task_id: str, index: int) -> int:
    return int(content_hash({"schema": CONSTRUCTION_SCHEMA, "task_id": task_id, "index": index})[:16], 16) % (2**31 - 2) + 1


def construction_case_payload(
    task: Any,
    index: int,
    grid: Mapping[str, float],
    solved: Mapping[str, Any],
    measured: Mapping[str, Any],
) -> dict[str, Any]:
    target_gap = float(grid["target_initial_arrival_gap_s"])
    actual_gap = float(measured["adversary_time_s"]) - float(measured["sut_time_s"])
    return {
        **dict(solved),
        "case_id": f"{task.task_id}_construction_pool_{index:03d}",
        "case_seed": construction_case_seed(task.task_id, index),
        "target_initial_arrival_gap_s": target_gap,
        "actual_initial_arrival_gap_s": actual_gap,
        "initial_relative_speed_mps": float(measured["initial_relative_speed_mps"]),
        "adversary_initial_conflict_distance_m": float(measured["adversary_distance_m"]),
        "sut_initial_conflict_distance_m": float(measured["sut_distance_m"]),
        "initial_pair_distance_m": float(measured["initial_pair_distance_m"]),
        "initial_target_overlap": bool(measured.get("initial_target_overlap", False)),
        "difficulty_class": "interaction_boundary",
        "calibration_hash": "construction_metric_predeclared",
        "pool_purpose": "construction_screening",
        "construction_grid": dict(grid),
        "eta_solver": {
            "name": "solve_interaction_boundary_case",
            "tolerance_s": 0.20,
            "uses_true_route_conflict_geometry": True,
            "map_hash": task.map_hash,
            "adversary_route_hash": task.adversary_route_hash,
            "sut_route_hash": task.sut_route_hash,
        },
    }


def _unique_best(matrix: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, float]:
    values = {name: float(row["valid_critical_strict_rate"]) for name, row in matrix.items()}
    maximum = max(values.values())
    winners = [name for name, value in values.items() if value == maximum]
    return (winners[0] if len(winners) == 1 else None, maximum)


def task_action_conflict_report(
    probe_matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    task_ids: list[str],
    *,
    minimum_reachability: float = 0.50,
    minimum_advantage: float = 0.25,
) -> dict[str, Any]:
    """Apply the predeclared non-training Task-Action Conflict Gate."""
    if len(task_ids) != 2 or len(set(task_ids)) != 2 or set(probe_matrix) != set(task_ids):
        raise ValueError("Task-Action Conflict Gate requires exactly two complete tasks")
    first, second = task_ids
    if not probe_matrix[first] or set(probe_matrix[first]) != set(probe_matrix[second]):
        raise ValueError("Task-Action Conflict probe matrix is incomplete")
    best_first, reach_first = _unique_best(probe_matrix[first])
    best_second, reach_second = _unique_best(probe_matrix[second])
    cross_advantages: dict[str, float | None] = {first: None, second: None}
    if best_first is not None and best_second is not None:
        cross_advantages = {
            first: float(probe_matrix[first][best_first]["valid_critical_strict_rate"])
            - float(probe_matrix[first][best_second]["valid_critical_strict_rate"]),
            second: float(probe_matrix[second][best_second]["valid_critical_strict_rate"])
            - float(probe_matrix[second][best_first]["valid_critical_strict_rate"]),
        }
    criteria = {
        "first_task_reachable": reach_first >= float(minimum_reachability),
        "second_task_reachable": reach_second >= float(minimum_reachability),
        "unique_best_probe_per_task": best_first is not None and best_second is not None,
        "best_probes_differ": best_first is not None and best_second is not None and best_first != best_second,
        "first_task_cross_advantage": cross_advantages[first] is not None and cross_advantages[first] >= float(minimum_advantage),
        "second_task_cross_advantage": cross_advantages[second] is not None and cross_advantages[second] >= float(minimum_advantage),
    }
    passed = all(criteria.values())
    return {
        "schema": "physical_task_action_conflict_gate_v1",
        "status": "pass" if passed else "fail",
        "task_ids": task_ids,
        "aggregate_best_probe": {first: best_first, second: best_second},
        "maximum_vcsr": {first: reach_first, second: reach_second},
        "cross_vcsr_advantage": cross_advantages,
        "minimum_reachability": float(minimum_reachability),
        "minimum_cross_advantage": float(minimum_advantage),
        "criteria": criteria,
        "failure_action": None if passed else "revise_physical_task_or_case_distribution_before_sac",
    }


def _pair_priority(first: Mapping[str, Any], second: Mapping[str, Any]) -> tuple[int, str, str]:
    kinds = frozenset((str(first["logical_type"]), str(second["logical_type"])))
    if kinds == {"y_merge", "bottleneck_merge"}:
        rank = 0
    elif kinds == {"on_ramp_merge", "bottleneck_merge"}:
        rank = 1
    else:
        rank = 2
    return rank, *sorted((str(first["geometry_id"]), str(second["geometry_id"])))


def select_construction_pair(
    candidates: list[Mapping[str, Any]],
    pair_reports: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the single predeclared best passing pair without Gate-case use."""
    by_id = {str(row["task_id"]): row for row in candidates}
    passing: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    for first_id, second_id in combinations(sorted(by_id), 2):
        first, second = by_id[first_id], by_id[second_id]
        if first["logical_type"] == second["logical_type"]:
            continue
        report = pair_reports.get((first_id, second_id)) or pair_reports.get((second_id, first_id))
        if report is not None and report["status"] == "pass":
            passing.append((first, second, report))
    if not passing:
        return {
            "schema": "physical_task_construction_selection_v1", "status": "fail",
            "selection_rule": "no candidate pair passed the fixed Task-Action Conflict Gate",
            "selected_pair": None,
        }

    def score(row: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]) -> tuple[float, float, tuple[int, str, str]]:
        first, second, report = row
        advantages = report["cross_vcsr_advantage"]
        return (
            min(float(advantages[first["task_id"]]), float(advantages[second["task_id"]])),
            float(report["maximum_vcsr"][first["task_id"]]) + float(report["maximum_vcsr"][second["task_id"]]),
            _pair_priority(first, second),
        )
    # Higher conflict/reachability wins; the final priority is ascending.
    passing.sort(key=lambda row: (-score(row)[0], -score(row)[1], score(row)[2]))
    first, second, report = passing[0]
    preferred = (
        ("y_merge", "bottleneck_merge"),
        ("on_ramp_merge", "bottleneck_merge"),
    )
    for left_type, right_type in preferred:
        if str(first["logical_type"]) == right_type and str(second["logical_type"]) == left_type:
            first, second = second, first
            break
    selected = {
        "task_ids": [str(first["task_id"]), str(second["task_id"])],
        "geometry_ids": [str(first["geometry_id"]), str(second["geometry_id"])],
        "logical_types": [str(first["logical_type"]), str(second["logical_type"])],
        "score": {"minimum_cross_advantage": score(passing[0])[0], "sum_own_vcsr": score(passing[0])[1]},
        "gate": report,
    }
    payload = {
        "schema": "physical_task_construction_selection_v1", "status": "pass",
        "selection_rule": "max min(bidirectional cross VCSR advantage), then max sum own VCSR, then y_merge-bottleneck, on_ramp-bottleneck, geometry-id",
        "selected_pair": selected,
        "passing_pair_count": len(passing),
    }
    return {**payload, "selection_hash": content_hash(payload)}
