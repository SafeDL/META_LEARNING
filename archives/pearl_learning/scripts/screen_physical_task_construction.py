"""Run the fixed-policy, zero-training physical task construction screen."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from archives.pearl_learning.src.casebook import CONSTRUCTION_CASEBOOK_SCHEMA, CASE_SPLITS, save_casebook
from archives.pearl_learning.src.casebook_v2 import solve_interaction_boundary_case
from archives.pearl_learning.src.io import content_hash, file_sha256, prepare_run_manifest, read_config, write_json
from archives.pearl_learning.src.metrics import summarize
from archives.pearl_learning.src.physical_construction import (
    CONSTRUCTION_SCHEMA, PROBE_LONGITUDINAL, construction_case_payload, construction_grid,
    probe_name, select_construction_pair, task_action_conflict_report,
)
from archives.pearl_learning.src.task_env import LogicalMergeEnv
from archives.pearl_learning.src.taskbook import load_taskbook, taskbook_payload, validate_physical_task_contract


def _measure(task: Any, cfg: Mapping[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    env = LogicalMergeEnv(task, cfg, [case])
    try:
        env.reset(options={"case": case})
        return dict(env.initial_case_measurements())
    finally:
        env.close()


def _construction_cases(task: Any, cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, grid in enumerate(construction_grid(cfg)):
        base = {
            "case_id": f"{task.task_id}_construction_seed_{index:03d}",
            "case_seed": int(content_hash({"task": task.task_id, "grid": grid})[:16], 16) % (2**31 - 2) + 1,
            "sut_initial_speed_mps": float(grid["sut_initial_speed_mps"]),
            "adversary_initial_speed_mps": float(grid["adversary_initial_speed_mps"]),
            "adversary_speed_mps": float(grid["adversary_initial_speed_mps"]),
        }
        try:
            solved, measured = solve_interaction_boundary_case(
                task, base, float(grid["target_initial_arrival_gap_s"]),
                lambda case: _measure(task, cfg, case),
            )
            cases.append(construction_case_payload(task, index, grid, solved, measured))
        except (RuntimeError, ValueError) as exc:
            failures.append({"grid_index": index, "grid": grid, "error": str(exc)})
    return cases, failures


def _run_probe(task: Any, cfg: Mapping[str, Any], cases: list[dict[str, Any]], longitudinal: float) -> dict[str, Any]:
    env = LogicalMergeEnv(task, cfg, cases)
    records: list[dict[str, Any]] = []
    try:
        for case in cases:
            observation, _ = env.reset(options={"case": case})
            terminated = truncated = False
            while not (terminated or truncated):
                observation, _, terminated, truncated, _ = env.step(
                    np.asarray([0.0, longitudinal], dtype=np.float32)
                )
            records.append(env.episode_record())
    finally:
        env.close()
    return {
        "action": {"steering_residual": 0.0, "longitudinal": float(longitudinal), "constant_full_episode": True},
        **summarize(records, case_metadata={str(case["case_id"]): case for case in cases}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    source = dict(cfg["physical_construction"])
    candidate_ids = [str(value) for value in source["candidate_geometry_ids"]]
    if len(candidate_ids) != 7 or len(set(candidate_ids)) != 7:
        raise ValueError("physical construction requires exactly the seven predeclared candidate geometries")
    taskbook = load_taskbook(args.taskbook)
    candidates = [task for tasks in taskbook.values() for task in tasks if task.geometry_id in candidate_ids]
    if {task.geometry_id for task in candidates} != set(candidate_ids):
        raise ValueError("construction candidates are missing from the frozen taskbook")
    for task in candidates:
        validate_physical_task_contract(task, cfg)
    candidates.sort(key=lambda task: candidate_ids.index(task.geometry_id))
    root = Path(args.output)
    run_manifest = {
        "schema": "physical_task_construction_run_v1", "run_name": "physical_task_construction_screen",
        "source_config_sha256": file_sha256(args.config), "resolved_config_sha256": content_hash(cfg),
        "taskbook_hash": content_hash(taskbook_payload(taskbook)), "casebook_hashes": {},
        "critical_threshold_hash": "construction_metric_predeclared",
        "construction_grid_size": len(construction_grid(cfg)),
        "probe_longitudinal_values": list(PROBE_LONGITUDINAL),
    }
    prepare_run_manifest(root, run_manifest, resume=args.resume)
    candidates_payload: list[dict[str, Any]] = []
    matrices: dict[str, dict[str, dict[str, Any]]] = {}
    case_hashes: dict[str, str] = {}
    usable: dict[str, Any] = {}
    for task in candidates:
        cases, failures = _construction_cases(task, cfg)
        row = {
            "task_id": task.task_id, "geometry_id": task.geometry_id, "logical_type": task.logical_type,
            "map_hash": task.map_hash, "adversary_route_hash": task.adversary_route_hash,
            "sut_route_hash": task.sut_route_hash, "expected_grid_cells": len(construction_grid(cfg)),
            "realized_grid_cells": len(cases), "grid_failures": failures,
            "status": "pass" if not failures else "fail_infeasible_grid",
        }
        if not failures:
            book = {split: ([] if split != "construction_pool" else cases) for split in CASE_SPLITS}
            case_hashes[task.task_id] = save_casebook(
                task, book, str(root / "construction_casebooks"), schema=CONSTRUCTION_CASEBOOK_SCHEMA,
                provenance={"pool_purpose": "construction_screening", "grid_size": len(cases), "taskbook_hash": run_manifest["taskbook_hash"]},
            )
            matrices[task.task_id] = {
                probe_name(value): _run_probe(task, cfg, cases, value) for value in PROBE_LONGITUDINAL
            }
            usable[task.task_id] = task
            row["probe_matrix"] = matrices[task.task_id]
        candidates_payload.append(row)
        print(f"construction {task.geometry_id}: {row['status']}", flush=True)
    pair_reports: dict[tuple[str, str], dict[str, Any]] = {}
    task_list = sorted(usable.values(), key=lambda task: task.task_id)
    for index, first in enumerate(task_list):
        for second in task_list[index + 1:]:
            if first.logical_type == second.logical_type:
                continue
            pair_reports[(first.task_id, second.task_id)] = task_action_conflict_report(
                {first.task_id: matrices[first.task_id], second.task_id: matrices[second.task_id]},
                [first.task_id, second.task_id],
            )
    selection = select_construction_pair(candidates_payload, pair_reports)
    report = {
        "schema": CONSTRUCTION_SCHEMA, "status": selection["status"], "run_manifest": run_manifest,
        "candidates": candidates_payload,
        "pair_reports": [report for _, report in sorted(pair_reports.items())],
        "selection": selection,
        "construction_casebook_hashes": case_hashes,
        "next_allowed_stage": "promote_construction_pair" if selection["status"] == "pass" else None,
        "failure_action": None if selection["status"] == "pass" else "revise_physical_task_or_case_distribution_before_sac",
    }
    write_json(root / "construction_screen_report.json", report)
    write_json(root / "selection_manifest.json", selection)
    print(f"Construction screen: {selection['status']}")
    if selection["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
