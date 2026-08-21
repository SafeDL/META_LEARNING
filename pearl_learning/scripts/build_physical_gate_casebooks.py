"""Calibrate from construction only, then freeze fresh train/Gate-eval pools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pearl_learning.src.benchmark_calibration import apply_calibration_manifest, calibrate_thresholds, run_baseline_rollout
from pearl_learning.src.casebook import CASE_SPLITS, PHYSICAL_GATE_CASEBOOK_SCHEMA, save_casebook, validate_casebook_disjoint
from pearl_learning.src.casebook_v2 import solve_interaction_boundary_case
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.physical_construction import construction_grid
from pearl_learning.src.task_env import LogicalMergeEnv
from pearl_learning.src.taskbook import load_taskbook, validate_physical_task_contract


def _measure(task: Any, cfg: Mapping[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    env = LogicalMergeEnv(task, cfg, [case])
    try:
        env.reset(options={"case": case})
        return dict(env.initial_case_measurements())
    finally:
        env.close()


def _condition_fingerprint(case: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(round(float(case[key]), 9) for key in (
        "target_initial_arrival_gap_s", "sut_initial_speed_mps", "adversary_initial_speed_mps",
        "sut_spawn_m", "adversary_spawn_m",
    ))


def _construction_cases(root: Path, original_task_id: str) -> list[dict[str, Any]]:
    path = root / "construction_casebooks" / "casebooks" / f"{original_task_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "physical_merge_construction_casebook_v1":
        raise ValueError("construction calibration must read construction-only casebooks")
    return [dict(row) for row in payload["cases"]["construction_pool"]]


def _construction_calibration(tasks: list[Any], selection: Mapping[str, Any], root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    source_ids = [str(value) for value in selection["selected_pair"]["task_ids"]]
    by_geometry = {task.geometry_id: task for task in tasks}
    rows: list[dict[str, Any]] = []
    for geometry_id, source_id in zip(selection["selected_pair"]["geometry_ids"], source_ids):
        cases = _construction_cases(root, source_id)
        if len(cases) != 63:
            raise ValueError("construction calibration requires complete 63-cell pools")
        task = by_geometry[str(geometry_id)]
        # Deterministic evenly spread ten-case construction-only calibration.
        indexes = np.linspace(0, len(cases) - 1, 10, dtype=int)
        for index in indexes:
            case = dict(cases[int(index)])
            for policy in ("zero", "random", "heuristic"):
                rows.append(run_baseline_rollout(task, case, cfg, policy))
    result = calibrate_thresholds(rows, min_arrival_gap_threshold_s=float(cfg["environment"]["decision_step_s"]))
    result["calibration_pool"] = "construction_pool"
    result["uses_gate_eval_pool"] = False
    result["selected_construction_task_ids"] = source_ids
    return result | {"calibration_hash": content_hash(result)}


def _gate_cases(
    task: Any,
    split: str,
    count: int,
    cfg: Mapping[str, Any],
    calibration_hash: str,
    forbidden_seeds: set[int],
    forbidden_conditions: set[tuple[float, ...]],
) -> list[dict[str, Any]]:
    recipe = {
        "schema": "physical_gate_eta_sampling_v1", "eta_levels_s": [row["target_initial_arrival_gap_s"] for row in construction_grid(cfg)[::9]],
        "sut_speed_range_mps": list(cfg["case_sampling"]["sut_initial_speed_mps"]),
        "adversary_speed_range_mps": list(cfg["case_sampling"]["adversary_initial_speed_mps"]),
        "split": split, "count": count,
    }
    recipe_hash = content_hash(recipe)
    rng = np.random.default_rng(int(content_hash({"task": task.task_id, "recipe": recipe_hash})[:16], 16))
    gaps = [float(value) for value in cfg["physical_construction"]["eta_gap_grid_s"]]
    cases: list[dict[str, Any]] = []
    used: set[int] = set(forbidden_seeds)
    attempts = 0
    while len(cases) < count and attempts < 1000:
        attempts += 1
        seed = int(rng.integers(1, 2**31 - 1))
        if seed in used:
            continue
        base = {
            "case_id": f"{task.task_id}_{split}_{len(cases):03d}", "case_seed": seed,
            "sut_initial_speed_mps": float(rng.uniform(*cfg["case_sampling"]["sut_initial_speed_mps"])),
            "adversary_initial_speed_mps": float(rng.uniform(*cfg["case_sampling"]["adversary_initial_speed_mps"])),
        }
        base["adversary_speed_mps"] = base["adversary_initial_speed_mps"]
        target_gap = float(rng.choice(gaps))
        try:
            solved, measured = solve_interaction_boundary_case(task, base, target_gap, lambda case: _measure(task, cfg, case))
        except (RuntimeError, ValueError):
            continue
        row = {
            **solved, "case_id": base["case_id"], "case_seed": seed,
            "target_initial_arrival_gap_s": target_gap,
            "actual_initial_arrival_gap_s": float(measured["adversary_time_s"]) - float(measured["sut_time_s"]),
            "initial_relative_speed_mps": float(measured["initial_relative_speed_mps"]),
            "adversary_initial_conflict_distance_m": float(measured["adversary_distance_m"]),
            "sut_initial_conflict_distance_m": float(measured["sut_distance_m"]),
            "initial_pair_distance_m": float(measured["initial_pair_distance_m"]),
            "initial_target_overlap": bool(measured.get("initial_target_overlap", False)),
            "difficulty_class": "interaction_boundary", "calibration_hash": calibration_hash,
            "pool_purpose": "physical_gate_train" if split == "train_pool" else "physical_gate_eval",
            "sampling_recipe_hash": recipe_hash,
            "eta_solver": {"name": "solve_interaction_boundary_case", "tolerance_s": 0.20, "uses_true_route_conflict_geometry": True},
        }
        fingerprint = _condition_fingerprint(row)
        if fingerprint in forbidden_conditions:
            continue
        used.add(seed); forbidden_conditions.add(fingerprint); cases.append(row)
    if len(cases) != count:
        raise RuntimeError(f"unable to produce {count} disjoint formal cases for {task.task_id}/{split}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--taskbook", required=True)
    parser.add_argument("--selection-manifest", required=True); parser.add_argument("--construction-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    selection = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))
    if selection.get("status") != "pass":
        raise ValueError("formal Gate pools require a passed construction selection")
    taskbook = load_taskbook(args.taskbook)
    geometries = [str(value) for value in selection["selected_pair"]["geometry_ids"]]
    tasks = [task for task in taskbook["meta_train"] if task.geometry_id in geometries]
    if len(tasks) != 2 or {task.geometry_id for task in tasks} != set(geometries):
        raise ValueError("promoted taskbook does not expose the selected pair in meta_train")
    for task in tasks:
        validate_physical_task_contract(task, cfg)
    tasks.sort(key=lambda task: geometries.index(task.geometry_id))
    construction_root = Path(args.construction_root)
    calibration = _construction_calibration(tasks, selection, construction_root, cfg)
    output = Path(args.output)
    write_json(output / "calibration" / "critical_thresholds.json", calibration)
    if calibration["status"] != "pass":
        raise SystemExit("construction-only calibration failed; formal Gate remains blocked")
    resolved = apply_calibration_manifest(cfg, calibration)
    books: dict[str, dict[str, list[dict[str, Any]]]] = {}
    hashes: dict[str, str] = {}
    source_ids = [str(value) for value in selection["selected_pair"]["task_ids"]]
    for task, source_id in zip(tasks, source_ids):
        construction = _construction_cases(construction_root, source_id)
        seeds = {int(row["case_seed"]) for row in construction}
        conditions = {_condition_fingerprint(row) for row in construction}
        train = _gate_cases(task, "train_pool", 8, resolved, str(calibration["calibration_hash"]), seeds, conditions)
        gate = _gate_cases(task, "gate_eval_pool", 4, resolved, str(calibration["calibration_hash"]), seeds, conditions)
        book = {split: [] for split in CASE_SPLITS} | {"train_pool": train, "gate_eval_pool": gate}
        books[task.task_id] = book
        hashes[task.task_id] = save_casebook(task, book, str(output), schema=PHYSICAL_GATE_CASEBOOK_SCHEMA, provenance={
            "selection_hash": selection["selection_hash"], "calibration_hash": calibration["calibration_hash"],
            "construction_pool_root": str(construction_root), "active_case_splits": ["train_pool", "gate_eval_pool"],
        })
    validate_casebook_disjoint(books)
    write_json(output / "casebooks" / "physical_gate_casebook_manifest.json", {
        "schema": PHYSICAL_GATE_CASEBOOK_SCHEMA, "selection_hash": selection["selection_hash"],
        "calibration_hash": calibration["calibration_hash"], "task_casebook_hashes": hashes,
        "uses_construction_pool_for_calibration_only": True, "uses_gate_eval_pool_for_calibration": False,
    })
    print("wrote disjoint physical Gate v2 casebooks")


if __name__ == "__main__":
    main()
