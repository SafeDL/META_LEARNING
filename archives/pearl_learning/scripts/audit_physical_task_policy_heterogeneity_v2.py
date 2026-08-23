"""Formal physical Gate v2: learnability plus bidirectional transfer advantage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from stable_baselines3 import SAC

from archives.pearl_learning.scripts.audit_physical_task_policy_heterogeneity import GATE_SAC_HYPERPARAMETERS, _action_distance, _evaluate_sac, _implementation_hash, _new_sac
from archives.pearl_learning.src.benchmark_calibration import apply_calibration_manifest
from archives.pearl_learning.src.casebook import PHYSICAL_GATE_CASEBOOK_SCHEMA, load_casebook
from archives.pearl_learning.src.io import content_hash, file_sha256, prepare_run_manifest, read_config, write_json
from archives.pearl_learning.src.task_env import LogicalMergeEnv
from archives.pearl_learning.src.taskbook import load_taskbook, taskbook_payload, validate_physical_task_contract


def physical_heterogeneity_gate_report_v2(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]], task_ids: list[str], *, minimum_own_vcsr: float = 0.25,
    minimum_advantage: float = 0.25,
) -> dict[str, Any]:
    if len(task_ids) != 2 or len(set(task_ids)) != 2 or set(matrix) != set(task_ids):
        raise ValueError("physical Gate v2 requires a complete 2x2 matrix")
    if any(set(matrix[source]) != set(task_ids) for source in task_ids):
        raise ValueError("physical Gate v2 matrix is incomplete")
    first, second = task_ids
    own = {first: float(matrix[first][first]["valid_critical_strict_rate"]), second: float(matrix[second][second]["valid_critical_strict_rate"])}
    advantages = {
        first: own[first] - float(matrix[second][first]["valid_critical_strict_rate"]),
        second: own[second] - float(matrix[first][second]["valid_critical_strict_rate"]),
    }
    criteria = {
        f"{first}_own_task_vcsr": own[first] >= float(minimum_own_vcsr),
        f"{second}_own_task_vcsr": own[second] >= float(minimum_own_vcsr),
        f"{first}_diagonal_advantage": advantages[first] >= float(minimum_advantage),
        f"{second}_diagonal_advantage": advantages[second] >= float(minimum_advantage),
    }
    passed = all(criteria.values())
    return {
        "schema": "physical_task_policy_heterogeneity_gate_v2", "gate_name": "physical_task_policy_heterogeneity",
        "status": "pass" if passed else "fail", "task_ids": task_ids,
        "hard_gate_metric": "valid_critical_strict_rate", "own_task_vcsr": own,
        "diagonal_vcsr_advantage": advantages, "minimum_own_task_vcsr": float(minimum_own_vcsr),
        "minimum_diagonal_vcsr_advantage": float(minimum_advantage), "criteria": criteria,
        "action_distance_is_diagnostic_only": True,
        "next_allowed_stage": "vanilla_pearl_physical_pilot" if passed else None,
        "failure_action": None if passed else "revise_physical_task_or_case_distribution; do_not_tune_pearl",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True); parser.add_argument("--critical-thresholds", required=True)
    parser.add_argument("--selection-manifest", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = read_config(args.config)
    calibration = json.loads(Path(args.critical_thresholds).read_text(encoding="utf-8"))
    cfg = apply_calibration_manifest(source, calibration)
    selection = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))
    if selection.get("status") != "pass":
        raise ValueError("Gate v2 requires a passed construction selection")
    taskbook = load_taskbook(args.taskbook)
    geometry_ids = [str(value) for value in selection["selected_pair"]["geometry_ids"]]
    tasks = [task for task in taskbook["meta_train"] if task.geometry_id in geometry_ids]
    if len(tasks) != 2 or {task.geometry_id for task in tasks} != set(geometry_ids):
        raise ValueError("Gate v2 selected pair must be promoted into meta_train")
    tasks.sort(key=lambda task: geometry_ids.index(task.geometry_id))
    for task in tasks:
        validate_physical_task_contract(task, cfg)
    books = {task.task_id: load_casebook(task, args.casebook_root, required_schema=PHYSICAL_GATE_CASEBOOK_SCHEMA) for task in tasks}
    for task in tasks:
        if len(books[task.task_id]["train_pool"]) != 8 or len(books[task.task_id]["gate_eval_pool"]) != 4:
            raise ValueError("Gate v2 requires exactly 8 train_pool and 4 gate_eval_pool cases per task")
        if {row["case_seed"] for row in books[task.task_id]["train_pool"]} & {row["case_seed"] for row in books[task.task_id]["gate_eval_pool"]}:
            raise ValueError("Gate v2 train/eval seed leakage")
    root = Path(args.output); task_ids = [task.task_id for task in tasks]
    manifest = {
        "schema": "physical_task_policy_heterogeneity_run_v2", "run_name": "physical_task_policy_heterogeneity_gate_v2",
        "source_config_sha256": file_sha256(args.config), "resolved_config_sha256": content_hash(cfg),
        "taskbook_hash": content_hash(taskbook_payload(taskbook)), "casebook_hashes": {task_id: content_hash(books[task_id]) for task_id in task_ids},
        "critical_threshold_hash": calibration["calibration_hash"], "selection_hash": selection["selection_hash"],
        "training_seed_shared_by_both_policies": 0, "environment_steps_per_task": 10_000,
        "training_split": "train_pool", "evaluation_split": "gate_eval_pool", "gate_sac_hyperparameters": GATE_SAC_HYPERPARAMETERS,
        "implementation_hash": _implementation_hash(),
    }
    prepare_run_manifest(root, manifest, resume=args.resume)
    models: dict[str, SAC] = {}
    for task in tasks:
        path = root / "policies" / f"{task.task_id}.zip"
        if args.resume and path.exists():
            models[task.task_id] = SAC.load(path, device="auto")
            continue
        env = LogicalMergeEnv(task, cfg, books[task.task_id]["train_pool"])
        try:
            model = _new_sac(env, 0); model.learn(total_timesteps=10_000)
            path.parent.mkdir(parents=True, exist_ok=True); model.save(path); models[task.task_id] = model
        finally:
            env.close()
    matrix: dict[str, dict[str, dict[str, Any]]] = {}; records: dict[str, dict[str, Any]] = {}
    for source_task in tasks:
        matrix[source_task.task_id] = {}; records[source_task.task_id] = {}
        for target_task in tasks:
            result = _evaluate_sac(models[source_task.task_id], target_task, cfg, books[target_task.task_id]["gate_eval_pool"])
            matrix[source_task.task_id][target_task.task_id] = result["summary"]
            records[source_task.task_id][target_task.task_id] = result["records"]
    diagnostics = {task.task_id: _action_distance(models, task, cfg, books[task.task_id]["gate_eval_pool"], task_ids) for task in tasks}
    gate = physical_heterogeneity_gate_report_v2(matrix, task_ids)
    gate["inputs"] = manifest; gate["action_distance_diagnostics"] = diagnostics
    write_json(root / "physical_task_transfer_matrix_v2.json", {"schema": "physical_task_policy_transfer_matrix_v2", "provenance": manifest, "policy_tasks": task_ids, "evaluation_tasks": task_ids, "matrix": matrix, "records": records, "action_distance_diagnostics": diagnostics})
    write_json(root / "physical_task_heterogeneity_gate_v2.json", gate)
    print(f"Physical task heterogeneity Gate v2: {gate['status']}")
    if gate["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
