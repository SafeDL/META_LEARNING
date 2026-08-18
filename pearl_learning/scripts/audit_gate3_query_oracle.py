"""Gate 3 query-oracle audit: Gate 1B single-task SAC policies on the query cases.

The four Gate 3 validation_query cases were added without the task-specific
feasibility screening that produced the train pool.  Before Stage C can ever
be judged, the frozen query cases must demonstrably support task-dependent
success: the two Gate 1B single-task SAC policies are re-evaluated here as a
2x2 transfer matrix on exactly those query cases.  No model is trained; this
audit costs roughly sixteen episodes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from pearl_learning.scripts.run_baselines import _evaluate_sac
from pearl_learning.src.benchmark_calibration import resolve_calibration
from pearl_learning.src.casebook import MECHANISM_CASEBOOK_SCHEMA, load_casebook
from pearl_learning.src.causal_audit import _initial_states
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _select_pair(taskbook, requested: list[str]):
    wanted = set(map(str, requested))
    tasks = [
        task
        for split in taskbook.values()
        for task in split
        if task.task_id in wanted or task.geometry_id in wanted
    ]
    if len(tasks) != 2 or len(tasks) != len(wanted):
        raise ValueError("the query oracle audit requires exactly two unique frozen task ids")
    return tasks


def _load_sac_model(sac_dir: Path, task_id: str) -> SAC:
    candidates = sorted(sac_dir.glob(f"{task_id}_sac_*.zip"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one Gate 1B SAC checkpoint for {task_id} in {sac_dir}, found {len(candidates)}"
        )
    return SAC.load(str(candidates[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--critical-thresholds", required=True)
    parser.add_argument("--sac-checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    args = parser.parse_args()
    cfg = resolve_calibration(read_config(args.config), args.critical_thresholds)
    taskbook = load_taskbook(args.taskbook)
    tasks = _select_pair(taskbook, args.task_id)
    books = {
        task.task_id: load_casebook(task, args.casebook_root, required_schema=MECHANISM_CASEBOOK_SCHEMA)
        for task in tasks
    }
    query_cases = {task_id: list(book["validation_query"]) for task_id, book in books.items()}
    if any(len(cases) != 4 for cases in query_cases.values()):
        raise ValueError("the query oracle audit requires four validation_query cases per task")
    sac_dir = Path(args.sac_checkpoint_dir)
    models = {task.task_id: _load_sac_model(sac_dir, task.task_id) for task in tasks}
    matrix: dict[str, dict[str, dict]] = {}
    records: dict[str, dict[str, list[dict]]] = {}
    try:
        for source in tasks:
            matrix[source.task_id] = {}
            records[source.task_id] = {}
            for target in tasks:
                result = _evaluate_sac(models[source.task_id], target, cfg, query_cases[target.task_id])
                matrix[source.task_id][target.task_id] = result["summary"]
                records[source.task_id][target.task_id] = result["records"]
    finally:
        for model in models.values():
            if getattr(model, "env", None) is not None:
                model.env.close()
    state_actions: dict[str, dict[str, list[list[float]]]] = {}
    task_probes: dict[str, dict[str, object]] = {}
    for task in tasks:
        states = _initial_states(task, cfg, query_cases[task.task_id])
        actions = {}
        for source in tasks:
            rows = [
                models[source.task_id].predict(state, deterministic=True)[0].tolist()
                for state in states
            ]
            actions[source.task_id] = rows
        state_actions[task.task_id] = actions
        a_rows = [np.asarray(row, dtype=float) for row in actions[tasks[0].task_id]]
        b_rows = [np.asarray(row, dtype=float) for row in actions[tasks[1].task_id]]
        task_probes[task.task_id] = {
            "state_bank_size": int(len(states)),
            "state_bank_hash": content_hash([state.tolist() for state in states]),
            "single_task_actions": actions,
            "single_task_action_l2_mean": float(
                np.mean([np.linalg.norm(a_row - b_row) for a_row, b_row in zip(a_rows, b_rows)])
            ),
        }
    task_ids = [task.task_id for task in tasks]
    diagonal_advantage = {
        task_id: (
            float(matrix[task_id][task_id]["valid_critical_strict_rate"])
            - float(matrix[other][task_id]["valid_critical_strict_rate"])
        )
        for task_id, other in ((task_ids[0], task_ids[1]), (task_ids[1], task_ids[0]))
    }
    achievable = {
        task_id: sum(bool(row["valid_critical_strict"]) for row in records[task_id][task_id])
        for task_id in task_ids
    }
    feasibility = {
        "schema": "gate3_query_oracle_feasibility_v1",
        "status": (
            "pass"
            if all(advantage > 0.0 for advantage in diagonal_advantage.values())
            and all(count >= 2 for count in achievable.values())
            else "fail"
        ),
        "diagonal_vcsr_advantage": diagonal_advantage,
        "query_cases_with_strict_vcsr": achievable,
        "requirements": {
            "diagonal_advantage_positive_for_both_tasks": True,
            "minimum_achievable_query_cases_per_task": 2,
        },
    }
    provenance = {
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "casebook_hashes": {task_id: content_hash(book) for task_id, book in books.items()},
        "config_hash": content_hash(cfg),
        "critical_threshold_hash": cfg["critical_metric"]["calibration_hash"],
        "sac_checkpoint_dir": str(sac_dir),
        "not_a_benchmark_or_holdout_result": True,
    }
    root = Path(args.output)
    write_json(root / "gate3_query_oracle_audit.json", {
        "schema": "gate3_query_oracle_audit_suite_v1",
        "evaluation_metric_schema": str(cfg["critical_metric"]["schema"]),
        "threshold_source_metric_schema": str(cfg["critical_metric"]["threshold_source_metric_schema"]),
        "query_case_count_per_task": {task_id: len(cases) for task_id, cases in query_cases.items()},
        "matrix": matrix,
        "records": records,
        "feasibility": feasibility,
        "tasks": task_probes,
        "provenance": provenance,
    })
    print(f"Gate 3 query oracle audit: feasibility={feasibility['status']} "
          f"(diagonal advantage {diagonal_advantage}, achievable {achievable})")


if __name__ == "__main__":
    main()
