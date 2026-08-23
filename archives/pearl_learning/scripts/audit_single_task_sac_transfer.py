"""Gate 1B: short independent SAC fits and a matched 2x2 transfer matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import SAC

from archives.pearl_learning.scripts.run_baselines import _evaluate_sac, _implementation_hash
from archives.pearl_learning.src.benchmark_calibration import apply_calibration_manifest
from archives.pearl_learning.src.casebook import MECHANISM_CASEBOOK_SCHEMA, load_casebook
from archives.pearl_learning.src.io import (
    assert_method_variant_contract,
    content_hash,
    file_sha256,
    prepare_run_manifest,
    read_config,
    write_json,
)
from archives.pearl_learning.src.mechanism_audit import single_task_sac_transfer_report
from archives.pearl_learning.src.mechanism_casebook import validate_matched_mechanism_cases
from archives.pearl_learning.src.task_env import LogicalMergeEnv
from archives.pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _select_tasks(taskbook, requested: list[str]):
    wanted = set(map(str, requested))
    tasks = [task for split in taskbook.values() for task in split if task.task_id in wanted or task.geometry_id in wanted]
    if len(tasks) != 2 or len(tasks) != len(wanted):
        raise ValueError("Gate 1B requires exactly two unique frozen task ids/geometry ids")
    return tasks


GATE_1B_SAC_HYPERPARAMETERS = {
    "learning_starts": 100,
    "buffer_size": 20_000,
    "batch_size": 64,
    "train_freq_steps": 8,
    "gradient_steps": 1,
}


def _new_gate_sac(env, seed: int) -> SAC:
    """A bounded-cost optimizer for feasibility, not a benchmark baseline."""
    return SAC(
        "MlpPolicy", env, seed=seed, verbose=0,
        learning_starts=GATE_1B_SAC_HYPERPARAMETERS["learning_starts"],
        buffer_size=GATE_1B_SAC_HYPERPARAMETERS["buffer_size"],
        batch_size=GATE_1B_SAC_HYPERPARAMETERS["batch_size"],
        train_freq=(GATE_1B_SAC_HYPERPARAMETERS["train_freq_steps"], "step"),
        gradient_steps=GATE_1B_SAC_HYPERPARAMETERS["gradient_steps"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--critical-thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--environment-steps", type=int, default=5_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.environment_steps <= 0:
        raise ValueError("--environment-steps must be positive")
    source_config = read_config(args.config)
    manifest = json.loads(Path(args.critical_thresholds).read_text(encoding="utf-8"))
    config = apply_calibration_manifest(source_config, manifest)
    assert_method_variant_contract(config, "mechanism_gate_1b", "mechanism")
    taskbook = load_taskbook(args.taskbook)
    tasks = _select_tasks(taskbook, args.task_id)
    books = {
        task.task_id: load_casebook(task, args.casebook_root, required_schema=MECHANISM_CASEBOOK_SCHEMA)
        for task in tasks
    }
    cases = {task_id: list(book["train_pool"]) for task_id, book in books.items()}
    if any(not rows for rows in cases.values()):
        raise ValueError("Gate 1B requires mechanism train_pool cases")
    validate_matched_mechanism_cases(cases)
    task_ids = [task.task_id for task in tasks]
    root = Path(args.output)
    run_manifest = {
        "run_name": "gate_1b_single_task_sac_transfer",
        "run_kind": "mechanism_gate",
        "requested_config_path": str(args.config),
        "source_config_sha256": file_sha256(args.config),
        "resolved_config_sha256": content_hash(config),
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "casebook_hashes": {task_id: content_hash(book) for task_id, book in books.items()},
        "critical_threshold_hash": str(manifest["calibration_hash"]),
        "training_seed": int(args.seed),
        "environment_steps_per_task": int(args.environment_steps),
        "evaluation_split": "train_pool_matched_mechanism_only",
        "gate_1b_sac_hyperparameters": GATE_1B_SAC_HYPERPARAMETERS,
        "implementation_hash": _implementation_hash(),
    }
    prepare_run_manifest(root, run_manifest, resume=args.resume)
    models: dict[str, SAC] = {}
    try:
        for index, task in enumerate(tasks):
            env = LogicalMergeEnv(task, config, cases[task.task_id])
            try:
                model = _new_gate_sac(env, int(args.seed) + index)
                model.learn(total_timesteps=int(args.environment_steps))
                model.save(root / f"{task.task_id}_sac_{args.environment_steps}_steps")
                models[task.task_id] = model
            finally:
                env.close()
        matrix: dict[str, dict[str, dict]] = {}
        records: dict[str, dict[str, list[dict]]] = {}
        for source in tasks:
            matrix[source.task_id] = {}
            records[source.task_id] = {}
            for target in tasks:
                result = _evaluate_sac(models[source.task_id], target, config, cases[target.task_id])
                matrix[source.task_id][target.task_id] = result["summary"]
                records[source.task_id][target.task_id] = result["records"]
    finally:
        for model in models.values():
            if getattr(model, "env", None) is not None:
                model.env.close()
    gate = single_task_sac_transfer_report(matrix, task_ids)
    provenance = {
        "run_manifest": run_manifest,
        "task_ids": task_ids,
        "policy_rows": "source_task_trained_policy",
        "evaluation_columns": "target_task_matched_train_pool",
        "not_a_benchmark_or_holdout_result": True,
    }
    write_json(root / "single_task_sac_transfer_matrix.json", {
        "schema": "logical_merge_single_task_sac_transfer_matrix_v1",
        "provenance": provenance,
        "matrix": matrix,
        "records": records,
    })
    write_json(root / "single_task_sac_transfer_gate.json", {**gate, "inputs": provenance})
    print(f"Gate 1B single-task SAC transfer: {gate['status']}")


if __name__ == "__main__":
    main()
