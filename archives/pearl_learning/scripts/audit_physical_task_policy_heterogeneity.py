"""Train two physical-task SACs and evaluate a frozen 2x2 transfer matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from stable_baselines3 import SAC

from archives.pearl_learning.scripts.run_baselines import _evaluate_sac, _implementation_hash
from archives.pearl_learning.src.benchmark_calibration import apply_calibration_manifest
from archives.pearl_learning.src.casebook import CASEBOOK_SCHEMA, load_casebook
from archives.pearl_learning.src.io import (
    content_hash,
    file_sha256,
    prepare_run_manifest,
    read_config,
    write_json,
)
from archives.pearl_learning.src.task_env import LogicalMergeEnv
from archives.pearl_learning.src.taskbook import (
    load_taskbook,
    taskbook_payload,
    validate_physical_task_contract,
)


GATE_SAC_HYPERPARAMETERS = {
    "learning_starts": 100,
    "buffer_size": 20_000,
    "batch_size": 64,
    "train_freq_steps": 8,
    "gradient_steps": 1,
}


def physical_heterogeneity_gate_report(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    task_ids: list[str],
    *,
    minimum_advantage: float,
) -> dict[str, Any]:
    """Apply only the predeclared two-sided diagonal VCSR criterion."""
    if len(task_ids) != 2 or len(set(task_ids)) != 2:
        raise ValueError("physical heterogeneity Gate requires exactly two tasks")
    if set(matrix) != set(task_ids) or any(
        set(matrix[source]) != set(task_ids) for source in task_ids
    ):
        raise ValueError("physical heterogeneity transfer matrix is incomplete")
    first, second = task_ids
    advantages = {
        first: (
            float(matrix[first][first]["valid_critical_strict_rate"])
            - float(matrix[second][first]["valid_critical_strict_rate"])
        ),
        second: (
            float(matrix[second][second]["valid_critical_strict_rate"])
            - float(matrix[first][second]["valid_critical_strict_rate"])
        ),
    }
    passed = all(value >= float(minimum_advantage) for value in advantages.values())
    return {
        "schema": "physical_task_policy_heterogeneity_gate_v1",
        "gate_name": "physical_task_policy_heterogeneity",
        "status": "pass" if passed else "fail",
        "task_ids": list(task_ids),
        "hard_gate_metric": "valid_critical_strict_rate",
        "diagonal_vcsr_advantage": advantages,
        "minimum_diagonal_vcsr_advantage": float(minimum_advantage),
        "action_distance_is_diagnostic_only": True,
        "next_allowed_stage": "vanilla_pearl_physical_pilot" if passed else None,
        "failure_action": (
            None if passed else "revise_physical_task_or_case_distribution; do_not_tune_pearl"
        ),
    }


def _select_tasks(taskbook: Mapping[str, list[Any]], requested: list[str]) -> list[Any]:
    wanted = set(map(str, requested))
    tasks = [
        task for task in taskbook["meta_train"]
        if task.geometry_id in wanted or task.task_id in wanted
    ]
    if len(tasks) != 2 or {task.geometry_id for task in tasks} != wanted:
        raise ValueError("configured Gate ids must select exactly two meta-train geometries")
    positions = {value: index for index, value in enumerate(requested)}
    return sorted(tasks, key=lambda task: positions[task.geometry_id])


def _new_sac(env: LogicalMergeEnv, seed: int) -> SAC:
    return SAC(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        learning_starts=GATE_SAC_HYPERPARAMETERS["learning_starts"],
        buffer_size=GATE_SAC_HYPERPARAMETERS["buffer_size"],
        batch_size=GATE_SAC_HYPERPARAMETERS["batch_size"],
        train_freq=(GATE_SAC_HYPERPARAMETERS["train_freq_steps"], "step"),
        gradient_steps=GATE_SAC_HYPERPARAMETERS["gradient_steps"],
    )


def _initial_observations(
    task: Any,
    config: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
) -> np.ndarray:
    env = LogicalMergeEnv(task, config, cases)
    rows = []
    try:
        for case in cases:
            observation, _ = env.reset(options={"case": case})
            rows.append(np.asarray(observation, dtype=np.float32))
    finally:
        env.close()
    return np.stack(rows)


def _action_distance(
    models: Mapping[str, SAC],
    task: Any,
    config: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
    task_ids: list[str],
) -> dict[str, Any]:
    observations = _initial_observations(task, config, cases)
    first_actions, _ = models[task_ids[0]].predict(observations, deterministic=True)
    second_actions, _ = models[task_ids[1]].predict(observations, deterministic=True)
    distances = np.linalg.norm(
        np.asarray(first_actions, dtype=float) - np.asarray(second_actions, dtype=float), axis=-1,
    )
    return {
        "state_bank": "frozen_gate_query_initial_observations",
        "state_count": len(observations),
        "mean_action_l2": float(np.mean(distances)),
        "max_action_l2": float(np.max(distances)),
        "per_case_action_l2": {
            str(case["case_id"]): float(value) for case, value in zip(cases, distances)
        },
        "diagnostic_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--critical-thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_config = read_config(args.config)
    calibration = json.loads(Path(args.critical_thresholds).read_text(encoding="utf-8"))
    config = apply_calibration_manifest(source_config, calibration)
    gate_cfg = dict(config["physical_heterogeneity_gate"])
    requested = [str(value) for value in gate_cfg["task_ids"]]
    seed = int(gate_cfg["training_seed"])
    environment_steps = int(gate_cfg["environment_steps_per_task"])
    minimum_advantage = float(gate_cfg["minimum_diagonal_vcsr_advantage"])
    training_split = str(gate_cfg["training_split"])
    evaluation_split = str(gate_cfg["evaluation_split"])
    if environment_steps != 10_000 or seed != 0:
        raise ValueError("the physical pilot freezes Gate training at seed=0 and 10,000 steps")
    if (training_split, evaluation_split) != ("train_pool", "validation_query"):
        raise ValueError("the physical Gate requires train_pool -> validation_query isolation")

    taskbook = load_taskbook(args.taskbook)
    tasks = _select_tasks(taskbook, requested)
    for task in tasks:
        validate_physical_task_contract(task, config)
    books = {
        task.task_id: load_casebook(task, args.casebook_root, required_schema=CASEBOOK_SCHEMA)
        for task in tasks
    }
    for task in tasks:
        book = books[task.task_id]
        if len(book[training_split]) != 8 or len(book[evaluation_split]) != 4:
            raise ValueError("each Gate task requires exactly 8 train and 4 query cases")
        train_ids = {str(row["case_id"]) for row in book[training_split]}
        query_ids = {str(row["case_id"]) for row in book[evaluation_split]}
        train_seeds = {int(row["case_seed"]) for row in book[training_split]}
        query_seeds = {int(row["case_seed"]) for row in book[evaluation_split]}
        if train_ids & query_ids or train_seeds & query_seeds:
            raise ValueError("Gate train/query cases are not disjoint")

    task_ids = [task.task_id for task in tasks]
    root = Path(args.output)
    run_manifest = {
        "schema": "physical_task_policy_heterogeneity_run_v1",
        "run_name": "physical_task_policy_heterogeneity_gate",
        "requested_config_path": str(args.config),
        "source_config_sha256": file_sha256(args.config),
        "resolved_config_sha256": content_hash(config),
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "casebook_hashes": {task_id: content_hash(books[task_id]) for task_id in task_ids},
        "critical_threshold_hash": str(calibration["calibration_hash"]),
        "training_seed_shared_by_both_policies": seed,
        "environment_steps_per_task": environment_steps,
        "training_split": training_split,
        "evaluation_split": evaluation_split,
        "gate_sac_hyperparameters": GATE_SAC_HYPERPARAMETERS,
        "implementation_hash": _implementation_hash(),
    }
    prepare_run_manifest(root, run_manifest, resume=args.resume)

    models: dict[str, SAC] = {}
    for task in tasks:
        target = root / "policies" / f"{task.task_id}.zip"
        if args.resume and target.exists():
            models[task.task_id] = SAC.load(target, device="auto")
            continue
        env = LogicalMergeEnv(task, config, books[task.task_id][training_split])
        try:
            model = _new_sac(env, seed)
            model.learn(total_timesteps=environment_steps)
            target.parent.mkdir(parents=True, exist_ok=True)
            model.save(target)
            models[task.task_id] = model
        finally:
            env.close()

    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for source in tasks:
        matrix[source.task_id] = {}
        records[source.task_id] = {}
        for target in tasks:
            result = _evaluate_sac(
                models[source.task_id], target, config, books[target.task_id][evaluation_split],
            )
            matrix[source.task_id][target.task_id] = result["summary"]
            records[source.task_id][target.task_id] = result["records"]

    action_diagnostics = {
        target.task_id: _action_distance(
            models,
            target,
            config,
            books[target.task_id][evaluation_split],
            task_ids,
        )
        for target in tasks
    }
    gate = physical_heterogeneity_gate_report(
        matrix, task_ids, minimum_advantage=minimum_advantage,
    )
    gate["inputs"] = run_manifest
    gate["action_distance_diagnostics"] = action_diagnostics
    write_json(root / "physical_task_transfer_matrix.json", {
        "schema": "physical_task_policy_transfer_matrix_v1",
        "provenance": run_manifest,
        "policy_tasks": task_ids,
        "evaluation_tasks": task_ids,
        "matrix": matrix,
        "records": records,
        "action_distance_diagnostics": action_diagnostics,
    })
    write_json(root / "physical_task_heterogeneity_gate.json", gate)
    print(f"Physical task heterogeneity Gate: {gate['status']}")
    if gate["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
