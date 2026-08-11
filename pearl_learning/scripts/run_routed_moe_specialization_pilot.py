"""Run validation-only posterior separability and router specialization pilots."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from pearl_learning.src.casebook import load_casebook, physical_geometry_id
from pearl_learning.src.checkpoint import load_checkpoint
from pearl_learning.src.evaluator import compact_fewshot_result, evaluate_fewshot
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.pearl_trainer import train
from pearl_learning.src.posterior_adaptation_analysis import posterior_pair_audit
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _variant_config(base: Mapping[str, Any], input_mode: str, balance_weight: float) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    config["experiment"] = {**config["experiment"], "run_kind": "routed_moe_specialization_pilot"}
    config["networks"]["actor_architecture"] = "posterior_routed_moe"
    config["networks"]["moe"] = {
        **config["networks"]["moe"],
        "input_mode": input_mode,
        "load_balance_weight": float(balance_weight),
    }
    return config


def _restore_rng(payload: Mapping[str, Any]) -> None:
    state = payload["rng_state"]
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu").clone())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([
            torch.as_tensor(item, dtype=torch.uint8, device="cpu").clone()
            for item in state["cuda"]
        ])


def _support_signature(result: Mapping[str, Any]) -> dict[tuple[str, str], tuple[Any, ...]]:
    return {
        (task_id, shot): (
            tuple(values["support_case_ids"]), values["context_sample_hash"],
            tuple(values["support_episode_lengths"]),
        )
        for task_id, shots in result["tasks"].items()
        for shot, values in shots.items()
    }


def _route_summary(result: Mapping[str, Any], tasks: list[Any]) -> dict[str, Any]:
    by_geometry: dict[str, list[str]] = {}
    for task in tasks:
        by_geometry.setdefault(physical_geometry_id(task.geometry_id), []).append(task.task_id)
    trajectories = []
    pair_distances: dict[str, dict[str, float]] = {}
    for task_id, shots in result["tasks"].items():
        w0 = np.asarray(shots["0"]["router_audit"]["routing_weights"], dtype=float)
        for shot, values in shots.items():
            route = values["router_audit"]
            weights = np.asarray(route["routing_weights"], dtype=float)
            trajectories.append({
                "task_id": task_id,
                "k": int(shot),
                "weights": weights.tolist(),
                "l1_from_k0": float(np.abs(weights - w0).sum()),
                "entropy": float(route["entropy"]),
                "top_k_indexes": route["top_k_indexes"],
            })
    for geometry, task_ids in by_geometry.items():
        if len(task_ids) != 2:
            continue
        left, right = task_ids
        pair_distances[geometry] = {}
        for shot in result["tasks"][left]:
            left_weights = np.asarray(result["tasks"][left][shot]["router_audit"]["routing_weights"], dtype=float)
            right_weights = np.asarray(result["tasks"][right][shot]["router_audit"]["routing_weights"], dtype=float)
            pair_distances[geometry][shot] = float(np.abs(left_weights - right_weights).sum())
    changed = [row["l1_from_k0"] for row in trajectories if row["k"] > 0]
    return {
        "mean_l1_from_k0_after_support": float(np.mean(changed)),
        "max_l1_from_k0_after_support": float(np.max(changed)),
        "mean_entropy_after_support": float(np.mean([
            row["entropy"] for row in trajectories if row["k"] > 0
        ])),
        "rule_pair_route_l1_by_geometry_and_k": pair_distances,
        "trajectory": trajectories,
    }


def _vcsr_difference(full: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, list[float]] = {}
    for task_id, shots in full["tasks"].items():
        for shot, values in shots.items():
            result.setdefault(shot, []).append(
                float(values["summary"]["valid_critical_strict_rate"])
                - float(frozen["tasks"][task_id][shot]["summary"]["valid_critical_strict_rate"])
            )
    return {shot: float(np.mean(values)) for shot, values in result.items()}


def _summarize_variant(
    input_mode: str,
    balance_weight: float,
    root: Path,
    full: Mapping[str, Any],
    frozen_router: Mapping[str, Any],
    validation_tasks: list[Any],
    shots: list[int],
) -> dict[str, Any]:
    training = json.loads((root / "training_summary.json").read_text(encoding="utf-8"))
    return {
        "input_mode": input_mode,
        "load_balance_weight": float(balance_weight),
        "checkpoint": full["provenance"]["checkpoint"],
        "environment_steps": int(training["environment_steps"]),
        "gradient_updates": int(training["gradient_updates"]),
        "posterior_separability": posterior_pair_audit(
            [full], validation_tasks, shots=shots, samples=1000, confidence=0.95,
        ),
        "routing": _route_summary(full, validation_tasks),
        "full_minus_frozen_router_vcsr": _vcsr_difference(full, frozen_router),
        "support_signatures_match": _support_signature(full) == _support_signature(frozen_router),
        "parameter_invariance": (
            full["parameter_hash_before"] == full["parameter_hash_after"]
            and frozen_router["parameter_hash_before"] == frozen_router["parameter_hash_after"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pearl_learning/configs/posterior_routed_moe_specialization_pilot.yaml")
    parser.add_argument("--taskbook", default="results/pearl_learning/posterior_adaptation/taskbooks")
    parser.add_argument("--casebook-root", default="results/pearl_learning/posterior_adaptation")
    parser.add_argument("--output-root", default="results/pearl_learning/posterior_routed_moe_mechanism/specialization_pilot")
    parser.add_argument("--max-env-steps", type=int)
    args = parser.parse_args()

    base = read_config(args.config)
    pilot = base["specialization_pilot"]
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    taskbook = load_taskbook(args.taskbook)
    train_tasks = taskbook["meta_train"][:int(pilot["train_task_count"])]
    validation_tasks = taskbook["meta_validation"][:int(pilot["validation_task_count"])]
    casebooks = {
        task.task_id: load_casebook(task, args.casebook_root)
        for task in train_tasks + validation_tasks
    }
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    seed = int(pilot["training_seed"])
    max_steps = int(args.max_env_steps or pilot["environment_steps_per_variant"])
    variants: dict[str, Any] = {}

    for input_mode in pilot["input_modes"]:
        for balance_weight in pilot["load_balance_weights"]:
            name = f"{input_mode}__balance_{float(balance_weight):.4f}"
            config = _variant_config(base, str(input_mode), float(balance_weight))
            config["project"] = {**config["project"], "output_root": (output / "runs").as_posix()}
            run_name = f"{name}__seed_{seed}__steps_{max_steps}"
            root = Path(config["project"]["output_root"]) / "models" / run_name
            artifact_path = output / "variants" / f"{name}.json"
            if artifact_path.exists():
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                variants[name] = _summarize_variant(
                    str(input_mode), float(balance_weight), root, artifact["full"],
                    artifact["frozen_router"], validation_tasks,
                    [int(value) for value in base["evaluation"]["shots"]],
                )
                continue
            root = train(config, train_tasks, [], casebooks, taskbook_hash, max_steps, seed, run_name, diagnostic_run=True)
            checkpoint = root / "best_model.pt"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            agent = PEARLAgent(
                int(config["environment"]["observation_dim"]),
                int(config["environment"]["action_dim"]),
                config,
                device,
            )
            payload = load_checkpoint(checkpoint, agent, device)
            _restore_rng(payload)
            provenance = {"variant": name, "training_seed": seed, "checkpoint": checkpoint.as_posix()}
            full = compact_fewshot_result(evaluate_fewshot(
                agent, config, validation_tasks, casebooks, "meta_validation",
                int(pilot["query_cases_per_task"]), provenance, "fixed", "posterior_sampled",
                mechanism_audit=True,
            ))
            _restore_rng(payload)
            frozen_router = compact_fewshot_result(evaluate_fewshot(
                agent, config, validation_tasks, casebooks, "meta_validation",
                int(pilot["query_cases_per_task"]), {**provenance, "intervention": "frozen_router"},
                "fixed", "posterior_sampled", "adaptive", "frozen_prior", mechanism_audit=False,
            ))
            variants[name] = _summarize_variant(
                str(input_mode), float(balance_weight), root, full, frozen_router,
                validation_tasks, [int(value) for value in base["evaluation"]["shots"]],
            )
            write_json(artifact_path, {"full": full, "frozen_router": frozen_router})

    summary = {
        "schema": "posterior_routed_moe_specialization_pilot_v1",
        "status": "PILOT",
        "scope": pilot["scope"],
        "test_split_used": False,
        "taskbook_hash": taskbook_hash,
        "training_seed": seed,
        "environment_steps_per_variant": max_steps,
        "train_task_ids": [task.task_id for task in train_tasks],
        "validation_task_ids": [task.task_id for task in validation_tasks],
        "variants": variants,
        "interpretation_rule": (
            "posterior separation without route movement indicates router learning failure; "
            "route movement without full-minus-frozen-router degradation is not causal routing evidence"
        ),
    }
    write_json(output / "summary.json", summary)
    print(f"Posterior-routed MoE specialization pilot completed: {output}")


if __name__ == "__main__":
    main()
