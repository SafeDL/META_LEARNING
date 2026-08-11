"""Run the bounded posterior-routed MoE mechanism pilot."""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.checkpoint import load_checkpoint
from pearl_learning.src.evaluator import compact_fewshot_result, evaluate_fewshot
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.pearl_trainer import train
from pearl_learning.src.task_env import freeze_physical_task_descriptor
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


METHODS = ("sac", "moe_sac", "pearl", "pearl_moe")
INTERVENTIONS = {
    "full": ("adaptive", "adaptive", None),
    "frozen_router": ("adaptive", "frozen_prior", None),
    "frozen_latent": ("frozen_prior", "adaptive", None),
    "both_frozen": ("frozen_prior", "frozen_prior", None),
    "uniform": ("adaptive", "uniform", None),
    "knockout_0": ("adaptive", "adaptive", 0),
    "knockout_1": ("adaptive", "adaptive", 1),
}
METRICS = (
    "valid_critical_strict_rate", "target_collision_rate", "critical_rate",
    "invalid_rate", "mean_episode_return", "median_min_ttc", "median_min_distance",
)


def _method_config(base: Mapping[str, Any], method: str) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    config.pop("task_representation", None)
    config["experiment"] = {**config["experiment"], "run_kind": "routed_moe_mechanism_pilot"}
    config["ablation"] = {
        **config.get("ablation", {}),
        "no_context_training": method in {"sac", "moe_sac"},
    }
    if method in {"sac", "pearl"}:
        config["networks"]["actor_architecture"] = "dense"
    else:
        config["networks"]["actor_architecture"] = "posterior_routed_moe"
        config["networks"]["moe"]["input_mode"] = (
            "static" if method == "moe_sac" else "static_posterior_mean_logvar"
        )
    return config


def _restore_rng(payload: Mapping[str, Any]) -> None:
    state = payload["rng_state"]
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu").clone())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([
            torch.as_tensor(item, dtype=torch.uint8, device="cpu").clone()
            for item in state["cuda"]
        ])


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parameter_count(module: nn.Module) -> tuple[int, int]:
    return sum(parameter.numel() for parameter in module.parameters()), sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )


def _linear_flops(module: nn.Module, operation: Callable[[], Any]) -> int:
    total = 0

    def hook(layer: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        rows = int(inputs[0].numel() // layer.in_features)
        total += 2 * rows * layer.in_features * layer.out_features

    handles = [layer.register_forward_hook(hook) for layer in module.modules() if isinstance(layer, nn.Linear)]
    try:
        with torch.no_grad():
            operation()
    finally:
        for handle in handles:
            handle.remove()
    return total


def _profile_agent(agent: PEARLAgent, config: Mapping[str, Any], task: Any,
                   casebook: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    total, trainable = _parameter_count(agent.context_encoder)
    modules = [agent.actor, agent.q1, agent.q2, agent.target_q1, agent.target_q2]
    if agent.router is not None:
        modules.append(agent.router)
    total += sum(_parameter_count(module)[0] for module in modules) + agent.log_alpha.numel()
    trainable += sum(_parameter_count(module)[1] for module in modules) + agent.log_alpha.numel()
    actor_total, actor_trainable = _parameter_count(agent.actor)
    router_total, router_trainable = (0, 0) if agent.router is None else _parameter_count(agent.router)
    observation = torch.zeros((1, agent.observation_dim), device=agent.device)
    mu, log_var = agent.prior()
    route = None
    if agent.router is not None:
        descriptor = freeze_physical_task_descriptor(task, config, casebook["validation_support"])
        route = agent.compute_route(descriptor, mu, log_var, 0)
    actor_flops = _linear_flops(agent.actor, lambda: agent.act(observation, mu, True, route))
    router_flops = 0
    if agent.router is not None:
        descriptor = freeze_physical_task_descriptor(task, config, casebook["validation_support"])
        router_flops = _linear_flops(
            agent.router, lambda: agent.compute_route(descriptor, mu, log_var, 0),
        )
    for _ in range(20):
        agent.act(observation, mu, True, route)
    if agent.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    iterations = 200
    for _ in range(iterations):
        agent.act(observation, mu, True, route)
    if agent.device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = 1000.0 * (time.perf_counter() - start) / iterations
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "actor_parameters": actor_total,
        "actor_trainable_parameters": actor_trainable,
        "router_parameters": router_total,
        "router_trainable_parameters": router_trainable,
        "actor_linear_flops_batch_1": actor_flops,
        "router_linear_flops_per_task": router_flops,
        "deterministic_actor_latency_ms_batch_1": latency_ms,
        "device": str(agent.device),
    }


def _rows(method: str, seed: int, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task_id, shots in result["tasks"].items():
        for shot, values in shots.items():
            rows.append({
                "schema": "routed_moe_pilot_method_task_k_v1",
                "method": method,
                "training_seed": seed,
                "task_id": task_id,
                "k": int(shot),
                **values["summary"],
                "posterior_mean": values["posterior_mean"],
                "posterior_log_variance": values["posterior_log_variance"],
                "support_case_ids": values["support_case_ids"],
                "context_sample_hash": values["context_sample_hash"],
                "router_audit": values.get("router_audit"),
            })
    return rows


def _mean_by_k(rows: list[Mapping[str, Any]], method: str, metric: str, k: int) -> float:
    values = [float(row[metric]) for row in rows if row["method"] == method and int(row["k"]) == k]
    return float(np.mean(values))


def _factorial(rows: list[Mapping[str, Any]], shots: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in shots:
        by_metric = {}
        for metric in METRICS:
            means = {method: _mean_by_k(rows, method, metric, k) for method in METHODS}
            by_metric[metric] = {
                "method_means": means,
                "delta_meta": means["pearl"] - means["sac"],
                "delta_moe_without_meta": means["moe_sac"] - means["sac"],
                "delta_moe_with_meta": means["pearl_moe"] - means["pearl"],
                "delta_interaction": (
                    means["pearl_moe"] - means["pearl"]
                    - means["moe_sac"] + means["sac"]
                ),
            }
        result[str(k)] = by_metric
    return {
        "schema": "routed_moe_pilot_factorial_effects_v1",
        "statistical_unit": "task",
        "seed_count": 1,
        "interval_estimates": None,
        "interpretation": "descriptive pilot effects only",
        "by_k": result,
    }


def _support_signature(result: Mapping[str, Any]) -> dict[tuple[str, str], tuple[Any, ...]]:
    return {
        (task_id, shot): (
            tuple(values["support_case_ids"]), values["context_sample_hash"],
            tuple(values["support_episode_lengths"]), values["support_environment_steps"],
            tuple(np.asarray(values["posterior_mean"]).reshape(-1)),
        )
        for task_id, shots in result["tasks"].items()
        for shot, values in shots.items()
    }


def _route_weights(values: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(values["router_audit"]["routing_weights"], dtype=float)


def _mechanism_summary(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    full = results["full"]
    signatures = {name: _support_signature(result) for name, result in results.items()}
    support_matched = all(signature == signatures["full"] for signature in signatures.values())
    trajectories = []
    action_distances = []
    for task_id, shots in full["tasks"].items():
        w0 = _route_weights(shots["0"])
        for shot, values in shots.items():
            weights = _route_weights(values)
            audit = values.get("expert_action_audit") or {}
            pairwise = audit.get("pairwise_mean_action_l2", {})
            action_distances.extend(float(value) for value in pairwise.values())
            trajectories.append({
                "task_id": task_id,
                "k": int(shot),
                "weights": weights.tolist(),
                "l1_from_k0": float(np.abs(weights - w0).sum()),
                "entropy": float(values["router_audit"]["entropy"]),
                "expert_action_pairwise_mean_l2": pairwise,
            })
    route_changed = any(row["k"] > 0 and row["l1_from_k0"] > 1e-8 for row in trajectories)
    mean_weights = np.mean(np.asarray([row["weights"] for row in trajectories]), axis=0)
    no_collapse = bool(np.max(mean_weights) < 0.95 and np.min(mean_weights) > 0.05)
    intervention_checks = []
    for task_id, shots in results["frozen_router"]["tasks"].items():
        prior = _route_weights(shots["0"])
        intervention_checks.extend(np.allclose(_route_weights(values), prior) for values in shots.values())
    for shots in results["uniform"]["tasks"].values():
        intervention_checks.extend(np.allclose(_route_weights(values), [0.5, 0.5]) for values in shots.values())
    for index, name in enumerate(("knockout_0", "knockout_1")):
        for shots in results[name]["tasks"].values():
            intervention_checks.extend(abs(_route_weights(values)[index]) < 1e-12 for values in shots.values())
    return {
        "schema": "routed_moe_pilot_routing_mechanisms_v1",
        "support_signatures_identical": support_matched,
        "posterior_dependent_route_change": route_changed,
        "mean_expert_weights": mean_weights.tolist(),
        "no_expert_load_collapse": no_collapse,
        "mean_pairwise_expert_action_l2": float(np.mean(action_distances)),
        "nonzero_expert_action_difference": bool(action_distances and np.mean(action_distances) > 1e-8),
        "intervention_contracts_hold": all(intervention_checks),
        "routing_trajectory": trajectories,
    }


def _intervention_effects(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = {name: _rows(name, 0, result) for name, result in results.items()}
    effects = {}
    for name, current in rows.items():
        if name == "full":
            continue
        effects[name] = {}
        for k in sorted({int(row["k"]) for row in current}):
            effects[name][str(k)] = {
                metric: _mean_by_k(rows["full"], "full", metric, k)
                - _mean_by_k(current, name, metric, k)
                for metric in METRICS
            }
    return {
        "schema": "routed_moe_pilot_intervention_effects_v1",
        "effect_direction": "full_minus_intervention",
        "shared_support_verified_separately": True,
        "effects": effects,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pearl_learning/configs/posterior_routed_moe_pilot.yaml")
    parser.add_argument("--taskbook", default="results/pearl_learning/posterior_adaptation/taskbooks")
    parser.add_argument("--casebook-root", default="results/pearl_learning/posterior_adaptation")
    parser.add_argument("--output-root", default="results/pearl_learning/posterior_routed_moe_mechanism/pilot")
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    base = read_config(args.config)
    pilot = base["routed_moe_pilot"]
    seed = int(pilot["training_seed"])
    max_steps = int(args.max_env_steps or pilot["environment_steps_per_method"])
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    base["project"] = {**base["project"], "output_root": (output / "runs").as_posix()}
    taskbook = load_taskbook(args.taskbook)
    train_tasks = taskbook["meta_train"][:int(pilot["train_task_count"])]
    validation_tasks = taskbook["meta_validation"][:int(pilot["validation_task_count"])]
    selected = train_tasks + validation_tasks
    casebooks = {task.task_id: load_casebook(task, args.casebook_root) for task in selected}
    taskbook_hash = content_hash(taskbook_payload(taskbook))

    method_results: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    checkpoints: dict[str, str] = {}
    for method in METHODS:
        config = _method_config(base, method)
        run_name = f"{method}_seed_{seed}_steps_{max_steps}"
        root = Path(config["project"]["output_root"]) / "models" / run_name
        if not args.skip_training:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            root = train(
                config, train_tasks, [], casebooks, taskbook_hash, max_steps, seed,
                run_name, diagnostic_run=True,
            )
            wallclock = time.perf_counter() - started
            peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            write_json(root / "mechanism_runtime.json", {
                "training_wallclock_seconds": wallclock,
                "peak_cuda_memory_bytes": peak_memory,
                "requested_environment_steps": max_steps,
            })
        checkpoint = root / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing pilot checkpoint: {checkpoint}")
        checkpoints[method] = checkpoint.as_posix()
        device = torch.device("cuda" if torch.cuda.is_available() and config["experiment"].get("device") != "cpu" else "cpu")
        agent = PEARLAgent(int(config["environment"]["observation_dim"]), int(config["environment"]["action_dim"]), config, device)
        payload = load_checkpoint(checkpoint, agent, device)
        _restore_rng(payload)
        adaptation = "no_context" if method in {"sac", "moe_sac"} else "posterior_sampled"
        result = compact_fewshot_result(evaluate_fewshot(
            agent, config, validation_tasks, casebooks, "meta_validation",
            int(pilot["query_cases_per_task"]),
            {"method": method, "training_seed": seed, "checkpoint": checkpoint.as_posix()},
            "fixed", adaptation, mechanism_audit=method == "pearl_moe",
        ))
        method_results[method] = result
        runtime = json.loads((root / "mechanism_runtime.json").read_text(encoding="utf-8"))
        training_summary = json.loads((root / "training_summary.json").read_text(encoding="utf-8"))
        profiles[method] = {
            **_profile_agent(agent, config, validation_tasks[0], casebooks[validation_tasks[0].task_id]),
            **runtime,
            "environment_steps": int(payload["step"]),
            "gradient_updates": int(training_summary["gradient_updates"]),
            "architecture": agent.architecture_metadata(),
        }
        write_json(output / "evaluations" / f"{method}.json", result)

    moe_config = _method_config(base, "pearl_moe")
    moe_agent = PEARLAgent(int(moe_config["environment"]["observation_dim"]), int(moe_config["environment"]["action_dim"]), moe_config,
                           torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    moe_payload = load_checkpoint(checkpoints["pearl_moe"], moe_agent, moe_agent.device)
    intervention_results: dict[str, Any] = {}
    for name, (latent_mode, route_mode, knockout) in INTERVENTIONS.items():
        _restore_rng(moe_payload)
        intervention_results[name] = compact_fewshot_result(evaluate_fewshot(
            moe_agent, moe_config, validation_tasks, casebooks, "meta_validation",
            int(pilot["query_cases_per_task"]),
            {"method": "pearl_moe", "training_seed": seed, "intervention": name},
            "fixed", "posterior_sampled", latent_mode, route_mode, knockout,
            mechanism_audit=name == "full",
        ))
        write_json(output / "interventions" / f"{name}.json", intervention_results[name])

    method_rows = [row for method, result in method_results.items() for row in _rows(method, seed, result)]
    _write_jsonl(output / "metrics_by_method_seed_task_k.jsonl", method_rows)
    factorial = _factorial(method_rows, [int(value) for value in base["evaluation"]["shots"]])
    mechanisms = _mechanism_summary(intervention_results)
    interventions = _intervention_effects(intervention_results)
    write_json(output / "factorial_effects.json", factorial)
    write_json(output / "routing_interventions.json", {**mechanisms, **interventions})
    write_json(output / "expert_specialization.json", {
        "schema": "routed_moe_pilot_expert_specialization_v1",
        "anonymous_experts": True,
        "mean_expert_weights": mechanisms["mean_expert_weights"],
        "mean_pairwise_expert_action_l2": mechanisms["mean_pairwise_expert_action_l2"],
        "collapse_detected": not mechanisms["no_expert_load_collapse"],
        "scope": "two validation tasks and one seed; not a semantic specialization claim",
    })
    write_json(output / "capacity_compute_profile.json", {
        "schema": "routed_moe_pilot_capacity_compute_profile_v1",
        "methods": profiles,
        "capacity_matched_dense_controls_completed": False,
        "reason": "deferred to formal mechanism validation; pilot records the unmatched resource gap",
        "hardware": {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                     "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    })
    write_json(output / "router_input_ablations.json", {
        "schema": "routed_moe_pilot_router_input_ablations_v1",
        "implemented_modes": ["static", "posterior_mean", "static_posterior_mean", "static_posterior_mean_logvar"],
        "trained_in_pilot": {"moe_sac": "static", "pearl_moe": "static_posterior_mean_logvar"},
        "causal_input_ablation_completed": False,
        "reason": "the two pilot modes differ in posterior training as well as router input; formal matched ablations are still required",
    })
    test = subprocess.run([sys.executable, "-m", "pytest", "pearl_learning/tests", "-q"], capture_output=True, text=True)
    test_report = {"command": "python -m pytest pearl_learning/tests -q", "returncode": test.returncode,
                   "stdout": test.stdout.strip(), "stderr": test.stderr.strip(), "status": "pass" if test.returncode == 0 else "fail"}
    write_json(output / "test_report.json", test_report)
    checks = {
        "all_four_methods_trained_and_evaluated": set(method_results) == set(METHODS),
        "all_methods_have_gradient_updates": all(
            int(profile["gradient_updates"]) > 0 for profile in profiles.values()
        ),
        "all_metrics_finite": _finite(method_results) and _finite(intervention_results),
        "parameter_invariance": all(result["parameter_hash_before"] == result["parameter_hash_after"] for result in method_results.values())
        and all(result["parameter_hash_before"] == result["parameter_hash_after"] for result in intervention_results.values()),
        "matched_support_across_interventions": mechanisms["support_signatures_identical"],
        "posterior_dependent_route_change": mechanisms["posterior_dependent_route_change"],
        "nonzero_expert_action_difference": mechanisms["nonzero_expert_action_difference"],
        "no_expert_collapse": mechanisms["no_expert_load_collapse"],
        "intervention_contracts": mechanisms["intervention_contracts_hold"],
        "automated_tests": test.returncode == 0,
    }
    frozen = {
        "schema": "routed_moe_pilot_frozen_selection_v1", "taskbook_hash": taskbook_hash,
        "train_task_ids": [task.task_id for task in train_tasks],
        "validation_task_ids": [task.task_id for task in validation_tasks],
        "test_split_used": False, "training_seed": seed, "environment_steps_per_method": max_steps,
        "query_cases_per_task": int(pilot["query_cases_per_task"]), "shots": base["evaluation"]["shots"],
        "method_config_hashes": {method: content_hash(_method_config(base, method)) for method in METHODS},
    }
    write_json(output / "frozen_validation_selection.json", frozen)
    write_json(output / "statistical_summary.json", {
        "schema": "routed_moe_pilot_statistical_summary_v1", "seed_count": 1,
        "validation_task_count": len(validation_tasks), "confidence_intervals": None,
        "formal_inference_allowed": False, "factorial_effects": factorial,
    })
    compact = {"methods": method_results, "interventions": intervention_results}
    write_json(output / "compact_results.json", compact)
    primary = factorial["by_k"]["4"]["valid_critical_strict_rate"]
    manifest = {
        "schema": "routed_moe_pilot_manifest_v1", "experiment": "posterior_routed_moe_mechanism", "status": "PILOT",
        "pilot_passed": all(checks.values()), "checks": checks, "scope": pilot["scope"],
        "performance_validation_passed": False,
        "performance_signal": {
            "primary_metric": "valid_critical_strict_rate",
            "primary_k": 4,
            "pearl_moe_minus_pearl": primary["delta_moe_with_meta"],
            "pearl_moe_minus_moe_sac": (
                primary["method_means"]["pearl_moe"] - primary["method_means"]["moe_sac"]
            ),
            "factorial_interaction": primary["delta_interaction"],
            "full_minus_frozen_router": interventions["effects"]["frozen_router"]["4"]["valid_critical_strict_rate"],
            "interpretation": "no positive PEARL-MoE performance or causal routing signal in this pilot",
        },
        "training_seed": seed, "environment_steps_per_method": max_steps,
        "train_task_count": len(train_tasks), "validation_task_count": len(validation_tasks),
        "test_split_used": False, "checkpoints": checkpoints,
        "formal_mechanism_status": "INCOMPLETE", "allows_constrained_mining": False,
        "formal_gaps": ["3-5 training seeds", "24+ train and 8-12 validation tasks", "capacity-matched dense controls",
                        "matched router-input ablations", "random routing and router swap", "hierarchical interval estimates", "frozen independent meta-test"],
        "artifact_hashes": {name: content_hash(json.loads((output / name).read_text(encoding="utf-8"))) for name in (
            "frozen_validation_selection.json", "factorial_effects.json", "capacity_compute_profile.json",
            "router_input_ablations.json", "routing_interventions.json", "expert_specialization.json",
            "statistical_summary.json", "compact_results.json", "test_report.json")},
    }
    write_json(output / "manifest.json", manifest)
    if not manifest["pilot_passed"]:
        raise RuntimeError(f"Routed-MoE pilot checks failed: {checks}")
    print(f"Posterior-routed MoE pilot passed: {output}")


if __name__ == "__main__":
    main()
