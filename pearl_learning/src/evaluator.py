"""No-gradient, episode-balanced few-shot evaluation."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import torch

from .collector import Rollout, collect_episode
from .io import content_hash
from .metrics import summarize
from .task_env import LogicalMergeEnv


def _sample_episode_context(rollouts: list[Rollout], total_size: int, per_episode: int, rng: np.random.Generator) -> list[list[Any]]:
    if not rollouts:
        raise ValueError("cannot infer a posterior without a support episode")
    count = min(len(rollouts), max(1, int(total_size) // int(per_episode)))
    indexes = rng.choice(len(rollouts), size=count, replace=False)
    groups = []
    for index in np.asarray(indexes).reshape(-1):
        rows = rollouts[int(index)].transitions
        chosen = rng.choice(len(rows), size=int(per_episode), replace=len(rows) < int(per_episode))
        groups.append([rows[int(item)] for item in np.asarray(chosen).reshape(-1)])
    return groups


def evaluate_fewshot(agent: Any, config: Mapping[str, Any], tasks: list[Any], casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]],
                     split: str, query_cases_per_task: int | None = None,
                     provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    device = agent.device
    shots = list(config["evaluation"]["shots"])
    query_limit = query_cases_per_task or int(config["evaluation"]["query_cases_per_task"])
    before = agent.parameter_hash(); all_results: dict[str, Any] = {}
    support_key, query_key = ("validation_support", "validation_query") if split == "meta_validation" else ("test_support", "test_query")
    base_seed = int(config["evaluation"]["context_sampling_seed"])
    output_provenance = dict(provenance or {})
    for task in tasks:
        book = casebooks[task.task_id]
        env = LogicalMergeEnv(task, config, book[query_key])
        results: dict[str, Any] = {}
        support_rollouts: list[Rollout] = []
        support_episode_lengths: list[int] = []
        try:
            mu, log_var = agent.prior()
            for shot in range(max(shots) + 1):
                if shot in shots:
                    queries = [collect_episode(env, task, case, agent, mu, "deterministic_query", device, episode_id=f"{task.task_id}:query:{shot}:{index}", posterior_version=shot) for index, case in enumerate(book[query_key][:query_limit])]
                    records = [rollout.record for rollout in queries]
                    posterior_mean = mu.detach().cpu().tolist()
                    results[str(shot)] = {
                        "summary": summarize(records), "records": records, "posterior_mean": posterior_mean,
                        "posterior_variance": torch.exp(log_var).detach().cpu().tolist(), "context_episode_count": len(support_rollouts),
                        "support_episode_lengths": list(support_episode_lengths),
                        "support_environment_steps": int(sum(support_episode_lengths)),
                    }
                if shot == max(shots):
                    break
                case = book[support_key][shot]
                rollout = collect_episode(env, task, case, agent, agent.sample_latent(mu, log_var), "prior_support" if shot == 0 else "posterior_rollout", device, episode_id=f"{task.task_id}:support:{shot}", posterior_version=shot)
                support_rollouts.append(rollout)
                support_episode_lengths.append(len(rollout.transitions))
                rng = np.random.default_rng(int(content_hash({"seed": base_seed, "task": task.task_id, "shot": shot})[:16], 16))
                context = _sample_episode_context(support_rollouts, int(config["pearl"]["context_sample_size_eval"]), int(config["pearl"]["context_transitions_per_episode"]), rng)
                mu, log_var = agent.infer_posterior([context])
        finally:
            env.close()
        all_results[task.task_id] = results
    after = agent.parameter_hash()
    if before != after:
        raise RuntimeError("meta-test changed model parameters, target critics, or alpha")
    return {"split": split, "parameter_hash_before": before, "parameter_hash_after": after, "no_gradient_adaptation": True, "no_topology_ablation": bool(config.get("ablation", {}).get("no_topology", False)), "context_protocol": {"sample_size": int(config["pearl"]["context_sample_size_eval"]), "transitions_per_episode": int(config["pearl"]["context_transitions_per_episode"]), "seed": base_seed}, "provenance": output_provenance, "tasks": all_results}


def compact_fewshot_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep reportable few-shot metrics while dropping per-episode records."""
    tasks: dict[str, Any] = {}
    for task_id, shots in result["tasks"].items():
        tasks[task_id] = {
            shot: {
                "summary": value["summary"],
                "support_environment_steps": value["support_environment_steps"],
            }
            for shot, value in shots.items()
        }
    return {
        key: result[key]
        for key in ("split", "parameter_hash_before", "parameter_hash_after", "no_gradient_adaptation", "no_topology_ablation", "context_protocol", "provenance")
    } | {"tasks": tasks}


def validation_score(result: Mapping[str, Any], shot: int = 5) -> tuple[float, float, float, float, float]:
    """Few-shot-sensitive lexicographic checkpoint score."""
    tasks = list(result["tasks"].values())
    if not tasks:
        raise ValueError("validation result has no task summaries")
    strict = lambda task, key: float(task[str(key)]["summary"]["valid_critical_strict_rate"])
    target_shot = str(shot)
    strict_at = float(np.mean([float(task[target_shot]["summary"]["valid_critical_strict_rate"]) for task in tasks]))
    aucs = []
    for task in tasks:
        xs = np.asarray(sorted(int(key) for key in task), dtype=float)
        ys = np.asarray([strict(task, int(key)) for key in xs], dtype=float)
        aucs.append(float(np.trapz(ys, xs) / max(float(xs[-1] - xs[0]), 1.0)))
    gain = float(np.mean([strict(task, shot) - strict(task, 0) for task in tasks]))
    invalid = float(np.mean([float(task[target_shot]["summary"]["invalid_rate"]) for task in tasks]))
    ttc = float(np.mean([float(task[target_shot]["summary"]["median_min_ttc"]) for task in tasks]))
    return strict_at, float(np.mean(aucs)), gain, -invalid, -ttc
