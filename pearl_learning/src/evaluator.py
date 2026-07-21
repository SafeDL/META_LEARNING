"""No-gradient, episode-balanced few-shot evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

from .collector import Rollout, collect_episode
from .io import content_hash
from .metrics import summarize
from .observation import OBSERVATION_SCHEMA
from .scenario_manifest import save_manifest
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


def _save_critical_scenarios(output_dir: Path, task: Any, shot: int, rollouts: list[Rollout], cases: list[Mapping[str, Any]], posterior_mean: list[float], top_k: int, provenance: Mapping[str, Any]) -> None:
    candidates = [rollout for rollout in rollouts if rollout.record["valid_critical_strict"]]
    cases_by_id = {case["case_id"]: case for case in cases}
    for rank, rollout in enumerate(sorted(candidates, key=lambda item: item.record["min_ttc"])[:top_k], 1):
        case_id = rollout.record["case_id"]
        manifest = {
            "task": task.to_dict(), "case": cases_by_id[case_id], "shot": shot, "posterior_mean": posterior_mean,
            "observation_schema": OBSERVATION_SCHEMA, "termination_reason": rollout.record["termination_reason"],
            "provenance": dict(provenance), "episode_id": rollout.episode_id,
        }
        save_manifest(output_dir / task.task_id / f"shot_{shot}" / "critical_scenarios" / f"rank_{rank:03d}", manifest, [row.action for row in rollout.transitions], rollout.record)


def evaluate_fewshot(agent: Any, config: Mapping[str, Any], tasks: list[Any], casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]],
                     split: str, query_cases_per_task: int | None = None, output_dir: Path | None = None,
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
        try:
            mu, log_var = agent.prior()
            for shot in range(max(shots) + 1):
                if shot in shots:
                    queries = [collect_episode(env, task, case, agent, mu, "deterministic_query", device, episode_id=f"{task.task_id}:query:{shot}:{index}", posterior_version=shot) for index, case in enumerate(book[query_key][:query_limit])]
                    records = [rollout.record for rollout in queries]
                    posterior_mean = mu.detach().cpu().tolist()
                    if output_dir is not None:
                        _save_critical_scenarios(output_dir, task, shot, queries, book[query_key][:query_limit], posterior_mean, int(config["evaluation"]["top_k_scenarios"]), output_provenance)
                    results[str(shot)] = {
                        "summary": summarize(records), "records": records, "posterior_mean": posterior_mean,
                        "posterior_variance": torch.exp(log_var).detach().cpu().tolist(), "context_episode_count": len(support_rollouts),
                    }
                if shot == max(shots):
                    break
                case = book[support_key][shot]
                rollout = collect_episode(env, task, case, agent, agent.sample_latent(mu, log_var), "prior_support" if shot == 0 else "posterior_rollout", device, episode_id=f"{task.task_id}:support:{shot}", posterior_version=shot)
                support_rollouts.append(rollout)
                rng = np.random.default_rng(int(content_hash({"seed": base_seed, "task": task.task_id, "shot": shot})[:16], 16))
                context = _sample_episode_context(support_rollouts, int(config["pearl"]["context_sample_size_eval"]), int(config["pearl"]["context_transitions_per_episode"]), rng)
                mu, log_var = agent.infer_posterior([context])
        finally:
            env.close()
        all_results[task.task_id] = results
    after = agent.parameter_hash()
    if before != after:
        raise RuntimeError("meta-test changed model parameters, target critics, or alpha")
    return {"split": split, "parameter_hash_before": before, "parameter_hash_after": after, "no_gradient_adaptation": True, "context_protocol": {"sample_size": int(config["pearl"]["context_sample_size_eval"]), "transitions_per_episode": int(config["pearl"]["context_transitions_per_episode"]), "seed": base_seed}, "provenance": output_provenance, "tasks": all_results}


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
