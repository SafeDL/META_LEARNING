"""No-gradient support/query few-shot evaluation and integrity audit."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import torch
from .casebook import build_casebook
from .collector import collect_episode
from .io import write_json
from .metrics import summarize
from .observation import OBSERVATION_SCHEMA
from .scenario_manifest import save_manifest
from .task_env import LogicalMergeEnv


def _save_critical_scenarios(output_dir: Path, task: Any, shot: int, rollouts: list[Any], cases: list[Mapping[str, Any]], posterior_mean: list[float], top_k: int) -> None:
    candidates = [rollout for rollout in rollouts if rollout.record["valid_critical_strict"]]
    cases_by_id = {case["case_id"]: case for case in cases}
    for rank, rollout in enumerate(sorted(candidates, key=lambda item: item.record["min_ttc"])[:top_k], 1):
        case_id = rollout.record["case_id"]
        manifest = {
            "task": task.to_dict(), "case": cases_by_id[case_id], "shot": shot,
            "posterior_mean": posterior_mean, "observation_schema": OBSERVATION_SCHEMA,
            "termination_reason": rollout.record["termination_reason"],
        }
        actions = [transition.action for transition in rollout.transitions]
        save_manifest(output_dir / task.task_id / f"shot_{shot}" / "critical_scenarios" / f"rank_{rank:03d}", manifest, actions, rollout.record)


def evaluate_fewshot(agent: Any, config: Mapping[str, Any], tasks: list[Any], split: str, query_cases_per_task: int | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    device = agent.device; shots = list(config["evaluation"]["shots"]); query_limit = query_cases_per_task or int(config["evaluation"]["query_cases_per_task"])
    before = agent.parameter_hash(); all_results: dict[str, Any] = {}
    for task in tasks:
        book = build_casebook(task, config); support_key, query_key = ("validation_support", "validation_query") if split in {"meta_validation", "meta_test_template"} else ("test_support", "test_query")
        env = LogicalMergeEnv(task, config, book[query_key]); task_results = {}
        try:
            mu, log_var = agent.prior(); context: list[Any] = []
            for shot in range(max(shots) + 1):
                if shot in shots:
                    z = mu  # posterior mean, deterministic query action
                    rollouts = [collect_episode(env, task, case, agent, z, "deterministic_query", device) for case in book[query_key][:query_limit]]
                    records = [rollout.record for rollout in rollouts]
                    posterior_mean = mu.detach().cpu().tolist()
                    if output_dir is not None:
                        _save_critical_scenarios(output_dir, task, shot, rollouts, book[query_key][:query_limit], posterior_mean, int(config["evaluation"]["top_k_scenarios"]))
                    task_results[str(shot)] = {"summary": summarize(records), "records": records, "posterior_mean": posterior_mean, "posterior_variance": torch.exp(log_var).detach().cpu().tolist()}
                if shot == max(shots): break
                rollout = collect_episode(env, task, book[support_key][shot], agent, agent.sample_latent(mu, log_var), "posterior_support", device)
                context.extend(rollout.transitions); mu, log_var = agent.infer_posterior([context])
        finally: env.close()
        all_results[task.task_id] = task_results
    after = agent.parameter_hash()
    if before != after: raise RuntimeError("meta-test changed network weights")
    return {"split": split, "parameter_hash_before": before, "parameter_hash_after": after, "no_gradient_adaptation": True, "tasks": all_results}


def validation_score(result: Mapping[str, Any], shot: int = 5) -> tuple[float, float, float, float]:
    """Stage-1-style lexicographic checkpoint score at one fixed shot count."""
    summaries = [task[str(shot)]["summary"] for task in result["tasks"].values()]
    if not summaries:
        raise ValueError("validation result has no task summaries")
    mean = lambda key: sum(float(summary[key]) for summary in summaries) / len(summaries)
    return (
        mean("target_collision_rate"),
        mean("valid_critical_strict_rate"),
        -mean("invalid_rate"),
        -mean("median_min_ttc"),
    )
