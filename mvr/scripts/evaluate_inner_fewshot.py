"""Evaluate paired Inner few-shot adaptation on calibration-casebook queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..evaluation.fewshot_inner import (
    AdaptationQualityProtocol,
    BudgetEfficiencyProtocol,
    paired_bootstrap,
    paired_policy_deltas,
    summarize_outcomes,
    valid_critical_score,
)
from ..failure.criteria import FailureCriteria
from ..scenario.taskbook import load_taskbook
from ..training.calibration_casebook import CalibrationCasebook
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import (
    assert_taskbook_compatible,
    build_model,
    checkpoint_config_hash,
    load_config,
)
from ..training.trainers import build_online


def _case_provider(sampler: Callable[..., Any], support_count: int, max_support: int):
    def provide(task, episode_index, candidates, space):
        source = episode_index if episode_index < support_count else max_support + episode_index - support_count
        return sampler(task, source, candidates, space)
    return provide


def _episode_seed(task, episode_index: int, support_count: int, max_support: int, seed: int) -> int:
    source = episode_index if episode_index < support_count else max_support + episode_index - support_count
    return int(task.geometry_seed + 100_000 * int(seed) + source)


def _query_records(
    episodes: Sequence[Any], *, task: Any, casebook: CalibrationCasebook,
    support_shots: int, seed: int, policy: str, max_support: int,
) -> list[dict[str, Any]]:
    records = []
    for query_case_id, episode in enumerate(episodes):
        source = max_support + query_case_id
        calibration = casebook.case_for(task.task_id, source)
        records.append({
            "task_id": task.task_id,
            "functional_scenario": task.functional_scenario,
            "geometry_id": task.geometry_id,
            "logical_domain_id": task.logical_domain_id,
            "support_shots": int(support_shots),
            "seed": int(seed),
            "query_case_id": int(query_case_id),
            "policy": policy,
            "score": valid_critical_score(episode.outcome),
            "invalid": not bool(episode.outcome.get("is_valid_episode", False)),
            "z": [float(value) for value in episode.latent_before.squeeze(0).tolist()],
            "concrete_provenance": {
                "casebook": calibration.provenance(),
                "source_case_index": source,
                "concrete_scenario": episode.concrete_scenario.to_dict(),
            },
        })
    return records


def _evaluate_task(
    model: Any, task: Any, criteria: FailureCriteria, sampler: Callable[..., Any],
    casebook: CalibrationCasebook, shots: int, queries: int, max_support: int, seed: int,
) -> list[dict[str, Any]]:
    online = build_online(model, task, 240, criteria)
    adapted_provider = _case_provider(sampler, shots, max_support)
    seed_provider = lambda current, index: _episode_seed(current, index, shots, max_support, seed)
    adapted = online.run(
        task, shots + queries, deterministic=True, posterior_support_limit=shots,
        scene_action_provider=adapted_provider, episode_seed_provider=seed_provider,
    ).episodes[shots:]
    fixed_provider = _case_provider(sampler, 0, max_support)
    fixed_seed = lambda current, index: _episode_seed(current, index, 0, max_support, seed)
    zero = online.run(
        task, queries, deterministic=True, posterior_support_limit=0,
        scene_action_provider=fixed_provider,
        inner_action_provider=lambda _: np.zeros(2, dtype=np.float32),
        episode_seed_provider=fixed_seed,
    ).episodes
    prior = online.run(
        task, queries, deterministic=True, posterior_support_limit=0,
        scene_action_provider=fixed_provider, episode_seed_provider=fixed_seed,
    ).episodes
    rng = np.random.default_rng(task.geometry_seed + seed)
    random = online.run(
        task, queries, deterministic=True, posterior_support_limit=0,
        scene_action_provider=fixed_provider,
        inner_action_provider=lambda _: rng.uniform(-0.75, 0.75, 2).astype(np.float32),
        episode_seed_provider=fixed_seed,
    ).episodes
    records = []
    for policy, episodes in (
        ("base_nominal", zero), ("random_residual", random),
        ("shared_prior", prior), ("adapted_h_z", adapted),
    ):
        records.extend(_query_records(
            episodes, task=task, casebook=casebook, support_shots=shots,
            seed=seed, policy=policy, max_support=max_support,
        ))
    for name, use_scene, use_latent in (
        ("state_only", False, False), ("state_h", True, False),
        ("state_z", False, True), ("state_h_z", True, True),
    ):
        episodes = online.run(
            task, shots + queries, deterministic=True, posterior_support_limit=shots,
            scene_action_provider=adapted_provider, episode_seed_provider=seed_provider,
            use_scene_context=use_scene, use_latent_context=use_latent,
        ).episodes[shots:]
        records.extend(_query_records(
            episodes, task=task, casebook=casebook, support_shots=shots,
            seed=seed, policy=name, max_support=max_support,
        ))
    return records


def _summary(records: Sequence[dict[str, Any]], support_shots: Sequence[int]) -> dict[str, Any]:
    summary = {}
    for shots in support_shots:
        rows = [row for row in records if row["support_shots"] == shots]
        summary[str(shots)] = {
            policy: summarize_outcomes([
                {"is_valid_episode": not row["invalid"], "valid_target_collision": row["score"] == 1.0,
                 "valid_critical_near_miss": row["score"] == 0.5}
                for row in rows if row["policy"] == policy
            ])
            for policy in sorted({row["policy"] for row in rows})
        }
    return summary


def run(config_path: str, checkpoint_path: str, casebook_path: str, protocol_name: str) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config)
    )
    assert_taskbook_compatible(checkpoint, taskbook_path)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    casebook = CalibrationCasebook.load(casebook_path)
    if bool(casebook.metadata.get("test_sut_base_safe_claim", True)):
        raise ValueError("casebook must explicitly reject test-SUT Base-safe inheritance")
    tasks = [
        task for task in load_taskbook(taskbook_path)
        if task.sut_split == "test" and task.logical_split == "test" and task.geometry_split == "train"
    ]
    if not tasks:
        raise ValueError("few-shot test requires held-out SUT/domain tasks on retained topology")
    evaluation = dict(config["evaluation"])
    support_shots = tuple(int(value) for value in evaluation["support_shots"])
    seeds = tuple(int(value) for value in evaluation["seeds"])
    max_support = max(support_shots)
    protocol = (
        AdaptationQualityProtocol(int(evaluation["query_cases"]), support_shots)
        if protocol_name == "adaptation_quality"
        else BudgetEfficiencyProtocol(int(evaluation["total_episode_budget"]), support_shots)
    )
    maximum_queries = max(
        protocol.query_cases if isinstance(protocol, AdaptationQualityProtocol)
        else protocol.query_cases(shots)
        for shots in support_shots
    )
    sampler = casebook.sampler(tasks, max_support + maximum_queries)
    records = []
    for shots in support_shots:
        queries = protocol.query_cases if isinstance(protocol, AdaptationQualityProtocol) else protocol.query_cases(shots)
        for seed in seeds:
            for task in tasks:
                records.extend(_evaluate_task(
                    model, task, FailureCriteria.from_config(config["failure"]), sampler,
                    casebook, shots, queries, max_support, seed,
                ))
    comparisons = {}
    bootstrap_samples = int(evaluation.get("paired_bootstrap_samples", 10_000))
    bootstrap_seed = int(evaluation.get("paired_bootstrap_seed", 11))
    for shots in support_shots:
        rows = [row for row in records if row["support_shots"] == shots]
        comparisons[str(shots)] = {
            "adapted_minus_shared_prior": paired_bootstrap(
                paired_policy_deltas(rows, "adapted_h_z", "shared_prior"),
                samples=bootstrap_samples, seed=bootstrap_seed,
            ),
            "adapted_minus_base_nominal": paired_bootstrap(
                paired_policy_deltas(rows, "adapted_h_z", "base_nominal"),
                samples=bootstrap_samples, seed=bootstrap_seed,
            ),
        }
    return {
        "protocol": protocol_name,
        "fixed_query_cases": getattr(protocol, "query_cases", None) if protocol_name == "adaptation_quality" else None,
        "total_budget": getattr(protocol, "total_episode_budget", None),
        "simulator_seeds": list(seeds),
        "casebook_metadata": dict(casebook.metadata),
        "query_records": records,
        "policy_summary": _summary(records, support_shots),
        "paired_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/mvr.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--casebook", required=True)
    parser.add_argument("--protocol", choices=("adaptation_quality", "budget_efficiency"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.config, args.checkpoint, args.casebook, args.protocol)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
