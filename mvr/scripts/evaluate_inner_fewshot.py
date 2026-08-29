"""Evaluate paired Inner few-shot adaptation without invoking Outer search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..evaluation.fewshot_inner import (
    AdaptationQualityProtocol,
    BudgetEfficiencyProtocol,
    summarize_outcomes,
)
from ..failure.criteria import FailureCriteria
from ..scenario.taskbook import load_taskbook
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.headroom_casebook import HeadroomCasebook
from ..training.pipeline import (
    assert_taskbook_compatible,
    build_model,
    checkpoint_config_hash,
    load_config,
)
from ..training.trainers import build_online


def _query_provider(sampler: Callable[..., Any], support_count: int, max_support: int):
    def provide(task, episode_index, candidates, space):
        source = episode_index if episode_index < support_count else max_support + episode_index - support_count
        return sampler(task, source, candidates, space)
    return provide


def _query_seed(task, episode_index: int, support_count: int) -> int:
    return task.geometry_seed + episode_index if episode_index < support_count else task.geometry_seed + 10_000 + episode_index - support_count


def _evaluate_task(model, task, criteria, sampler, shots: int, queries: int, max_support: int) -> dict[str, Any]:
    online = build_online(model, task, 240, criteria)
    query_provider = _query_provider(sampler, shots, max_support)
    adapted = online.run(
        task, shots + queries, deterministic=True, posterior_support_limit=shots,
        scene_action_provider=query_provider,
        episode_seed_provider=lambda current, index: _query_seed(current, index, shots),
    ).episodes[shots:]

    fixed_queries = _query_provider(sampler, 0, max_support)
    fixed_seed = lambda current, index: current.geometry_seed + 10_000 + index
    zero = online.run(
        task, queries, deterministic=True, posterior_support_limit=0,
        scene_action_provider=fixed_queries, inner_action_provider=lambda _: np.zeros(2, dtype=np.float32),
        episode_seed_provider=fixed_seed,
    ).episodes
    prior = online.run(
        task, queries, deterministic=True, posterior_support_limit=0,
        scene_action_provider=fixed_queries, episode_seed_provider=fixed_seed,
    ).episodes
    rng = np.random.default_rng(task.geometry_seed)
    random = online.run(
        task, queries, deterministic=True, posterior_support_limit=0,
        scene_action_provider=fixed_queries,
        inner_action_provider=lambda _: rng.uniform(-0.75, 0.75, 2).astype(np.float32),
        episode_seed_provider=fixed_seed,
    ).episodes
    ablations = {}
    for name, use_scene, use_latent in (
        ("state_only", False, False), ("state_h", True, False),
        ("state_z", False, True), ("state_h_z", True, True),
    ):
        rows = online.run(
            task, shots + queries, deterministic=True, posterior_support_limit=shots,
            scene_action_provider=query_provider,
            episode_seed_provider=lambda current, index: _query_seed(current, index, shots),
            use_scene_context=use_scene, use_latent_context=use_latent,
        ).episodes[shots:]
        ablations[name] = summarize_outcomes([episode.outcome for episode in rows])
    return {
        "task_id": task.task_id,
        "support_shots": shots,
        "queries": queries,
        "base": summarize_outcomes([episode.outcome for episode in zero]),
        "random_residual": summarize_outcomes([episode.outcome for episode in random]),
        "shared_prior": summarize_outcomes([episode.outcome for episode in prior]),
        "adapted_h_z": summarize_outcomes([episode.outcome for episode in adapted]),
        "ablations": ablations,
    }


def run(config_path: str, checkpoint_path: str, casebook_path: str, protocol_name: str) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config)
    )
    assert_taskbook_compatible(checkpoint, taskbook_path)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    casebook = HeadroomCasebook.load(casebook_path)
    tasks = [task for task in load_taskbook(taskbook_path) if task.sut_split == "test" and task.logical_split == "test" and task.geometry_split == "train"]
    if not tasks:
        raise ValueError("few-shot test requires held-out SUT/domain tasks on retained topology")
    evaluation = dict(config["evaluation"])
    support_shots = tuple(int(value) for value in evaluation["support_shots"])
    max_support = max(support_shots)
    protocol = (
        AdaptationQualityProtocol(int(evaluation["query_cases"]), support_shots)
        if protocol_name == "adaptation_quality"
        else BudgetEfficiencyProtocol(int(evaluation["total_episode_budget"]), support_shots)
    )
    maximum_queries = max(
        protocol.query_cases if isinstance(protocol, AdaptationQualityProtocol)
        else protocol.query_cases(shots)
        for shots in protocol.support_shots
    )
    sampler = casebook.sampler(tasks, max_support + maximum_queries)
    reports = []
    for shots in protocol.support_shots:
        queries = protocol.query_cases if isinstance(protocol, AdaptationQualityProtocol) else protocol.query_cases(shots)
        reports.extend(_evaluate_task(
            model, task, FailureCriteria.from_config(config["failure"]), sampler,
            shots, queries, max_support,
        ) for task in tasks)
    return {
        "protocol": protocol_name,
        "fixed_query_cases": getattr(protocol, "query_cases", None) if protocol_name == "adaptation_quality" else None,
        "total_budget": getattr(protocol, "total_episode_budget", None),
        "reports": reports,
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
