"""Measure validation failure rates of the trained Cut-in Inner SAC policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..experiments.cutin_inner import select_cutin_validation_tasks
from ..failure.criteria import FailureCriteria
from ..scenario.taskbook import load_taskbook
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import (
    assert_taskbook_compatible,
    build_model,
    checkpoint_config_hash,
    load_config,
)
from ..training.stage1_sampling import PretrainSceneSampler
from ..training.stages import TrainingStage
from ..training.trainers import build_online


def _seed(task: Any, index: int, support_shots: int, max_support: int, seed: int) -> int:
    source = index if index < support_shots else max_support + index - support_shots
    return int(task.geometry_seed + 100_000 * int(seed) + source)


def _records(episodes: Sequence[Any], task: Any, seed: int, policy: str) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task.task_id,
            "sut_ref": task.sut_ref,
            "geometry_id": task.geometry_id,
            "logical_domain_id": task.logical_domain_id,
            "seed": seed,
            "policy": policy,
            "valid": bool(episode.outcome["is_valid_episode"]),
            "failure": bool(episode.outcome["is_failure"]),
            "valid_target_collision": bool(episode.outcome["valid_target_collision"]),
            "valid_critical_near_miss": bool(episode.outcome["valid_critical_near_miss"]),
            "min_ttc": float(episode.outcome["min_ttc"]),
            "min_distance": float(episode.outcome["min_distance"]),
            "concrete_scenario": episode.concrete_scenario.to_dict(),
        }
        for episode in episodes
    ]


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    report = {}
    for policy in sorted({str(row["policy"]) for row in records}):
        rows = [row for row in records if row["policy"] == policy]
        valid = [row for row in rows if row["valid"]]
        report[policy] = {
            "episodes": float(len(rows)),
            "valid_rate": float(len(valid) / len(rows)),
            "failure_rate": float(sum(bool(row["failure"]) for row in rows) / len(rows)),
            "failure_rate_given_valid": float(
                sum(bool(row["failure"]) for row in valid) / max(len(valid), 1)
            ),
            "target_collision_rate": float(
                sum(bool(row["valid_target_collision"]) for row in rows) / len(rows)
            ),
            "near_miss_rate": float(
                sum(bool(row["valid_critical_near_miss"]) for row in rows) / len(rows)
            ),
        }
    return report


def run(config_path: str, checkpoint_path: str) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    cutin_inner = config.get("cutin_inner")
    if cutin_inner is None or bool(cutin_inner.get("allow_outer", True)):
        raise ValueError("failure-rate evaluation requires the no-Outer Cut-in configuration")
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config),
    )
    if checkpoint.stage != TrainingStage.CONTEXT_META.value:
        raise ValueError("failure-rate evaluation requires a context_meta checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook_path)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    tasks = select_cutin_validation_tasks(load_taskbook(taskbook_path))
    settings = config["evaluation"]["failure_rate"]
    queries = int(settings["query_cases"])
    support_shots = int(settings["support_shots"])
    criteria = FailureCriteria.from_config(config["failure"])
    rows: list[dict[str, Any]] = []
    for seed in (int(value) for value in settings["seeds"]):
        sampler = PretrainSceneSampler(tuple(tasks), support_shots + queries, seed)
        for task in tasks:
            online = build_online(model, task, int(config["training"]["step_budget"]), criteria)
            shared = online.run(
                task, queries, deterministic=True, posterior_support_limit=0,
                scene_action_provider=sampler,
                episode_seed_provider=lambda current, index, value=seed: _seed(
                    current, index, 0, support_shots, value,
                ),
            ).episodes
            adapted = online.run(
                task, support_shots + queries, deterministic=True,
                posterior_support_limit=support_shots, scene_action_provider=sampler,
                episode_seed_provider=lambda current, index, value=seed: _seed(
                    current, index, support_shots, support_shots, value,
                ),
            ).episodes[support_shots:]
            rows.extend(_records(shared, task, seed, "shared_prior"))
            rows.extend(_records(adapted, task, seed, "adapted_h_z"))
    return {
        "scope": {
            "functional_scenario": "cutin",
            "sut_split": "validation",
            "geometry_split": "train",
            "logical_split": "validation",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "support_shots": support_shots,
        "query_cases": queries,
        "seeds": list(settings["seeds"]),
        "summary": _summarize(rows),
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.config, args.checkpoint)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
