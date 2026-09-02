"""Evaluate whether a Cut-in Inner checkpoint has learned an executable policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..experiments.cutin_inner import expand_cutin_training_domains
from ..failure.criteria import FailureCriteria
from ..scenario.parameter_space import NormalizedScenarioAction
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import (
    assert_taskbook_compatible,
    build_model,
    checkpoint_config_hash,
    load_config,
    selected_tasks,
)
from ..training.trainers import build_online


def _training_tasks(config: dict[str, Any], taskbook: Path) -> list[Any]:
    cutin = config["cutin_inner"]
    tasks = selected_tasks(config, taskbook, "train", "train", "train")
    tasks = [
        task for task in tasks
        if task.sut_ref in set(cutin["training_sut_refs"])
        and task.geometry_id in set(cutin["training_geometry_ids"])
    ]
    return expand_cutin_training_domains(tasks, cutin["training_logical_domains"])


def _centre_action(task: Any) -> NormalizedScenarioAction:
    return NormalizedScenarioAction(
        0,
        tuple(
            0.5 * (float(lower) + float(upper))
            for lower, upper in task.logical_domain_bounds.values()
        ),
    )


def run(config_path: str, checkpoint_path: str) -> dict[str, Any]:
    config, taskbook, device = load_config(config_path)
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config),
    )
    assert_taskbook_compatible(checkpoint, taskbook)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    criteria = FailureCriteria.from_config(config["failure"])
    rows = []
    for task in _training_tasks(config, taskbook):
        action = _centre_action(task)
        for seed in (11, 22, 33):
            result = build_online(
                model, task, int(config["training"]["step_budget"]), criteria,
            ).run(
                task,
                1,
                deterministic=True,
                posterior_support_limit=0,
                scene_action_provider=lambda *_args, value=action: value,
                episode_seed_provider=lambda current, _index, value=seed: (
                    current.geometry_seed + 10_000 * value
                ),
            )
            episode = result.episodes[0]
            transitions = episode.rollout.transitions
            rows.append({
                "task_id": task.task_id,
                "logical_domain_id": task.logical_domain_id,
                "sut_ref": task.sut_ref,
                "geometry_id": task.geometry_id,
                "seed": seed,
                "valid": bool(episode.outcome["is_valid_episode"]),
                "event": bool(
                    episode.outcome["valid_target_collision"]
                    or episode.outcome["valid_critical_near_miss"]
                ),
                "cutin_completed": any(
                    bool(row["info"].get("semantic_maneuver_completed", False))
                    for row in transitions
                ),
                "minimum_target_lateral_m": min(
                    abs(float(row["info"].get("traffic_cutin_lateral_m", np.inf)))
                    for row in transitions
                ),
                "min_ttc": float(episode.outcome["min_ttc"]),
                "min_distance": float(episode.outcome["min_distance"]),
                "termination_reason": episode.outcome["termination_reason"],
                "traffic_violation_counts": dict(
                    episode.outcome.get("traffic_telemetry", {}).get(
                        "violation_counts", {}
                    )
                ),
            })

    for row in rows:
        # A valid event terminates the simulator by contract.  Requiring a
        # later reference endpoint after that terminal interaction would
        # reject the intended successful Cut-in behavior without changing
        # any failure or semantic criterion.
        row["cutin_path_or_event"] = bool(row["cutin_completed"] or row["event"])

    def rate(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    gate = {
        "minimum_valid_rate": 0.75,
        "minimum_cutin_path_or_event_rate": 0.75,
        "minimum_valid_event_rate": 1.0 / len(rows),
    }
    summary = {
        "valid_rate": rate("valid"),
        "cutin_path_or_event_rate": rate("cutin_path_or_event"),
        "valid_event_rate": rate("event"),
    }
    passed = all(summary[name.removeprefix("minimum_")] >= threshold for name, threshold in gate.items())
    return {
        "scope": {
            "functional_scenario": "cutin",
            "split": "train_only_policy_gate",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "checkpoint_stage": checkpoint.stage,
        "gate": gate,
        "summary": summary,
        "passed": passed,
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.config, args.checkpoint)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
