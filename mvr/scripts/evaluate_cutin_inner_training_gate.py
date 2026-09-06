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


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summarize_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the Stage 1 shared-policy gate to each Logical Domain."""
    if not rows:
        raise ValueError("training gate requires at least one evaluation record")
    gate = {
        "minimum_valid_rate_per_domain": 0.75,
        "minimum_valid_event_count_per_domain": 1,
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["logical_domain_id"]), []).append(row)

    domains = {}
    for domain, domain_rows in sorted(grouped.items()):
        valid_count = sum(bool(row["valid"]) for row in domain_rows)
        event_count = sum(bool(row["event"]) for row in domain_rows)
        summary = {
            "cases": len(domain_rows),
            "valid_count": valid_count,
            "event_count": event_count,
            "valid_rate": _rate(domain_rows, "valid"),
            "event_rate": _rate(domain_rows, "event"),
            "cutin_path_or_event_rate": _rate(domain_rows, "cutin_path_or_event"),
        }
        summary["passed"] = bool(
            summary["valid_rate"] >= gate["minimum_valid_rate_per_domain"]
            and event_count >= gate["minimum_valid_event_count_per_domain"]
        )
        domains[domain] = summary

    passed_domain_count = sum(summary["passed"] for summary in domains.values())
    total_domain_count = len(domains)
    passed = passed_domain_count == total_domain_count
    return {
        "gate": gate,
        "summary": {
            "valid_rate": _rate(rows, "valid"),
            "cutin_path_or_event_rate": _rate(rows, "cutin_path_or_event"),
            "valid_event_rate": _rate(rows, "event"),
        },
        "domains": domains,
        "passed_domain_count": passed_domain_count,
        "total_domain_count": total_domain_count,
        "status": "passed" if passed else (
            "partially_passed" if passed_domain_count else "failed"
        ),
        "passed": passed,
    }


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
                "raw_near_miss_candidate_steps": sum(
                    bool(row["info"].get("raw_near_miss_candidate", False))
                    for row in transitions
                ),
                "valid_event_capture_steps": sum(
                    bool(row["info"].get("event_just_captured", False))
                    and (
                        bool(row["info"].get("valid_target_collision", False))
                        or bool(row["info"].get(
                            "valid_critical_near_miss", False
                        ))
                    )
                    for row in transitions
                ),
            })

    for row in rows:
        # A valid event terminates the simulator by contract.  Requiring a
        # later reference endpoint after that terminal interaction would
        # reject the intended successful Cut-in behavior without changing
        # any failure or semantic criterion.
        row["cutin_path_or_event"] = bool(row["cutin_completed"] or row["event"])

    report = summarize_records(rows)
    return {
        "scope": {
            "functional_scenario": "cutin",
            "split": "train_only_policy_gate",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "checkpoint_stage": checkpoint.stage,
        **report,
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
