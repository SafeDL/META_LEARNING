"""Prove train-domain Cut-in risk reachability before learned-policy training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..experiments.cutin_inner import expand_cutin_training_domains
from ..failure.criteria import FailureCriteria
from ..state import INNER_STATE_FIELDS
from ..training.pipeline import build_model, load_config, selected_tasks
from ..training.stage1_sampling import PretrainSceneSampler
from ..training.trainers import build_online


PROGRESS_INDEX = INNER_STATE_FIELDS.index("maneuver_reference_progress")


def _policy(
    length: float, early: float, late: float, brake_progress: float
) -> Callable[[np.ndarray], np.ndarray]:
    def action(state: np.ndarray) -> np.ndarray:
        longitudinal = -1.0 if float(state[PROGRESS_INDEX]) >= brake_progress else 0.15
        return np.asarray((length, early, late, longitudinal), dtype=np.float32)

    return action


SCRIPTED_POLICIES = {
    "quintic_then_brake": _policy(0.0, 0.0, 0.0, 0.70),
    "short_early_then_brake": _policy(-1.0, 1.0, -1.0, 0.62),
    "long_late_then_brake": _policy(1.0, -1.0, 1.0, 0.78),
}


def _representative_tasks(config: dict[str, Any], taskbook: Path) -> list[Any]:
    settings = config["cutin_inner"]
    tasks = selected_tasks(config, taskbook, "train", "train", "train")
    tasks = [
        task for task in tasks
        if task.sut_ref in set(settings["training_sut_refs"])
        and task.geometry_id in set(settings["training_geometry_ids"])
    ]
    expanded = expand_cutin_training_domains(
        tasks, settings["training_logical_domains"]
    )
    representatives = {}
    for task in expanded:
        representatives.setdefault(task.logical_domain_id, task)
    return list(representatives.values())


def run(config_path: str) -> dict[str, Any]:
    config, taskbook, device = load_config(config_path)
    model = build_model(config, device)
    model.eval()
    tasks = _representative_tasks(config, taskbook)
    criteria = FailureCriteria.from_config(config["failure"])
    cases_per_policy = 6
    rows = []
    for task in tasks:
        sampler = PretrainSceneSampler(
            (task,), cases_per_policy, int(config["seed"])
        )
        online = build_online(
            model,
            task,
            int(config["training"]["step_budget"]),
            criteria,
        )
        for policy_name, policy in SCRIPTED_POLICIES.items():
            result = online.run(
                task,
                cases_per_policy,
                deterministic=True,
                posterior_support_limit=0,
                scene_action_provider=sampler,
                inner_action_provider=policy,
                episode_seed_provider=lambda current, index: int(
                    current.geometry_seed + 50_000 + index
                ),
            )
            for episode in result.episodes:
                rows.append({
                    "task_id": task.task_id,
                    "logical_domain_id": task.logical_domain_id,
                    "policy": policy_name,
                    "candidate_id": episode.concrete_scenario.candidate_id,
                    "formal_valid": bool(episode.outcome["is_valid_episode"]),
                    "valid_target_collision": bool(
                        episode.outcome["valid_target_collision"]
                    ),
                    "valid_critical_near_miss": bool(
                        episode.outcome["valid_critical_near_miss"]
                    ),
                    "termination_reason": episode.outcome["termination_reason"],
                })
    domain_reachable = {}
    for domain in {task.logical_domain_id for task in tasks}:
        domain_rows = [row for row in rows if row["logical_domain_id"] == domain]
        domain_reachable[domain] = any(
            row["valid_target_collision"] or row["valid_critical_near_miss"]
            for row in domain_rows
        )
    return {
        "scope": {
            "functional_scenario": "cutin",
            "split": "train_only",
            "outer_run": False,
            "test_split_accessed": False,
        },
        "scripted_policies": list(SCRIPTED_POLICIES),
        "rows": rows,
        "domain_reachable": domain_reachable,
        "passed": all(domain_reachable.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
