"""Measure whether fixed Inner residuals can reach valid critical behavior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..failure.criteria import FailureCriteria
from ..scenario.taskbook import load_taskbook
from ..training.pipeline import build_model, load_config
from ..training.stage1_sampling import PretrainSceneSampler
from ..training.trainers import build_online


CASES_PER_TASK = 4
SWEEP_ACTIONS = (
    ("base", (0.0, 0.0)),
    ("steering_negative", (-0.5, 0.0)),
    ("steering_positive", (0.5, 0.0)),
    ("acceleration_brake", (0.0, -0.75)),
    ("acceleration_press", (0.0, 0.75)),
)
OUTCOME_FIELDS = (
    "is_valid_episode",
    "is_failure",
    "valid_target_collision",
    "valid_critical_near_miss",
    "adversary_traffic_violation",
    "adversary_out_of_road",
    "sut_out_of_road",
    "wrong_route",
    "min_ttc",
    "min_distance",
    "max_closing_speed",
    "event_kind",
    "termination_reason",
)


def _outcome(outcome: Mapping[str, Any]) -> dict[str, Any]:
    return {field: outcome.get(field) for field in OUTCOME_FIELDS}


def _scenario(episode: Any) -> dict[str, Any]:
    scenario = episode.concrete_scenario
    return {
        "candidate_id": scenario.candidate_id,
        "option": scenario.option,
        "initial_state": dict(scenario.initial_state),
        "normalized_continuous": list(scenario.normalized_continuous),
        "episode_seed": scenario.episode_seed,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = {
        (str(row["task_id"]), int(row["case_index"])): row
        for row in rows
        if row["residual_name"] == "base"
    }
    paired = []
    for row in rows:
        if row["residual_name"] == "base":
            continue
        reference = base[(str(row["task_id"]), int(row["case_index"]))]
        paired.append((reference, row))

    def aggregate(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not group:
            return {
                "episodes": 0,
                "valid_rate": 0.0,
                "valid_critical_rate": 0.0,
                "valid_target_collision_rate": 0.0,
                "valid_near_miss_rate": 0.0,
                "median_min_ttc": None,
                "median_min_distance": None,
            }
        outcomes = [row["outcome"] for row in group]
        return {
            "episodes": len(group),
            "valid_rate": float(np.mean([bool(row["is_valid_episode"]) for row in outcomes])),
            "valid_critical_rate": float(np.mean([bool(row["is_failure"]) for row in outcomes])),
            "valid_target_collision_rate": float(
                np.mean([bool(row["valid_target_collision"]) for row in outcomes])
            ),
            "valid_near_miss_rate": float(
                np.mean([bool(row["valid_critical_near_miss"]) for row in outcomes])
            ),
            "median_min_ttc": float(np.median([float(row["min_ttc"]) for row in outcomes])),
            "median_min_distance": float(
                np.median([float(row["min_distance"]) for row in outcomes])
            ),
        }

    by_family = {}
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "any_valid_critical": any(bool(row["outcome"]["is_failure"]) for row in family_rows),
            "by_residual": {
                name: aggregate([
                    row for row in family_rows if row["residual_name"] == name
                ])
                for name, _ in SWEEP_ACTIONS
            },
        }

    paired_by_action = {}
    for name, _ in SWEEP_ACTIONS[1:]:
        pairs = [(reference, row) for reference, row in paired if row["residual_name"] == name]
        paired_by_action[name] = {
            "pairs": len(pairs),
            "mean_ttc_reduction_s": float(np.mean([
                float(reference["outcome"]["min_ttc"]) - float(row["outcome"]["min_ttc"])
                for reference, row in pairs
            ])) if pairs else 0.0,
            "mean_distance_reduction_m": float(np.mean([
                float(reference["outcome"]["min_distance"]) - float(row["outcome"]["min_distance"])
                for reference, row in pairs
            ])) if pairs else 0.0,
        }
    return {
        "episodes": len(rows),
        "paired_initial_conditions_verified": all(
            reference["scenario"] == row["scenario"] for reference, row in paired
        ),
        "by_family": by_family,
        "paired_against_base": paired_by_action,
    }


def run(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    criteria = FailureCriteria.from_config(config["failure"])
    model = build_model(config, device)
    model.eval()
    tasks = [
        task for task in load_taskbook(taskbook_path)
        if task.sut_split == "validation"
        and task.geometry_split == "validation"
        and task.functional_split == "train"
    ]
    if {task.functional_scenario for task in tasks} != {"merge", "cutin", "roundabout"}:
        raise ValueError("S0 sweep requires one validation task for each functional family")
    sampler = PretrainSceneSampler(tuple(tasks), CASES_PER_TASK, int(config["seed"]))
    rows = []
    for task in tasks:
        online = build_online(model, task, int(config["training"]["step_budget"]), criteria)
        for residual_name, residual in SWEEP_ACTIONS:
            result = online.run(
                task,
                CASES_PER_TASK,
                deterministic=True,
                posterior_support_limit=0,
                scene_action_provider=sampler,
                inner_action_provider=lambda _, value=residual: np.asarray(value, dtype=np.float32),
            )
            for case_index, episode in enumerate(result.episodes):
                transitions = episode.rollout.transitions
                rows.append({
                    "family": task.functional_scenario,
                    "task_id": task.task_id,
                    "case_index": case_index,
                    "residual_name": residual_name,
                    "residual": list(residual),
                    "scenario": _scenario(episode),
                    "outcome": _outcome(episode.outcome),
                    "challenge_steps": sum(
                        bool(row["info"].get("semantic_challenge_phase_active", False))
                        for row in transitions
                    ),
                    "positive_reward_transitions": sum(
                        float(row["reward_inner"]) > 0.0 for row in transitions
                    ),
                })
    report = {
        "mode": "s0_fixed_residual_action_reachability",
        "scope": "no_rl_training",
        "config": str(config_path),
        "cases_per_task": CASES_PER_TASK,
        "residual_actions": {name: list(action) for name, action in SWEEP_ACTIONS},
        "summary": _summary(rows),
        "rows": rows,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
