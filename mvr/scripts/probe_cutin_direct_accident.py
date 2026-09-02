"""Probe a legal direct Cut-in action against the validation contract.

This is a deterministic environment sanity check, not a training or test-split
evaluation.  It demonstrates that the direct two-dimensional SAC interface can
reach a valid target collision when the fixed logical query and a constant
bounded steering command are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..experiments.cutin_inner import select_cutin_validation_tasks
from ..failure.criteria import FailureCriteria
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.taskbook import load_taskbook
from ..state import INNER_STATE_FIELDS, PhysicalStateExtractor
from ..training.pipeline import (
    build_model,
    load_config,
)
from ..training.trainers import build_online


def run(config_path: str) -> dict[str, object]:
    config, taskbook_path, device = load_config(config_path)
    cutin_inner = config.get("cutin_inner", {})
    if bool(cutin_inner.get("allow_outer", True)):
        raise ValueError("direct Cut-in probe requires allow_outer: false")
    model = build_model(config, device)
    model.eval()
    task = select_cutin_validation_tasks(load_taskbook(taskbook_path))[0]
    payload = config["evaluation"]["fixed_query_x0"]
    x0 = NormalizedScenarioAction(
        int(payload["candidate_index"]),
        tuple(float(value) for value in payload["continuous"]),
    )
    online = build_online(
        model, task, int(config["training"]["step_budget"]),
        FailureCriteria.from_config(config["failure"]),
    )
    lateral_index = INNER_STATE_FIELDS.index("cutin_reference_lateral_error_m")
    progress_index = INNER_STATE_FIELDS.index("cutin_reference_progress")

    def direct_cutin_controller(state: np.ndarray) -> tuple[float, float]:
        lateral_error = float(state[lateral_index] * PhysicalStateExtractor.scales[lateral_index])
        steering = float(np.clip(0.25 * lateral_error, -1.0, 1.0))
        progress = float(state[progress_index] * PhysicalStateExtractor.scales[progress_index])
        # A legal feasibility witness: follow the immutable reference, then
        # apply bounded direct braking only after entering the target lane.
        return steering, -1.0 if progress >= 0.35 else 0.0

    result = online.run(
        task,
        1,
        deterministic=True,
        posterior_support_limit=0,
        scene_action_provider=lambda *_: x0,
        # This fixed feedback controller is a simulator feasibility check,
        # not an SAC policy or a source of replay transitions.
        inner_action_provider=direct_cutin_controller,
        episode_seed_provider=lambda current, _index: current.geometry_seed + 1_100_000,
    )
    episode = result.episodes[0]
    outcome = episode.outcome
    telemetry = outcome.get("traffic_telemetry", {})
    trace = [
        {
            "step": index,
            "raw_action": np.asarray(row["raw_action"], dtype=float).tolist(),
            "executed_action": np.asarray(row["executed_action"], dtype=float).tolist(),
            "acceleration_mps2": float(row["info"]["traffic_acceleration_mps2"]),
            "target_lateral_m": row["info"].get("traffic_cutin_lateral_m"),
            "reference_speed_limit_mps": float(
                row["info"]["cutin_reference_speed_limit_mps"]
            ),
            "maneuver_active": bool(row["info"].get("semantic_maneuver_active", False)),
            "violations": dict(row["info"]["traffic_violation_counts"]),
        }
        for index, row in enumerate(episode.rollout.transitions)
    ]
    return {
        "scope": {
            "functional_scenario": "cutin",
            "sut_split": "validation",
            "geometry_split": "train",
            "logical_split": "validation",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "task_id": task.task_id,
        "fixed_query_x0": {
            "candidate_index": x0.candidate_index,
            "continuous": list(x0.continuous),
        },
        "direct_controller": "reference_path_follow_then_bounded_brake",
        "outcome": {
            key: outcome.get(key)
            for key in (
                "termination_reason", "event_kind", "event_semantic_valid",
                "event_traffic_valid", "valid_target_collision",
                "valid_critical_near_miss", "is_valid_episode", "is_failure",
                "min_ttc", "min_distance", "max_closing_speed",
            )
        },
        "traffic_telemetry": {
            key: telemetry.get(key)
            for key in (
                "violation_counts", "warning_counts", "max_speed_mps",
                "max_abs_acceleration_mps2", "max_abs_jerk_mps3",
                "max_lateral_acceleration_mps2",
            )
        },
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.config)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
