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

from ..experiments.cutin_inner import select_cutin_validation_tasks
from ..failure.criteria import FailureCriteria
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.taskbook import load_taskbook
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import (
    build_model,
    checkpoint_config_hash,
    load_config,
)
from ..training.trainers import build_online


def run(config_path: str, checkpoint_path: str) -> dict[str, object]:
    config, taskbook_path, device = load_config(config_path)
    cutin_inner = config.get("cutin_inner", {})
    if bool(cutin_inner.get("allow_outer", True)):
        raise ValueError("direct Cut-in probe requires allow_outer: false")
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config),
    )
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
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
    result = online.run(
        task,
        1,
        deterministic=True,
        posterior_support_limit=0,
        scene_action_provider=lambda *_: x0,
        # A fixed bounded direct steering command is used only to verify that
        # the physical environment remains capable of a valid target event.
        inner_action_provider=lambda _state: (-1.0, 0.0),
        episode_seed_provider=lambda current, _index: current.geometry_seed + 1_100_000,
    )
    episode = result.episodes[0]
    outcome = episode.outcome
    telemetry = outcome.get("traffic_telemetry", {})
    return {
        "scope": {
            "functional_scenario": "cutin",
            "sut_split": "validation",
            "geometry_split": "train",
            "logical_split": "validation",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "checkpoint_stage": checkpoint.stage,
        "task_id": task.task_id,
        "fixed_query_x0": {
            "candidate_index": x0.candidate_index,
            "continuous": list(x0.continuous),
        },
        "direct_action": [-1.0, 0.0],
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.config, args.checkpoint)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
