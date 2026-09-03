"""Run train-split physical probes for the shared Frenet action contract."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..failure.criteria import DEFAULT_FAILURE_CRITERIA
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.registry import load_adapters
from ..scenario.taskbook import load_taskbook
from ..training.runner import HierarchicalRunner


PROBE_ACTIONS = {
    "baseline": np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32),
    "short_early": np.asarray((-1.0, 1.0, -1.0, 0.0), dtype=np.float32),
    "long_late": np.asarray((1.0, -1.0, 1.0, 0.0), dtype=np.float32),
}


def _task(family: str) -> Any:
    task = next(
        task for task in load_taskbook("mvr/configs/taskbook.json")
        if task.functional_scenario == family
        and task.sut_split == "train"
        and task.geometry_split == "train"
        and task.logical_split == "train"
    )
    if family == "cutin":
        return task
    bounds = dict(task.logical_domain_bounds)
    bounds["maneuver_onset_progress"] = (-1.0, 1.0)
    return replace(
        task,
        task_id=f"{task.task_id}:frenet_probe",
        logical_domain_id="frenet_probe",
        logical_domain_bounds=bounds,
        logical_parameter_mask=(True,) * len(task.logical_parameter_mask),
    )


def _oscillations(values: np.ndarray, deadband: float = 0.03) -> int:
    signs = np.sign(values[np.abs(values) >= deadband])
    return int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


def run() -> dict[str, Any]:
    executor = ScenarioExecutor(load_adapters(), mvr_parameter_spaces())
    records = []
    for family in ("cutin", "merge", "roundabout"):
        task = _task(family)
        candidate_count = len(mvr_parameter_spaces()[family].candidates)
        for candidate in range(candidate_count):
            for action_label, planner_action in PROBE_ACTIONS.items():
                if family == "cutin":
                    logical_action = tuple(
                        float(upper if index in {0, 2, 4} else lower)
                        for index, (lower, upper) in enumerate(
                            task.logical_domain_bounds.values()
                        )
                    )
                else:
                    logical_action = tuple(
                        0.0 if not task.logical_parameter_mask[index]
                        else float(
                            upper if index in {0, 3} else lower
                        )
                        for index, (lower, upper) in enumerate(
                            task.logical_domain_bounds.values()
                        )
                    )
                episode = executor.reset(
                    task,
                    NormalizedScenarioAction(
                        candidate,
                        logical_action,
                    ),
                    episode_seed=task.geometry_seed + 100 * candidate,
                )
                try:
                    rollout = HierarchicalRunner(
                        max_steps=360,
                        criteria=DEFAULT_FAILURE_CRITERIA,
                    ).rollout(
                        episode,
                        family,
                        lambda _state, value=planner_action: value,
                    )
                finally:
                    episode.env.close()
                transitions = rollout.transitions
                control_transitions = (
                    transitions[:-1]
                    if rollout.outcome["target_collision"] and len(transitions) > 1
                    else transitions
                )
                active = [
                    row for row in control_transitions
                    if float(row["info"].get("maneuver_start_remaining_m", 1.0)) <= 0.0
                ] or control_transitions
                tracking = np.asarray([
                    abs(float(row["info"]["maneuver_reference_lateral_error_m"]))
                    for row in active
                ])
                steering = np.asarray([
                    float(row["executed_vehicle_action"][0]) for row in active
                ])
                actual_planner = np.asarray([
                    row["planner_action"] for row in active
                ], dtype=float)
                final = control_transitions[-1]["info"]
                records.append({
                    "family": family,
                    "task_id": task.task_id,
                    "candidate_index": candidate,
                    "action_label": action_label,
                    "planner_action": planner_action.tolist(),
                    "effective_planner_action_mean": actual_planner.mean(axis=0).tolist(),
                    "formal_valid": bool(rollout.outcome["is_valid_episode"]),
                    "termination_reason": rollout.outcome["termination_reason"],
                    "maneuver_completed": any(
                        row["info"].get("semantic_maneuver_completed", False)
                        for row in transitions
                    ),
                    "tracking_rms_m": float(np.sqrt(np.mean(np.square(tracking)))),
                    "tracking_p95_m": float(np.quantile(tracking, 0.95)),
                    "steering_sign_changes": _oscillations(steering),
                    "max_abs_acceleration_mps2": float(
                        final["traffic_max_abs_acceleration_mps2"]
                    ),
                    "max_abs_jerk_mps3": float(
                        final["traffic_max_abs_jerk_mps3"]
                    ),
                    "max_lateral_acceleration_mps2": float(
                        final["traffic_max_lateral_acceleration_mps2"]
                    ),
                })
    gates = {
        "all_shape_actions_reach_planner": all(
            (
                row["action_label"] == "baseline"
                and np.linalg.norm(row["effective_planner_action_mean"][:3]) < 1e-6
            )
            or (
                row["action_label"] != "baseline"
                and np.linalg.norm(row["effective_planner_action_mean"][:3]) > 0.5
            )
            for row in records
        ),
        "tracking_rms_at_most_0_35_m": all(
            row["tracking_rms_m"] <= 0.35 for row in records
        ),
        "tracking_p95_at_most_0_75_m": all(
            row["tracking_p95_m"] <= 0.75 for row in records
        ),
    }
    return {
        "scope": {
            "splits": "train_only",
            "outer_run": False,
            "test_split_accessed": False,
        },
        "records": records,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
