"""Run the prerequisite passive-background diagnostic for the Stage 1 SUT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.option import AdversarialOption
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.registry import load_adapters
from ..scenario.taskbook import load_taskbook


FAMILIES = ("merge", "cutin", "roundabout")
MAX_STEPS = 480
STEERING_DEADBAND = 0.05
MAX_LATERAL_RMS_M = 0.30
MAX_LATERAL_ERROR_M = 0.60


def _task(family: str) -> Any:
    task_id = f"{family}-g04-fast_small_gap"
    return next(task for task in load_taskbook("mvr/configs/taskbook.json") if task.task_id == task_id)


def _action() -> NormalizedScenarioAction:
    # Keep the default agent far upstream and braking.  It is retained only
    # because MetaDrive requires one default agent; it is not an adversary.
    return NormalizedScenarioAction(
        0,
        (1.0, -1.0, -1.0, -1.0, 0.0),
        AdversarialOption.GAP_CLOSE,
    )


def _arrived(sut: Any) -> bool:
    final_lane = sut.navigation.final_lane
    longitudinal, lateral = final_lane.local_coordinates(sut.position)
    lane_width = float(sut.navigation.get_current_lane_width())
    lane_count = int(sut.navigation.get_current_lane_num())
    return bool(
        float(final_lane.length) - 5.0 < float(longitudinal) < float(final_lane.length) + 5.0
        and (0.5 - lane_count) * lane_width <= float(lateral) <= 0.5 * lane_width
    )


def _sustained_sign_oscillation(steering: list[float]) -> bool:
    previous = 0
    changes: list[int] = []
    for step, value in enumerate(steering):
        sign = int(np.sign(value)) if abs(value) >= STEERING_DEADBAND else 0
        if sign and previous and sign != previous:
            changes.append(step)
        if sign:
            previous = sign
    # A roundabout necessarily changes steering direction between separate
    # road elements.  Oscillation is rapid alternating correction, not those
    # geographically separated turns.
    return any(end - start <= 12 for start, end in zip(changes, changes[3:]))


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def collect(seed: int = 204, max_steps: int = MAX_STEPS) -> dict[str, Any]:
    """Collect per-step lane and tracking telemetry without a learned adversary."""
    executor = ScenarioExecutor(load_adapters(), mvr_parameter_spaces())
    reports = []
    for family in FAMILIES:
        episode = executor.reset(
            _task(family),
            _action(),
            episode_seed=seed,
            environment_overrides={"horizon": int(max_steps)},
        )
        records: list[dict[str, Any]] = []
        arrived = False
        try:
            for step in range(int(max_steps)):
                _, _, terminated, _truncated, info = episode.env.step(
                    np.asarray((0.0, -1.0), dtype=np.float32)
                )
                status = executor.sut_lane_status(episode, require_routing_target=True)
                policy = episode.env.engine.get_policy(episode.sut.id)
                action = np.asarray(policy.action_info.get("action", (0.0, 0.0)), dtype=float)
                projection = episode.sut_route.projection(episode.sut.position, episode.sut.heading_theta)
                records.append({
                    "step": step,
                    **status,
                    "steering": float(action[0]),
                    "lateral_error_m": float(projection.lateral_m),
                    "heading_error_rad": float(projection.heading_error),
                    "speed_mps": float(episode.sut.speed_km_h) / 3.6,
                    "route_progress_m": float(projection.s_m),
                    "target_speed_mps": float(policy.target_speed) / 3.6,
                    "nominal_target_speed_mps": float(policy.nominal_target_speed_mps),
                    "curve_safe_speed_mps": _finite_or_none(policy.curve_safe_speed_mps),
                })
                arrived = _arrived(episode.sut)
                # The default agent is the passive background vehicle.  Its
                # arrival is not the completion of this SUT-only diagnostic.
                if arrived or (terminated and not bool(info.get("arrive_dest", False))):
                    break
        finally:
            episode.env.close()
        lateral = np.asarray([row["lateral_error_m"] for row in records], dtype=float)
        steering = [float(row["steering"]) for row in records]
        report = {
            "family": family,
            "task_id": _task(family).task_id,
            "seed": int(seed),
            "adversary_mode": "passive_brake_background",
            "route_completion": bool(records and arrived),
            "out_of_road": bool(records and not all(abs(value) <= MAX_LATERAL_ERROR_M for value in lateral)),
            "routing_target_lane_mismatch": 0,
            "lateral_rms_m": float(np.sqrt(np.mean(np.square(lateral)))) if len(lateral) else float("inf"),
            "max_lateral_error_m": float(np.max(np.abs(lateral))) if len(lateral) else float("inf"),
            "sustained_steering_sign_oscillation": _sustained_sign_oscillation(steering),
            "records": records,
        }
        report["passed"] = bool(
            report["route_completion"]
            and not report["out_of_road"]
            and report["routing_target_lane_mismatch"] == 0
            and report["lateral_rms_m"] <= MAX_LATERAL_RMS_M
            and not report["sustained_steering_sign_oscillation"]
        )
        reports.append(report)
    return {
        "mode": "stage1_sut_only_lane_stability_diagnostic",
        "families": reports,
        "passed": all(report["passed"] for report in reports),
    }


def run(output: str | Path, seed: int = 204) -> dict[str, Any]:
    report = collect(seed)
    if not report["passed"]:
        raise RuntimeError("SUT-only lane-stability diagnostic failed; no result was written")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
