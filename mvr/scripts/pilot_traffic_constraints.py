"""Run six fixed lawful and emergency cut-in checks without writing artifacts."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.option import AdversarialOption
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.registry import load_adapters
from ..scenario.taskbook import load_taskbook
from ..safety import TrafficActionShield
from ..training.runner import HierarchicalRunner


def _task() -> Any:
    return next(
        task
        for task in load_taskbook("mvr/configs/taskbook.json")
        if task.task_id == "cutin-g04-fast_small_gap"
    )


def _case_action(candidate_index: int, name: str) -> NormalizedScenarioAction:
    if name == "emergency_merge":
        continuous = (0.0, 0.0, 0.0, 0.0)
    elif name == "speed_and_brake":
        continuous = (0.0, 0.0, 1.0, 1.0)
    elif name == "legal_merge":
        continuous = (-1.0, 1.0, 0.0, 0.0)
    else:
        raise ValueError(f"unknown pilot case {name!r}")
    return NormalizedScenarioAction(
        candidate_index,
        continuous,
        AdversarialOption.APPROACH_CONFLICT,
    )


def _inner_action(name: str, lane_change_steering: float, step: int) -> np.ndarray:
    return np.asarray(
        (0.0, 1.0 if step < 18 else -1.0)
        if name == "speed_and_brake"
        else (lane_change_steering, 0.0),
        dtype=np.float32,
    )


def run_case(candidate_index: int, name: str) -> dict[str, Any]:
    task = _task()
    executor = ScenarioExecutor(load_adapters(), mvr_parameter_spaces())
    episode = executor.reset(
        task,
        _case_action(candidate_index, name),
        episode_seed=204 + candidate_index,
    )
    try:
        contract = episode.layout.traffic_contract
        target_lane = episode.env.current_map.road_network.get_lane(
            (*episode.layout.adversary_lane[:2], contract.target_lane_number)
        )
        lane_change_steering = float(np.sign(
            TrafficActionShield._lane_follow_action(episode.adversary, target_lane)
        ))
        max_steps = 60 if name == "legal_merge" else 35
        step = 0

        def inner_action(_: np.ndarray) -> np.ndarray:
            nonlocal step
            action = _inner_action(name, lane_change_steering, step)
            step += 1
            return action

        rollout = HierarchicalRunner(max_steps=max_steps).rollout(
            episode,
            "cutin",
            AdversarialOption.APPROACH_CONFLICT.value,
            inner_action,
        )
        route_progress_m = episode.adversary_route.projection(
            episode.adversary.position,
            episode.adversary.heading_theta,
        ).s_m
    finally:
        episode.env.close()
    infos = [row["info"] for row in rollout.transitions]
    final = infos[-1]
    report = {
        "candidate": episode.layout.candidate,
        "case": name,
        "steps": len(infos),
        "route_length_m": episode.adversary_route.length_m,
        "merge_window_s": episode.layout.traffic_contract.merge_window_s,
        "route_progress_m": float(route_progress_m),
        "outcome": dict(rollout.outcome),
        "is_valid_episode": bool(rollout.outcome["is_valid_episode"]),
        "target_collision": bool(rollout.outcome["target_collision"]),
        "rejection_counts": dict(final["traffic_rejection_counts"]),
        "max_speed_mps": float(final["traffic_max_speed_mps"]),
        "max_abs_acceleration_mps2": float(final["traffic_max_abs_acceleration_mps2"]),
        "max_abs_jerk_mps3": float(final["traffic_max_abs_jerk_mps3"]),
        "max_lateral_acceleration_mps2": float(final["traffic_max_lateral_acceleration_mps2"]),
        "min_applied_longitudinal_action": min(
            float(row.get("executed_action", row["action"])[1])
            for row in rollout.transitions
        ),
        "lane_change_started": bool(final["traffic_lane_change_started"]),
        "lane_change_completed": bool(final["traffic_lane_change_completed"]),
    }
    if name == "emergency_merge":
        if not report["lane_change_started"]:
            raise AssertionError("lawful emergency cut-in was blocked by a conservative gap rule")
    elif name == "speed_and_brake":
        if report["max_speed_mps"] > 20.5:
            raise AssertionError("speed cap was not enforced")
        if report["min_applied_longitudinal_action"] >= 0.0:
            raise AssertionError("emergency braking command was not executed")
    else:
        if not report["lane_change_started"] or not report["lane_change_completed"]:
            raise AssertionError("legal cut-in did not complete")
        if report["route_progress_m"] < report["merge_window_s"][1]:
            raise AssertionError("legal cut-in did not pass through the full merge window")
    if report["outcome"]["adversary_traffic_violation"]:
        raise AssertionError(
            f"shielded pilot produced a traffic violation: "
            f"{report['outcome']['traffic_telemetry']['violation_counts']}"
            f", max_abs_lane_lateral_m="
            f"{report['outcome']['traffic_telemetry']['max_abs_lane_lateral_m']:.3f}"
        )
    return report


def run() -> list[dict[str, Any]]:
    reports = [
        run_case(candidate_index, name)
        for candidate_index in range(2)
        for name in ("emergency_merge", "speed_and_brake", "legal_merge")
    ]
    return reports


def main() -> None:
    reports = run()
    print(json.dumps([
        {
            key: report[key]
            for key in (
                "candidate",
                "case",
                "steps",
                "route_length_m",
                "route_progress_m",
                "is_valid_episode",
                "target_collision",
                "rejection_counts",
                "max_speed_mps",
                "max_abs_acceleration_mps2",
                "max_abs_jerk_mps3",
                "max_lateral_acceleration_mps2",
                "min_applied_longitudinal_action",
                "lane_change_started",
                "lane_change_completed",
            )
        }
        for report in reports
    ], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
