"""Hard all-task topology audit for frozen taskbooks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import numpy as np

from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.task_env import LogicalMergeEnv
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _lane_graph(env: LogicalMergeEnv) -> list[dict[str, Any]]:
    return env.adapter._lane_graph_payload(env._env)


def _run_trace(task: Any, cfg: dict[str, Any], case: dict[str, Any], count: int = 16) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wrapped = LogicalMergeEnv(task, cfg, [case])
    trace: list[dict[str, Any]] = []
    try:
        observation, info = wrapped.reset(options={"case": case})
        ids = (str(wrapped.adversary.id), str(wrapped.sut.id))
        for _ in range(count):
            next_observation, _, terminated, truncated, step_info = wrapped.step(np.zeros(2, dtype=np.float32))
            trace.append({"adv_s": float(step_info["adversary_route_progress_m"]), "sut_s": float(step_info["sut_route_progress_m"]), "wrong_route": bool(step_info["wrong_route"]), "ids": (str(wrapped.adversary.id), str(wrapped.sut.id)), "terminated": bool(terminated), "truncated": bool(truncated)})
            observation = next_observation
            if terminated or truncated:
                break
        summary = {
            "task_id": task.task_id, "geometry_id": task.geometry_id, "map_hash": info["map_hash"], "lane_graph": _lane_graph(wrapped),
            "conflict_frame": {"origin": np.asarray(wrapped._frame["origin"]).round(6).tolist(), "radius_m": wrapped._frame["radius_m"]},
            "role_ids": ids, "adversary_lane_index": list(wrapped.adversary.lane_index), "sut_lane_index": list(wrapped.sut.lane_index),
            "observation_shape": list(observation.shape), "observation_finite": bool(np.all(np.isfinite(observation))), "trace_length": len(trace),
        }
        return summary, trace
    finally:
        wrapped.close()


def audit_task(task: Any, cfg: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    try:
        first, trace = _run_trace(task, cfg, case)
        second, replay = _run_trace(task, cfg, case)
        progress = all(row["adv_s"] >= previous["adv_s"] - 1.0 and row["sut_s"] >= previous["sut_s"] - 1.0 for previous, row in zip(trace, trace[1:]))
        stable_ids = all(tuple(row["ids"]) == tuple(first["role_ids"]) for row in trace)
        natural_wrong = any(row["wrong_route"] for row in trace)
        replayable = len(trace) == len(replay) and all(abs(a["adv_s"] - b["adv_s"]) <= 1e-5 and abs(a["sut_s"] - b["sut_s"]) <= 1e-5 for a, b in zip(trace, replay))
        route_geometry = []
        for role in ("adversary", "sut"):
            # Reconstruct once from the trusted replay environment so a frozen
            # lane-centre discontinuity cannot hide beyond the short trace.
            wrapped = LogicalMergeEnv(task, cfg, [case])
            try:
                wrapped.reset(options={"case": case})
                route = wrapped._frame[f"{role}_route"]
                segments = np.diff(route.points, axis=0)
                lengths = np.linalg.norm(segments, axis=1)
                headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
                route_geometry.append({
                    "role": role,
                    "max_segment_m": float(lengths.max()),
                    "max_heading_jump_rad": float(np.max(np.abs(np.diff(headings)))) if len(headings) > 1 else 0.0,
                })
            finally:
                wrapped.close()
        smooth_route_geometry = all(
            row["max_segment_m"] <= 3.0 and row["max_heading_jump_rad"] <= 0.5
            for row in route_geometry
        )
        checks = {
            "map_hash_matches_taskbook": first["map_hash"] == task.map_hash,
            "conflict_route_constructed": bool(first["conflict_frame"]),
            "zero_action_roles_move": len(trace) > 0 and trace[-1]["adv_s"] > trace[0]["adv_s"] and trace[-1]["sut_s"] > trace[0]["sut_s"],
            "no_natural_wrong_route": not natural_wrong,
            "route_progress_monotonic": progress,
            "role_ids_stable": stable_ids,
            "case_trace_replayable": replayable,
            "observation_contract": first["observation_shape"] == [37] and first["observation_finite"],
            "smooth_route_geometry": smooth_route_geometry,
        }
        return {**first, "route_geometry": route_geometry, "checks": checks, "status": "pass" if all(checks.values()) else "fail"}
    except Exception as exc:
        return {"task_id": task.task_id, "geometry_id": task.geometry_id, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--taskbook", required=True); parser.add_argument("--casebook-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = read_config(args.config); taskbook = load_taskbook(args.taskbook); reports = []
    for tasks in taskbook.values():
        for task in tasks:
            case = load_casebook(task, args.casebook_root)["test_support"][0]
            reports.append(audit_task(task, cfg, case))
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    payload = {"schema": "logical_merge_topology_audit", "taskbook_hash": taskbook_hash, "reports": reports, "passed": sum(row["status"] == "pass" for row in reports), "total": len(reports)}
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    for report in reports:
        write_json(root / f"{report['task_id']}.json", report)
    write_json(root / "topology_audit.json", payload)
    if payload["passed"] != payload["total"]:
        raise SystemExit("topology audit failed; formal training is blocked")
    print(f"topology audit passed: {payload['passed']}/{payload['total']}")


if __name__ == "__main__":
    main()
