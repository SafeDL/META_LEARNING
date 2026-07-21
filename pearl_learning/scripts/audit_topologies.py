"""Audit actual MetaDrive maps before any PEARL training is permitted."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np

from pearl_learning.src.casebook import build_casebook
from pearl_learning.src.io import read_config, write_json
from pearl_learning.src.task_env import LogicalMergeEnv
from pearl_learning.src.taskbook import build_taskbook


def lane_graph(env: Any) -> list[dict[str, Any]]:
    rows = []
    for start, ends in env._env.current_map.road_network.graph.items():
        for end, lanes in ends.items():
            for index, lane in enumerate(lanes):
                rows.append({"lane_index": [str(start), str(end), index], "length_m": float(lane.length), "start": np.asarray(lane.position(0.0, 0.0), dtype=float).round(5).tolist(), "end": np.asarray(lane.position(float(lane.length), 0.0), dtype=float).round(5).tolist()})
    return rows


def audit_task(task: Any, cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    case = build_casebook(task, cfg)["test_support"][0]
    wrapped = LogicalMergeEnv(task, cfg, [case])
    try:
        observation, info = wrapped.reset(options={"case": case})
        graph = lane_graph(wrapped)
        before = (np.asarray(wrapped.adversary.position, dtype=float), np.asarray(wrapped.sut.position, dtype=float))
        next_observation, _, _, _, step_info = wrapped.step(np.zeros(2, dtype=np.float32))
        after = (np.asarray(wrapped.adversary.position, dtype=float), np.asarray(wrapped.sut.position, dtype=float))
        route_motion = [float(np.linalg.norm(a - b)) for a, b in zip(before, after)]
        # A genuine candidate needs two initial routes and both roles to move;
        # map rendering is diagnostic only and never defines topology semantics.
        result = {"task_id": task.task_id, "logical_type": task.logical_type, "status": "pass", "lane_graph": graph, "conflict_frame": {"origin": np.asarray(wrapped._frame["origin"]).round(5).tolist(), "radius_m": wrapped._frame["radius_m"]}, "adversary_lane_index": [str(x) for x in wrapped.adversary.lane_index], "sut_lane_index": [str(x) for x in wrapped.sut.lane_index], "initial_role_distance_m": float(np.linalg.norm(before[0] - before[1])), "route_motion_m_one_step": route_motion, "observation_shape": list(observation.shape), "observation_finite": bool(np.all(np.isfinite(observation)) and np.all(np.isfinite(next_observation))), "target_contact_probe_method": step_info["target_contact_method"], "replay_case_id": info["case_id"]}
    except Exception as exc:
        result = {"task_id": task.task_id, "logical_type": task.logical_type, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        wrapped.close()
    write_json(root / f"{task.task_id}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    cfg = read_config(args.config); root = Path(cfg["project"]["output_root"]) / "topology_audit"; root.mkdir(parents=True, exist_ok=True)
    tasks = build_taskbook(cfg)
    representatives = [tasks["meta_train"][0], tasks["meta_train"][4], tasks["meta_train"][8], tasks["meta_test_logical"][0]]
    report = [audit_task(task, cfg, root) for task in representatives]
    write_json(root / "topology_audit.json", {"metadrive_required": True, "reports": report, "passed": sum(x["status"] == "pass" for x in report), "total": len(report)})
    if any(x["status"] != "pass" for x in report): raise SystemExit("topology audit failed; do not start PEARL")
    print(f"topology audit passed: {len(report)}/{len(report)}")


if __name__ == "__main__": main()
