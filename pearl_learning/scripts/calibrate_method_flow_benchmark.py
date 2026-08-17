"""Run the validation-only 36/60 episode method-flow benchmark calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pearl_learning.src.benchmark_calibration import calibrate_thresholds, run_baseline_rollout
from pearl_learning.src.casebook_v2 import generate_controlled_cases
from pearl_learning.src.io import content_hash
from pearl_learning.src.io import read_config, write_json
from pearl_learning.src.task_env import LogicalMergeEnv
from pearl_learning.src.taskbook import load_taskbook


def _run(
    count: int,
    cfg: dict[str, Any],
    tasks: list[Any],
    casebooks: dict[str, Any],
    *,
    start: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        # The frozen pilot has six validation query cases and four disjoint
        # validation support cases.  The second-stage 10-case expansion stays
        # validation-only by using their union; test/OOD cases never enter.
        candidate_pool = (
            list(casebooks[task.task_id]["validation_query"])
            + list(casebooks[task.task_id]["validation_support"])
        )
        cases = candidate_pool[start:count]
        if len(candidate_pool) < count:
            raise ValueError(f"{task.task_id} lacks {count} validation candidates")
        for case in cases:
            for policy in ("zero", "random", "heuristic"):
                rows.append(run_baseline_rollout(task, case, cfg, policy))
    return rows


def _calibration_candidates(task: Any, cfg: dict[str, Any], count: int = 10) -> list[dict[str, Any]]:
    """Create validation-only candidates with a controlled provisional gap."""
    def measure(case: dict[str, Any]) -> dict[str, Any]:
        env = LogicalMergeEnv(task, cfg, [case])
        try:
            env.reset(options={"case": case})
            return dict(env.initial_case_measurements())
        finally:
            env.close()

    candidate_hash = content_hash({
        "schema": "merge_calibration_candidates_v1", "task_id": task.task_id,
        "provisional_gap_s": float(cfg["critical_metric"]["arrival_gap_threshold_s"]),
    })
    return generate_controlled_cases(
        task, "validation_query", count, cfg,
        arrival_gap_threshold_s=float(cfg["critical_metric"]["arrival_gap_threshold_s"]),
        calibration_hash=candidate_hash,
        measure_case=measure,
        reachable_count=count // 2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-rollouts", help="reuse an already generated validation-only rollout JSON")
    args = parser.parse_args()
    cfg = read_config(args.config)
    taskbook = load_taskbook(args.taskbook)
    wanted = set(cfg["method_flow_pilot"]["task_ids"]["meta_validation"])
    tasks = [task for task in taskbook["meta_validation"] if task.geometry_id in wanted]
    if args.reuse_rollouts:
        rollouts = json.loads(Path(args.reuse_rollouts).read_text(encoding="utf-8"))
        if len(rollouts) not in {36, 60}:
            raise ValueError("reused calibration must contain exactly 36 or 60 episodes")
        if {row["task_id"] for row in rollouts} != {task.task_id for task in tasks}:
            raise ValueError("reused calibration contains a non-validation or unexpected task")
    else:
        # Existing casebooks are deliberately not used to tune v2 thresholds.
        # Generate a tiny, geometry-controlled validation candidate pool instead.
        casebooks = {
            task.task_id: {"validation_query": _calibration_candidates(task, cfg), "validation_support": []}
            for task in tasks
        }
        rollouts = _run(6, cfg, tasks, casebooks)
    result = calibrate_thresholds(
        rollouts,
        min_arrival_gap_threshold_s=float(cfg["environment"]["decision_step_s"]),
    )
    if result["status"] != "pass" and not args.reuse_rollouts:
        rollouts.extend(_run(10, cfg, tasks, casebooks, start=6))
        result = calibrate_thresholds(
            rollouts,
            min_arrival_gap_threshold_s=float(cfg["environment"]["decision_step_s"]),
        )
    root = Path(args.output)
    write_json(root / "calibration_rollouts.json", rollouts)
    write_json(root / "critical_thresholds.json", result)
    print(f"benchmark calibration {result['status']}: {root / 'critical_thresholds.json'}")
    if result["status"] != "pass":
        raise SystemExit("calibration failed; training remains blocked")


if __name__ == "__main__":
    main()
