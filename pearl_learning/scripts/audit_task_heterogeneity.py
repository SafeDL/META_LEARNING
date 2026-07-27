"""Reject formal PEARL runs when pooled SAC matches per-task SAC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pearl_learning.src.io import content_hash, write_json
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _load_metrics(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise SystemExit(f"required baseline metric artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _strict(payload: Mapping[str, Any]) -> float:
    return float(payload["summary"]["valid_critical_strict_rate"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transfer-matrix", help="optional cross-task matrix artifact; defaults to <baseline-root>/cross_task_policy_matrix/cross_task_matrix.json")
    parser.add_argument("--minimum-gap", type=float, default=0.02,
                        help="minimum mean per-task minus pooled strict-success gap")
    args = parser.parse_args()
    taskbook = load_taskbook(args.taskbook)
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    root = Path(args.baseline_root)
    per_task = _load_metrics(root / "per_task_sac" / "per_task_metrics.json")["tasks"]
    pooled = _load_metrics(root / "topology_conditioned_pooled_sac" / "pooled_metrics.json")["tasks"]
    matrix_path = Path(args.transfer_matrix) if args.transfer_matrix else root / "cross_task_policy_matrix" / "cross_task_matrix.json"
    matrix = _load_metrics(matrix_path)
    train_ids = [task.task_id for task in taskbook["meta_train"]]
    missing = [task_id for task_id in train_ids if task_id not in per_task or task_id not in pooled]
    if missing:
        raise SystemExit(f"matched per-task/pooled metrics are missing for: {missing}")
    rows = [{
        "task_id": task_id,
        "per_task_valid_critical_strict_rate": _strict(per_task[task_id]),
        "pooled_valid_critical_strict_rate": _strict(pooled[task_id]),
    } for task_id in train_ids]
    for row in rows:
        row["gap"] = row["per_task_valid_critical_strict_rate"] - row["pooled_valid_critical_strict_rate"]
    mean_gap = float(np.mean([row["gap"] for row in rows]))
    # The transfer matrix is preserved verbatim as evidence that policies are
    # not interchangeable across the frozen logical-task partition.
    result = {
        "schema": "logical_merge_task_heterogeneity_audit",
        "taskbook_hash": taskbook_hash,
        "metric": "valid_critical_strict_rate",
        "minimum_gap": float(args.minimum_gap),
        "mean_per_task_minus_pooled_gap": mean_gap,
        "per_task": rows,
        "policy_transfer_matrix": matrix,
        "policy_transfer_matrix_source": str(matrix_path),
        "heterogeneous": bool(mean_gap >= float(args.minimum_gap)),
    }
    result["status"] = "pass" if result["heterogeneous"] else "fail"
    write_json(args.output, result)
    if result["status"] != "pass":
        raise SystemExit("pooled SAC is too close to per-task SAC; formal PEARL training is blocked")


if __name__ == "__main__":
    main()
