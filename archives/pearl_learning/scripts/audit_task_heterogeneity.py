"""Verify task heterogeneity from one or more independently seeded SAC runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from archives.pearl_learning.src.io import content_hash, write_json
from archives.pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise SystemExit(f"required baseline artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _strict(payload: Mapping[str, Any]) -> float:
    return float(payload["summary"]["valid_critical_strict_rate"])


def _budget_value(manifest: Mapping[str, Any], baseline: str, field: str, root: Path) -> float:
    budget = manifest.get("training_budget")
    if not isinstance(budget, Mapping) or field not in budget:
        raise SystemExit(f"{baseline} manifest lacks {field} training-budget provenance: {root}")
    value = budget[field]
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise SystemExit(f"{baseline} manifest has invalid {field} training budget: {root}")
    return float(value)


def _compact_transfer_matrix(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only summaries; per-query records do not establish heterogeneity."""
    matrix = payload.get("matrix", {})
    return {
        "policy_tasks": list(payload.get("policy_tasks", [])),
        "evaluation_tasks": list(payload.get("evaluation_tasks", [])),
        "strict_summary": {
            str(policy_id): {
                str(task_id): float(value["summary"]["valid_critical_strict_rate"])
                for task_id, value in rows.items()
            }
            for policy_id, rows in matrix.items()
        },
    }


def _seed_evidence(root: Path, taskbook_hash: str, train_ids: list[str]) -> dict[str, Any]:
    manifests = {
        name: _load_json(root / name / "baseline_manifest.json")
        for name in ("per_task_sac", "cross_task_policy_matrix", "topology_conditioned_pooled_sac")
    }
    for name, manifest in manifests.items():
        if manifest.get("taskbook_hash") != taskbook_hash or manifest.get("status") != "completed":
            raise SystemExit(f"{name} manifest is incomplete or belongs to another taskbook: {root}")
    seeds = {int(manifest["seed"]) for manifest in manifests.values()}
    if len(seeds) != 1:
        raise SystemExit(f"heterogeneity baseline seeds differ within root: {root}")
    per_task_steps = _budget_value(manifests["per_task_sac"], "per_task_sac", "per_task_environment_steps", root)
    cross_steps = _budget_value(manifests["cross_task_policy_matrix"], "cross_task_policy_matrix", "per_task_environment_steps", root)
    pooled_steps = _budget_value(manifests["topology_conditioned_pooled_sac"], "topology_conditioned_pooled_sac", "per_task_environment_steps", root)
    pooled_total_steps = _budget_value(manifests["topology_conditioned_pooled_sac"], "topology_conditioned_pooled_sac", "total_environment_steps", root)
    if not np.isclose(per_task_steps, cross_steps):
        raise SystemExit(f"cross-task matrix does not match the per-task training budget: {root}")
    if not np.isclose(per_task_steps, pooled_steps):
        raise SystemExit(f"pooled SAC does not receive the same per-task training budget: {root}")
    if not np.isclose(pooled_total_steps, pooled_steps * len(train_ids)):
        raise SystemExit(f"pooled SAC total budget is inconsistent with its per-task allocation: {root}")
    per_task = _load_json(root / "per_task_sac" / "per_task_metrics.json")["tasks"]
    pooled = _load_json(root / "topology_conditioned_pooled_sac" / "pooled_metrics.json")["tasks"]
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
    matrix = _load_json(root / "cross_task_policy_matrix" / "cross_task_matrix.json")
    return {
        "baseline_root": str(root),
        "seed": seeds.pop(),
        "training_budget": {
            "per_task_environment_steps": per_task_steps,
            "pooled_total_environment_steps": pooled_total_steps,
        },
        "mean_per_task_minus_pooled_gap": float(np.mean([row["gap"] for row in rows])),
        "rows": rows,
        "policy_transfer_matrix": _compact_transfer_matrix(matrix),
    }


def heterogeneity_report(taskbook: Mapping[str, list[Any]], baseline_roots: list[str | Path], *, minimum_gap: float, minimum_seeds: int) -> dict[str, Any]:
    if minimum_seeds < 1:
        raise ValueError("minimum_seeds must be positive")
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    train_ids = [task.task_id for task in taskbook["meta_train"]]
    evidence = [_seed_evidence(Path(root), taskbook_hash, train_ids) for root in baseline_roots]
    seeds = [int(item["seed"]) for item in evidence]
    if len(set(seeds)) != len(seeds):
        raise ValueError("each heterogeneity baseline root must use a distinct seed")
    by_task = {task_id: [] for task_id in train_ids}
    for item in evidence:
        for row in item["rows"]:
            by_task[str(row["task_id"])].append(row)
    rows = []
    for task_id in train_ids:
        task_rows = by_task[task_id]
        gaps = np.asarray([float(row["gap"]) for row in task_rows], dtype=float)
        rows.append({
            "task_id": task_id,
            "per_task_valid_critical_strict_rate_mean": float(np.mean([row["per_task_valid_critical_strict_rate"] for row in task_rows])),
            "pooled_valid_critical_strict_rate_mean": float(np.mean([row["pooled_valid_critical_strict_rate"] for row in task_rows])),
            "gap_mean": float(np.mean(gaps)),
            "gap_std_across_seeds": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
        })
    mean_gap = float(np.mean([row["gap_mean"] for row in rows]))
    result = {
        "schema": "logical_merge_task_heterogeneity_audit",
        "taskbook_hash": taskbook_hash,
        "metric": "valid_critical_strict_rate",
        "minimum_gap": float(minimum_gap),
        "minimum_seeds": int(minimum_seeds),
        "seed_count": len(evidence),
        "seed_evidence": evidence,
        "mean_per_task_minus_pooled_gap": mean_gap,
        "per_task": rows,
        "heterogeneous": bool(len(evidence) >= minimum_seeds and mean_gap >= float(minimum_gap)),
    }
    result["status"] = "pass" if result["heterogeneous"] else "fail"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--baseline-root", action="append", required=True, help="repeat once for each independent seed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-gap", type=float, default=0.02, help="minimum mean per-task minus pooled strict-success gap")
    parser.add_argument("--minimum-seeds", type=int, default=1)
    args = parser.parse_args()
    result = heterogeneity_report(
        load_taskbook(args.taskbook), args.baseline_root,
        minimum_gap=args.minimum_gap, minimum_seeds=args.minimum_seeds,
    )
    write_json(args.output, result)
    if result["status"] != "pass":
        raise SystemExit("pooled SAC is too close to per-task SAC; formal PEARL training is blocked")


if __name__ == "__main__":
    main()
