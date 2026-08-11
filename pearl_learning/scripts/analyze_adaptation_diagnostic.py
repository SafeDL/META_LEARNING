"""Summarize a non-formal posterior-adaptation training diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pearl_learning.src.casebook import physical_geometry_id
from pearl_learning.src.io import content_hash, write_json
from pearl_learning.src.taskbook import load_taskbook


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records_summary(records: list[Mapping[str, Any]]) -> dict[str, float | int]:
    count = len(records)
    target = sum(bool(record["target_collision"]) for record in records)
    non_target = sum(bool(record["non_target_collision"]) for record in records)
    contact = target + non_target
    return {
        "episodes": count,
        "physical_contact_episodes": contact,
        "physical_contact_rate": float(contact / count) if count else 0.0,
        "target_contact_episodes": target,
        "target_contact_rate": float(target / count) if count else 0.0,
        "non_target_collision_episodes": non_target,
        "non_target_collision_rate": float(non_target / count) if count else 0.0,
        "mean_episode_return": float(np.mean([float(record["episode_return"]) for record in records])) if records else 0.0,
    }


def _posterior_summary(tasks: Mapping[str, Any], shot: int) -> dict[str, Any]:
    means = {
        task_id: np.asarray(rows[str(shot)]["posterior_mean"], dtype=float).reshape(-1)
        for task_id, rows in tasks.items()
    }
    stds = [
        np.sqrt(np.asarray(rows[str(shot)]["posterior_variance"], dtype=float).reshape(-1))
        for rows in tasks.values()
    ]
    moves = [
        float(rows[str(shot)]["posterior_change"]["mean_l2_from_k0"])
        for rows in tasks.values()
    ]
    pairs: dict[str, list[str]] = {}
    for task_id in means:
        pairs.setdefault(physical_geometry_id(task_id.split("__rule_")[0]), []).append(task_id)
    pair_distances = {
        geometry: float(np.linalg.norm(means[ids[0]] - means[ids[1]]))
        for geometry, ids in pairs.items()
        if len(ids) == 2
    }
    return {
        "mean_l2_from_k0": float(np.mean(moves)),
        "mean_posterior_standard_deviation": float(np.mean(np.concatenate(stds))),
        "rule_pair_distances": pair_distances,
        "mean_rule_pair_distance": float(np.mean(list(pair_distances.values()))) if pair_distances else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--full-evaluation", required=True)
    parser.add_argument("--no-context-evaluation", required=True)
    parser.add_argument("--support-audit", required=True)
    parser.add_argument("--training-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    taskbook = load_taskbook(args.taskbook)
    task_ids = {task.task_id for task in taskbook["meta_validation"]}
    full = _read(args.full_evaluation)
    no_context = _read(args.no_context_evaluation)
    support = _read(args.support_audit)
    training = _read(args.training_summary)
    for name, artifact in (("full", full), ("no_context", no_context), ("support", support)):
        if set(artifact["tasks"]) != task_ids:
            raise ValueError(f"{name} diagnostic does not cover the frozen validation task set")
    if full["provenance"].get("checkpoint_hash") != no_context["provenance"].get("checkpoint_hash"):
        raise ValueError("full and no-context diagnostics must use the same checkpoint")
    if support.get("uses_query_cases") is not False:
        raise ValueError("support diagnostic must not use query cases")

    shots = sorted(int(value) for value in full["tasks"][next(iter(task_ids))])
    effects: dict[str, Any] = {}
    support_signal: dict[str, Any] = {}
    for task_id in sorted(task_ids):
        effects[task_id] = {
            str(shot): {
                "valid_critical_strict_rate_difference": float(
                    full["tasks"][task_id][str(shot)]["summary"]["valid_critical_strict_rate"]
                    - no_context["tasks"][task_id][str(shot)]["summary"]["valid_critical_strict_rate"]
                ),
                "mean_episode_return_difference": float(
                    full["tasks"][task_id][str(shot)]["summary"]["mean_episode_return"]
                    - no_context["tasks"][task_id][str(shot)]["summary"]["mean_episode_return"]
                ),
            }
            for shot in shots
        }
        records = support["tasks"][task_id][str(max(shots))]["support_episode_records"]
        support_signal[task_id] = _records_summary(records)

    output = {
        "schema": "posterior_adaptation_medium_diagnostic_v1",
        "run_kind": "medium_diagnostic",
        "not_formal_adaptation_evidence": True,
        "training": training,
        "checkpoint_hash": full["provenance"].get("checkpoint_hash"),
        "training_seed": full["provenance"].get("training_seed"),
        "query_cases_per_task": sorted({
            int(rows[str(shots[0])]["summary"]["episodes"])
            for rows in full["tasks"].values()
        }),
        "support_uses_query_cases": False,
        "full_minus_no_context_by_task": effects,
        "support_signal_at_max_k": support_signal,
        "support_signal_overall": _records_summary([
            record
            for rows in support["tasks"].values()
            for record in rows[str(max(shots))]["support_episode_records"]
        ]),
        "posterior_by_k": {
            str(shot): _posterior_summary(support["tasks"], shot)
            for shot in shots
        },
        "artifact_hashes": {
            "full_evaluation": content_hash(full),
            "no_context_evaluation": content_hash(no_context),
            "support_audit": content_hash(support),
            "training_summary": content_hash(training),
        },
    }
    write_json(args.output, output)
    print(f"medium diagnostic summary: {args.output}")


if __name__ == "__main__":
    main()
