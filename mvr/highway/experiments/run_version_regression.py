"""Chronological, reveal-only replay for controller version regression testing."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mvr.highway.config import VersionRegressionConfig
from mvr.highway.data.version_bank import VersionedBank
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.outcomes import formal_rewards
from mvr.highway.diva.version_mining import TargetReplay, sequential_indices, trace_counts
from mvr.highway.diva.version_reference import build_version_reference, classify_regressions


RANDOM_ELIGIBLE = "Random-Eligible"
PREVIOUS_BOUNDARY = "Previous-Boundary"
FROZEN_HISTORY = "Frozen-History"
SEQUENTIAL_HISTORY = "Sequential-History"
CSV_FIELDS = (
    "lineage_id", "previous_version", "target_version", "history_versions",
    "method", "repeat", "budget_limit", "budget_used", "eligible_count",
    "regression_count", "new_in_archive_count", "reintroduced_count",
    "available_regressions", "regression_recall", "raw_critical_score",
    "stop_reason", "protocol_hash", "bank_hash", "code_commit",
)


def _fixed_indices(reference, method: str, rng: np.random.Generator, budget: int) -> list[int]:
    eligible = np.flatnonzero(reference.previous_safe)
    if method == RANDOM_ELIGIBLE:
        return rng.permutation(eligible)[:budget].tolist()
    if method == PREVIOUS_BOUNDARY:
        order = np.lexsort((eligible, -reference.previous_vulnerability[eligible]))
        return eligible[order][:budget].tolist()
    raise ValueError(f"unsupported fixed method: {method}")


def _row(common: dict, method: str, repeat: int, indices: list[int], labels, collisions, near_misses, protocol: dict) -> dict:
    counts = trace_counts(indices, labels)
    counts.pop("regression_curve")
    available = int(labels.regression.sum())
    selected = np.asarray(indices, dtype=int)
    rewards = formal_rewards(collisions[selected], near_misses[selected])
    return {
        **common,
        "method": method,
        "repeat": repeat,
        "budget_used": len(indices),
        **counts,
        "available_regressions": available,
        "regression_recall": counts["regression_count"] / available if available else None,
        "raw_critical_score": float(rewards.sum()),
        "stop_reason": (
            "no_eligible_previous_safe_cases"
            if len(indices) < common["budget_limit"]
            else (
                "no_regression_observed_within_budget"
                if counts["regression_count"] == 0
                else "budget_exhausted"
            )
        ),
        "protocol_hash": protocol["protocol_hash"],
        "bank_hash": protocol["bank_hash"],
        "code_commit": protocol["code_commit"],
    }


def _traces(row: dict, indices: list[int], scores: list[np.ndarray], outcomes, labels) -> list[dict]:
    payload = []
    cumulative = 0
    for step, index in enumerate(indices, start=1):
        regression = bool(labels.regression[index])
        cumulative += regression
        outcome = outcomes[step - 1]
        payload.append({
            "lineage_id": row["lineage_id"],
            "target_version": row["target_version"],
            "method": row["method"],
            "repeat": row["repeat"],
            "step": step,
            "anchor_index": index,
            "selection_probability": float(scores[step - 1][index]) if scores else None,
            "vulnerability": outcome.vulnerability,
            "collision": outcome.collision,
            "near_miss": outcome.near_miss,
            "regression": regression,
            "cumulative_regressions": cumulative,
        })
    return payload


def run_version_regression(bank: VersionedBank, config: VersionRegressionConfig, protocol: dict) -> tuple[list[dict], list[dict]]:
    """Replay every planned transition using strictly earlier same-lineage sources."""
    config.validate()
    rows: list[dict] = []
    traces: list[dict] = []
    response = bank.responses
    transition_id = 0
    for lineage_id in sorted({version.lineage_id for version in bank.versions}):
        source_rows, versions = bank.lineage_rows(lineage_id)
        for target_order in config.target_orders:
            if target_order > len(versions):
                continue
            target_position = target_order - 1
            history_rows = source_rows[:target_position]
            target_row = source_rows[target_position]
            previous_row = source_rows[target_position - 1]
            if len(history_rows) < 3:
                raise ValueError("every target requires at least three historical versions")
            prior = LowRankPrior.fit(response.vulnerability[history_rows], config.prior_rank)
            reference = build_version_reference(
                response.collisions[history_rows], response.near_misses[history_rows],
                response.vulnerability[previous_row], response.collisions[previous_row],
                response.near_misses[previous_row], bank.valid_mask[previous_row],
            )
            labels = classify_regressions(
                reference, response.collisions[target_row], response.near_misses[target_row],
                bank.valid_mask[target_row],
            )
            eligible_count = int(reference.previous_safe.sum())
            budget = min(config.total_budget, eligible_count)
            common = {
                "lineage_id": lineage_id,
                "previous_version": versions[target_position - 1].version_id,
                "target_version": versions[target_position].version_id,
                "history_versions": ";".join(v.version_id for v in versions[:target_position]),
                "budget_limit": config.total_budget,
                "eligible_count": eligible_count,
            }
            target_arrays = (
                response.vulnerability[target_row], response.collisions[target_row],
                response.near_misses[target_row], bank.valid_mask[target_row], budget,
            )
            for method in (RANDOM_ELIGIBLE, PREVIOUS_BOUNDARY):
                repeats = config.random_repeats if method == RANDOM_ELIGIBLE else 1
                for repeat in range(repeats):
                    rng = np.random.default_rng(config.evaluation_seed + 1000 * transition_id + repeat)
                    indices = _fixed_indices(reference, method, rng, budget)
                    replay = TargetReplay(*target_arrays)
                    outcomes = [replay.reveal(index) for index in indices]
                    row = _row(common, method, repeat, indices, labels, response.collisions[target_row], response.near_misses[target_row], protocol)
                    rows.append(row)
                    traces.extend(_traces(row, indices, [], outcomes, labels))
            for method, update in ((FROZEN_HISTORY, False), (SEQUENTIAL_HISTORY, True)):
                indices, outcomes, scores = sequential_indices(
                    prior, reference, TargetReplay(*target_arrays), budget,
                    config.critical_threshold, config.observation_noise, update,
                )
                row = _row(common, method, 0, indices, labels, response.collisions[target_row], response.near_misses[target_row], protocol)
                rows.append(row)
                traces.extend(_traces(row, indices, scores, outcomes, labels))
            transition_id += 1
    return rows, traces


def summarize_version_regression(rows: list[dict]) -> dict:
    """Apply the frozen development decision without claiming significance."""
    task_keys = sorted({(row["lineage_id"], row["target_version"]) for row in rows})
    per_task = {}
    nonzero_tasks = []
    for task in task_keys:
        task_rows = [row for row in rows if (row["lineage_id"], row["target_version"]) == task]
        if task_rows[0]["available_regressions"]:
            per_task["/".join(task)] = {
                method: float(np.mean([row["regression_recall"] for row in task_rows if row["method"] == method]))
                for method in (RANDOM_ELIGIBLE, PREVIOUS_BOUNDARY, FROZEN_HISTORY, SEQUENTIAL_HISTORY)
            }
            nonzero_tasks.append(task)
    lineages = sorted({row["lineage_id"] for row in rows})
    if not nonzero_tasks or not all(any(task[0] == lineage for task in nonzero_tasks) for lineage in lineages):
        status = "insufficient_regression_evidence"
        means = {}
    else:
        means = {
            method: float(np.mean([per_task["/".join(task)][method] for task in nonzero_tasks]))
            for method in (PREVIOUS_BOUNDARY, FROZEN_HISTORY, SEQUENTIAL_HISTORY)
        }
        tolerance = 1e-12
        if means[SEQUENTIAL_HISTORY] <= means[FROZEN_HISTORY] + tolerance:
            status = "no_adaptation_gain"
        elif means[SEQUENTIAL_HISTORY] <= means[PREVIOUS_BOUNDARY] + tolerance:
            status = "no_advantage_over_previous"
        else:
            gains = []
            for lineage in lineages:
                tasks = [task for task in nonzero_tasks if task[0] == lineage]
                gains.append(all(np.mean([
                    per_task["/".join(task)][SEQUENTIAL_HISTORY] - per_task["/".join(task)][baseline]
                    for task in tasks
                ]) >= -tolerance for baseline in (PREVIOUS_BOUNDARY, FROZEN_HISTORY)))
            status = "provisional_support" if all(gains) else "lineage_limited_support"
    return {
        "schema": "pr_brvt_version_regression_v1",
        "status": status,
        "transitions_with_regressions": ["/".join(task) for task in nonzero_tasks],
        "per_transition_recall": per_task,
        "mean_recall": means,
        "row_count": len(rows),
    }


def write_replay_artifacts(rows: list[dict], traces: list[dict], output: Path) -> None:
    """Write the protocol's table, trace log, breakdown, and one curve."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_transition.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "query_traces.jsonl").open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace) + "\n")
    fields = ("lineage_id", "target_version", "method", "repeat", "regression_count", "new_in_archive_count", "reintroduced_count")
    with (output / "outcome_breakdown.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for trace in traces:
        grouped[trace["method"]].append((trace["step"], trace["cumulative_regressions"]))
    figure, axis = plt.subplots(figsize=(7, 4))
    for method, values in grouped.items():
        per_step: dict[int, list[int]] = defaultdict(list)
        for step, total in values:
            per_step[step].append(total)
        steps = sorted(per_step)
        axis.plot(steps, [np.mean(per_step[step]) for step in steps], label=method)
    axis.set(xlabel="Target test budget", ylabel="Cumulative regressions")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "regression_count_curve.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
