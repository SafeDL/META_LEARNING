"""Evidence aggregation for posterior-adaptation validation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .casebook import physical_geometry_id


def task_cluster_interval(
    values_by_task: Mapping[str, Iterable[float]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int | None]:
    groups: dict[str, np.ndarray] = {}
    for task, values in values_by_task.items():
        array = np.asarray(list(values), dtype=float)
        if array.size:
            groups[str(task)] = array
    if not groups:
        return {"task_count": 0, "mean": None, "ci_lower": None, "ci_upper": None}
    tasks = list(groups)
    point = float(np.mean([float(np.mean(groups[task])) for task in tasks]))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(samples), dtype=float)
    for index in range(int(samples)):
        selected_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        task_means = []
        for task in selected_tasks:
            values = groups[str(task)]
            sampled = rng.choice(values, size=len(values), replace=True)
            task_means.append(float(np.mean(sampled)))
        draws[index] = float(np.mean(task_means))
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "task_count": len(tasks),
        "mean": point,
        "ci_lower": float(np.quantile(draws, alpha)),
        "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
    }


def validate_evaluation_artifact(
    payload: Mapping[str, Any],
    *,
    taskbook_hash: str,
    shots: list[int],
    query_cases: int,
) -> list[str]:
    problems: list[str] = []
    provenance = payload.get("provenance", {})
    if payload.get("split") != "meta_validation":
        problems.append("split_is_not_meta_validation")
    if not isinstance(provenance, Mapping) or provenance.get("taskbook_hash") != taskbook_hash:
        problems.append("taskbook_hash_mismatch")
    if payload.get("support_selection") != "fixed":
        problems.append("support_selection_is_not_fixed")
    context = payload.get("context_protocol", {})
    if not isinstance(context, Mapping) or context.get("name") != "fixed_nested":
        problems.append("context_protocol_is_not_fixed_nested")
    if payload.get("parameter_hash_before") != payload.get("parameter_hash_after"):
        problems.append("parameter_hash_changed")
    if payload.get("module_hashes_before") != payload.get("module_hashes_after"):
        problems.append("module_hash_changed")
    if not bool(payload.get("no_gradient_adaptation", False)):
        problems.append("no_gradient_invariant_missing")
    query_counts: set[int] = set()
    for task_id, task_rows in payload.get("tasks", {}).items():
        observed = sorted(int(value) for value in task_rows)
        if observed != shots:
            problems.append(f"{task_id}:shots_mismatch")
        for shot, row in task_rows.items():
            query_counts.add(int(row.get("summary", {}).get("episodes", -1)))
            if int(shot) > 0 and int(row.get("context_transition_count", -1)) != int(shot) * 32:
                problems.append(f"{task_id}:K={shot}:context_transition_count_mismatch")
            hashes = list(row.get("context_episode_sample_hashes", []))
            if len(hashes) != int(shot):
                problems.append(f"{task_id}:K={shot}:context_episode_hash_count_mismatch")
        ordered = [task_rows[str(shot)] for shot in shots]
        for previous, current in zip(ordered, ordered[1:]):
            previous_hashes = list(previous.get("context_episode_sample_hashes", []))
            current_hashes = list(current.get("context_episode_sample_hashes", []))
            if current_hashes[:len(previous_hashes)] != previous_hashes:
                problems.append(f"{task_id}:context_is_not_a_nested_prefix")
                break
        prior = task_rows.get("0", {})
        prior_mu = np.asarray(prior.get("posterior_mean", []), dtype=float)
        prior_log_var = np.asarray(prior.get("posterior_log_variance", []), dtype=float)
        if prior_mu.size == 0 or not np.array_equal(prior_mu, np.zeros_like(prior_mu)):
            problems.append(f"{task_id}:K=0_mean_is_not_exact_prior")
        if prior_log_var.size == 0 or not np.array_equal(prior_log_var, np.zeros_like(prior_log_var)):
            problems.append(f"{task_id}:K=0_log_variance_is_not_exact_prior")
    if query_counts != {int(query_cases)}:
        problems.append(
            f"query_case_count_mismatch:observed={sorted(query_counts)}:expected={int(query_cases)}"
        )
    return problems


def flatten_evaluations(
    evaluations: Mapping[str, list[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, artifacts in evaluations.items():
        for artifact in artifacts:
            seed = int(artifact.get("provenance", {}).get("training_seed", -1))
            for task_id, task_rows in artifact["tasks"].items():
                for shot, value in task_rows.items():
                    rows.append({
                        "method": method,
                        "training_seed": seed,
                        "split": artifact["split"],
                        "task_id": task_id,
                        "K": int(shot),
                        "support_environment_steps": int(value["support_environment_steps"]),
                        "context_sample_hash": value.get("context_sample_hash"),
                        "posterior_mean": value.get("posterior_mean"),
                        "posterior_log_variance": value.get("posterior_log_variance"),
                        "posterior_change": value.get("posterior_change"),
                        **dict(value["summary"]),
                    })
    return rows


def paired_method_effect(
    rows: list[Mapping[str, Any]],
    *,
    method: str,
    reference: str,
    shot: int,
    metric: str,
    samples: int,
    confidence: float,
) -> dict[str, Any]:
    lookup = {
        (str(row["method"]), int(row["training_seed"]), str(row["task_id"]), int(row["K"])): float(row[metric])
        for row in rows
    }
    by_task: dict[str, list[float]] = defaultdict(list)
    seeds: set[int] = set()
    for key, value in lookup.items():
        name, seed, task_id, k = key
        if name != method or k != int(shot):
            continue
        other = lookup.get((reference, seed, task_id, k))
        if other is not None:
            by_task[task_id].append(value - other)
            seeds.add(seed)
    result = task_cluster_interval(
        by_task, samples=samples, confidence=confidence, seed=91027 + int(shot),
    )
    task_means = {task: float(np.mean(values)) for task, values in by_task.items()}
    return {
        "method": method,
        "reference": reference,
        "metric": metric,
        "K": int(shot),
        "training_seed_count": len(seeds),
        **result,
        "task_mean_differences": task_means,
        "negative_effect_tasks": sorted(task for task, value in task_means.items() if value < 0.0),
    }


def _rule_label(task: Any) -> int:
    value = str(task.priority_spec.get("target_contact_entry_order"))
    if value == "adversary_first":
        return 0
    if value == "sut_first":
        return 1
    raise ValueError(f"task {task.task_id} lacks a binary entry-order rule")


def posterior_pair_audit(
    artifacts: list[Mapping[str, Any]],
    tasks: list[Any],
    *,
    shots: list[int],
    samples: int,
    confidence: float,
) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    groups: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        groups[physical_geometry_id(task.geometry_id)].append(task.task_id)
    invalid = {name: ids for name, ids in groups.items() if len(ids) != 2}
    if invalid:
        raise ValueError(f"posterior pair audit requires exactly two tasks per geometry: {invalid}")
    result: dict[str, Any] = {}
    for shot in shots:
        distances_by_pair: dict[str, list[float]] = defaultdict(list)
        accuracy_by_pair: dict[str, list[float]] = defaultdict(list)
        for artifact in artifacts:
            features = {
                task_id: np.asarray(artifact["tasks"][task_id][str(shot)]["posterior_mean"], dtype=float).reshape(-1)
                for task_id in task_by_id
            }
            for group, ids in groups.items():
                left, right = ids
                distances_by_pair[group].append(float(np.linalg.norm(features[left] - features[right])))
                train_ids = [
                    task_id
                    for other_group, other_ids in groups.items()
                    if other_group != group
                    for task_id in other_ids
                ]
                if not train_ids:
                    continue
                centroids = {
                    label: np.mean(
                        [features[task_id] for task_id in train_ids if _rule_label(task_by_id[task_id]) == label],
                        axis=0,
                    )
                    for label in (0, 1)
                }
                correct = []
                for task_id in ids:
                    distances = [float(np.linalg.norm(features[task_id] - centroids[label])) for label in (0, 1)]
                    prediction = int(np.argmin(distances))
                    correct.append(float(prediction == _rule_label(task_by_id[task_id])))
                accuracy_by_pair[group].append(float(np.mean(correct)))
        result[str(shot)] = {
            "mean_pair_distance": task_cluster_interval(
                distances_by_pair,
                samples=samples,
                confidence=confidence,
                seed=32003 + int(shot),
            ),
            "leave_one_geometry_pair_out_rule_accuracy": task_cluster_interval(
                accuracy_by_pair,
                samples=samples,
                confidence=confidence,
                seed=47017 + int(shot),
            ),
            "pair_distances_by_geometry": {
                group: [float(value) for value in values]
                for group, values in distances_by_pair.items()
            },
        }
    return {
        "schema": "posterior_adaptation_pair_audit_v1",
        "split": "meta_validation",
        "classifier": "nearest posterior centroid with leave-one-physical-geometry-pair-out validation",
        "uses_hidden_rules_for_posthoc_audit_only": True,
        "uses_query_cases_for_classifier": False,
        "shots": result,
    }


def budget_curves(rows: list[Mapping[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    result: dict[str, Any] = {}
    for shot in sorted({int(row["K"]) for row in selected}):
        current = [row for row in selected if int(row["K"]) == shot]
        result[str(shot)] = {
            "observations": len(current),
            "mean_support_environment_steps": float(np.mean([row["support_environment_steps"] for row in current])),
            "mean_valid_critical_strict_rate": float(np.mean([row["valid_critical_strict_rate"] for row in current])),
            "mean_query_return": float(np.mean([row["mean_episode_return"] for row in current])),
            "mean_invalid_rate": float(np.mean([row["invalid_rate"] for row in current])),
        }
    return result
