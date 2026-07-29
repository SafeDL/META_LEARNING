"""Validation-only calibration for descriptor-space transferability scores."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .io import content_hash
from .transferability import TRANSFERABILITY_REPORT_SCHEMA


CALIBRATION_SCHEMA = "logical_merge_transferability_calibration_v1"
POSTERIOR_AUDIT_SCHEMA = "logical_merge_support_posterior_diagnostic_v1"


def posterior_uncertainty_by_task(posterior_audit: Mapping[str, Any], *, taskbook_hash: str,
                                  split: str, shot: int) -> dict[str, float]:
    """Read a support-only posterior-variance proxy for a frozen split."""
    if posterior_audit.get("schema") != POSTERIOR_AUDIT_SCHEMA:
        raise ValueError("unsupported posterior audit schema")
    if posterior_audit.get("taskbook_hash") != taskbook_hash or posterior_audit.get("split") != split:
        raise ValueError("posterior audit belongs to a different taskbook or split")
    if bool(posterior_audit.get("uses_query_cases", True)) or not bool(posterior_audit.get("no_gradient_adaptation", False)):
        raise ValueError("posterior uncertainty input must be query-free and no-gradient")
    result: dict[str, float] = {}
    for task_id, shots in dict(posterior_audit.get("tasks", {})).items():
        try:
            variance = np.asarray(shots[str(shot)]["posterior_variance"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"posterior audit lacks finite K={shot} variance for {task_id}") from exc
        if variance.size == 0 or not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError(f"posterior audit has invalid K={shot} variance for {task_id}")
        result[str(task_id)] = float(variance.mean())
    return result


def _task_rows(descriptor_report: Mapping[str, Any], taskwise_summary: Mapping[str, Any], shot: int,
               posterior_audit: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if descriptor_report.get("schema") != TRANSFERABILITY_REPORT_SCHEMA:
        raise ValueError("unsupported descriptor report schema")
    if taskwise_summary.get("schema") != "pearl_equal_new_task_budget":
        raise ValueError("unsupported equal-budget summary schema")
    if descriptor_report.get("taskbook_hash") != taskwise_summary.get("taskbook_hash"):
        raise ValueError("descriptor and outcome reports belong to different taskbooks")
    if descriptor_report.get("candidate_split") != taskwise_summary.get("split"):
        raise ValueError("descriptor and outcome reports use different candidate splits")
    if descriptor_report.get("uses_hidden_rules"):
        raise ValueError("deployment calibration cannot use hidden-rule descriptors")
    try:
        outcome_tasks = taskwise_summary["budgets"][str(shot)]["tasks"]
    except KeyError as exc:
        raise ValueError(f"taskwise summary lacks K={shot}") from exc
    descriptors = {str(row["task_id"]): row for row in descriptor_report.get("candidates", [])}
    if set(descriptors) != set(outcome_tasks):
        raise ValueError("descriptor and outcome task IDs differ")
    uncertainties = None if posterior_audit is None else posterior_uncertainty_by_task(
        posterior_audit, taskbook_hash=str(descriptor_report["taskbook_hash"]),
        split=str(descriptor_report["candidate_split"]), shot=shot,
    )
    if uncertainties is not None and set(uncertainties) != set(outcome_tasks):
        raise ValueError("posterior audit and outcome task IDs differ")
    rows = []
    for task_id in sorted(outcome_tasks):
        outcome = outcome_tasks[task_id]
        nearest = descriptors[task_id]["nearest_meta_train"]
        baseline = max(float(outcome["scratch_sac_mean"]), float(outcome["pooled_finetune_sac_mean"]))
        gain = float(outcome["pearl_valid_critical_strict_rate"]) - baseline
        rows.append({
            "task_id": task_id,
            "similarity": float(nearest["similarity"]),
            "descriptor_distance": float(nearest["distance"]["total"]),
            "unseen_logical_type": bool(descriptors[task_id]["coverage_flags"]["unseen_logical_type"]),
            "unseen_map_kind": bool(descriptors[task_id]["coverage_flags"]["unseen_map_kind"]),
            "support_environment_steps": int(outcome["support_environment_steps"]),
            "pearl_valid_critical_strict_rate": float(outcome["pearl_valid_critical_strict_rate"]),
            "best_online_baseline": baseline,
            "adaptation_gain": gain,
            "beneficial": bool(gain > 0.0),
            "posterior_variance_mean": None if uncertainties is None else float(uncertainties[task_id]),
        })
    return rows


def _classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    positive, negative = int(truth.sum()), int((~truth).sum())
    tpr = float(np.mean(prediction[truth])) if positive else 0.0
    tnr = float(np.mean(~prediction[~truth])) if negative else 0.0
    return {"balanced_accuracy": (tpr + tnr) / 2.0, "true_positive_rate": tpr, "true_negative_rate": tnr, "acceptance_rate": float(np.mean(prediction))}


def _threshold_metrics(rows: list[Mapping[str, Any]], threshold: float,
                       uncertainty_threshold: float | None = None) -> dict[str, float]:
    truth = np.asarray([bool(row["beneficial"]) for row in rows], dtype=bool)
    prediction = np.asarray([float(row["similarity"]) >= threshold for row in rows], dtype=bool)
    if uncertainty_threshold is not None:
        uncertainty = np.asarray([float(row["posterior_variance_mean"]) for row in rows], dtype=float)
        prediction &= uncertainty <= uncertainty_threshold
    return _classification_metrics(truth, prediction)


def _threshold_candidates(rows: list[Mapping[str, Any]]) -> list[dict[str, float]]:
    similarities = sorted({0.0, 1.0, *(float(row["similarity"]) for row in rows)})
    has_uncertainty = all(row.get("posterior_variance_mean") is not None for row in rows)
    if not has_uncertainty:
        return [{"threshold": threshold, **_threshold_metrics(rows, threshold)} for threshold in similarities]
    values = sorted({float(row["posterior_variance_mean"]) for row in rows})
    uncertainty_thresholds = [-1.0, *values, float(max(values) + 1.0)]
    return [
        {"threshold": threshold, "uncertainty_threshold": uncertainty, **_threshold_metrics(rows, threshold, uncertainty)}
        for threshold in similarities for uncertainty in uncertainty_thresholds
    ]


def _select_threshold(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    scored = _threshold_candidates(rows)
    return max(scored, key=lambda row: (
        row["balanced_accuracy"], row["true_positive_rate"], -row["acceptance_rate"],
        -float(row.get("uncertainty_threshold", float("inf"))), -row["threshold"],
    ))


def _leave_one_task_out(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Estimate threshold stability without letting a task choose its own threshold."""
    predictions: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, held_out in enumerate(rows):
        training = rows[:index] + rows[index + 1:]
        if len({bool(row["beneficial"]) for row in training}) < 2:
            skipped.append(str(held_out["task_id"]))
            continue
        selected = _select_threshold(training)
        accepted = float(held_out["similarity"]) >= float(selected["threshold"])
        if selected.get("uncertainty_threshold") is not None:
            accepted = accepted and float(held_out["posterior_variance_mean"]) <= float(selected["uncertainty_threshold"])
        predictions.append({
            "task_id": str(held_out["task_id"]),
            "threshold": float(selected["threshold"]),
            "uncertainty_threshold": selected.get("uncertainty_threshold"),
            "beneficial": bool(held_out["beneficial"]),
            "accepted": accepted,
            "correct": bool(accepted == bool(held_out["beneficial"])),
        })
    truth = np.asarray([row["beneficial"] for row in predictions], dtype=bool)
    accepted = np.asarray([row["accepted"] for row in predictions], dtype=bool)
    has_both_classes = bool(len(set(truth.tolist())) == 2)
    return {
        "strategy": "leave_one_task_out_threshold_selection",
        "evaluable_task_count": len(predictions),
        "skipped_task_ids": skipped,
        "coverage": float(len(predictions) / len(rows)) if rows else 0.0,
        "accuracy": float(np.mean(accepted == truth)) if len(predictions) else None,
        "balanced_accuracy": float(_classification_metrics(truth, accepted)["balanced_accuracy"]) if len(predictions) and has_both_classes else None,
        "predictions": predictions,
    }


def calibration_report(descriptor_report: Mapping[str, Any], taskwise_summary: Mapping[str, Any], *, shot: int,
                       minimum_independent_tasks: int = 8, posterior_audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Calibrate a similarity acceptance threshold only when validation is sufficient.

    A task is the independent unit.  Multiple K values or random seeds from the
    same task must never be counted as additional calibration tasks.
    """
    if minimum_independent_tasks < 2:
        raise ValueError("minimum_independent_tasks must be at least two")
    rows = _task_rows(descriptor_report, taskwise_summary, shot, posterior_audit)
    labels = {bool(row["beneficial"]) for row in rows}
    sufficient = len(rows) >= minimum_independent_tasks and len(labels) == 2
    payload: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "taskbook_hash": descriptor_report["taskbook_hash"],
        "split": descriptor_report["candidate_split"],
        "shot": int(shot),
        "minimum_independent_tasks": int(minimum_independent_tasks),
        "independent_task_count": len(rows),
        "beneficial_task_count": int(sum(bool(row["beneficial"]) for row in rows)),
        "nonbeneficial_task_count": int(sum(not bool(row["beneficial"]) for row in rows)),
        "rows": rows,
        "uses_query_for_calibration": True,
        "calibration_scope": "validation only; never fit this report on meta_test splits",
        "posterior_uncertainty_input": "support_only_posterior_variance" if posterior_audit is not None else None,
        "leave_one_task_out": _leave_one_task_out(rows),
    }
    if not sufficient:
        reasons = []
        if len(rows) < minimum_independent_tasks:
            reasons.append("too_few_independent_validation_tasks")
        if len(labels) < 2:
            reasons.append("validation_labels_have_only_one_class")
        return payload | {
            "status": "insufficient_validation_evidence_no_threshold",
            "reasons": reasons,
            "threshold": None,
            "uncertainty_threshold": None,
        }
    scored = _threshold_candidates(rows)
    best = _select_threshold(rows)
    return payload | {
        "status": "calibrated_validation_only",
        "threshold": float(best["threshold"]),
        "uncertainty_threshold": None if best.get("uncertainty_threshold") is None else float(best["uncertainty_threshold"]),
        "selection_rule": "accept PEARL adaptation when descriptor similarity >= threshold and, when calibrated, posterior variance <= uncertainty_threshold; separately report OOD coverage flags",
        "threshold_candidates": scored,
        "report_hash": content_hash({"rows": rows, "shot": int(shot), "minimum_independent_tasks": int(minimum_independent_tasks), "uses_posterior_audit": posterior_audit is not None}),
    }
