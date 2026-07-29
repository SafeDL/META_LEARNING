"""Runtime transfer/defer decisions from validation-only calibration evidence."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .transferability import TRANSFERABILITY_REPORT_SCHEMA
from .transferability_calibration import CALIBRATION_SCHEMA, posterior_uncertainty_by_task


DECISION_SCHEMA = "logical_merge_transferability_decision_v1"
DEFAULT_FALLBACK = "collect_representative_support_or_run_budgeted_online_sac"


def transferability_decision_report(
    descriptor_report: Mapping[str, Any], calibration: Mapping[str, Any], *, fallback: str = DEFAULT_FALLBACK,
    minimum_leave_one_out_coverage: float = 0.5, minimum_leave_one_out_tasks: int = 2,
    posterior_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return per-task meta-adapt/defer decisions without observing query data.

    A missing or insufficient validation calibration is deliberately converted
    to ``defer`` for every task.  This is a conservative deployment action,
    not evidence that the task is intrinsically untransferable.
    """
    if descriptor_report.get("schema") != TRANSFERABILITY_REPORT_SCHEMA:
        raise ValueError("unsupported descriptor report schema")
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported calibration report schema")
    if descriptor_report.get("taskbook_hash") != calibration.get("taskbook_hash"):
        raise ValueError("descriptor and calibration taskbooks differ")
    if calibration.get("split") != "meta_validation":
        raise ValueError("runtime decisions require validation-only calibration")
    if descriptor_report.get("uses_hidden_rules"):
        raise ValueError("runtime decisions cannot use hidden-rule descriptors")
    if not 0.0 <= float(minimum_leave_one_out_coverage) <= 1.0:
        raise ValueError("minimum leave-one-out coverage must lie in [0, 1]")
    if int(minimum_leave_one_out_tasks) < 1:
        raise ValueError("minimum leave-one-out task count must be positive")
    candidates = list(descriptor_report.get("candidates", []))
    status = str(calibration.get("status"))
    uncertainty_threshold = calibration.get("uncertainty_threshold")
    if uncertainty_threshold is not None and float(uncertainty_threshold) < 0.0:
        raise ValueError("calibrated posterior uncertainty threshold must be non-negative")
    common = {
        "schema": DECISION_SCHEMA,
        "taskbook_hash": descriptor_report["taskbook_hash"],
        "candidate_split": descriptor_report.get("candidate_split"),
        "calibration_split": "meta_validation",
        "uses_query_cases_for_decision": False,
        "uses_hidden_rules": False,
        "fallback": str(fallback),
        "uncertainty_threshold": uncertainty_threshold,
        "uncertainty_note": "A posterior-uncertainty threshold is used only when validation calibration supplied one.",
        "minimum_leave_one_out_coverage": float(minimum_leave_one_out_coverage),
        "minimum_leave_one_out_tasks": int(minimum_leave_one_out_tasks),
    }
    if status == "insufficient_validation_evidence_no_threshold":
        reasons = [str(reason) for reason in calibration.get("reasons", [])]
        decisions = [{
            "task_id": str(candidate["task_id"]),
            "decision": "defer",
            "allow_meta_adaptation": False,
            "reason": ["no_validated_transfer_threshold", *reasons],
            "nearest_meta_train": dict(candidate["nearest_meta_train"]),
            "coverage_flags": dict(candidate["coverage_flags"]),
            "fallback": str(fallback),
        } for candidate in candidates]
        return common | {
            "status": "deferred_insufficient_validation_evidence",
            "threshold": None,
            "decisions": decisions,
        }
    if status != "calibrated_validation_only":
        raise ValueError(f"unsupported calibration status: {status!r}")
    leave_one_out = calibration.get("leave_one_task_out")
    if not isinstance(leave_one_out, Mapping):
        leave_one_out = {}
    coverage = float(leave_one_out.get("coverage", 0.0))
    evaluable = int(leave_one_out.get("evaluable_task_count", 0))
    if coverage < float(minimum_leave_one_out_coverage) or evaluable < int(minimum_leave_one_out_tasks):
        decisions = [{
            "task_id": str(candidate["task_id"]),
            "decision": "defer",
            "allow_meta_adaptation": False,
            "reason": ["insufficient_leave_one_task_out_validation"],
            "nearest_meta_train": dict(candidate["nearest_meta_train"]),
            "coverage_flags": dict(candidate["coverage_flags"]),
            "fallback": str(fallback),
        } for candidate in candidates]
        return common | {
            "status": "deferred_insufficient_out_of_sample_validation",
            "threshold": None,
            "leave_one_task_out": dict(leave_one_out),
            "decisions": decisions,
        }
    threshold = float(calibration.get("threshold"))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("calibrated similarity threshold must lie in [0, 1]")
    uncertainties: dict[str, float] | None = None
    if uncertainty_threshold is not None:
        if posterior_audit is None:
            decisions = [{
                "task_id": str(candidate["task_id"]), "decision": "defer", "allow_meta_adaptation": False,
                "reason": ["missing_support_only_posterior_uncertainty"],
                "nearest_meta_train": dict(candidate["nearest_meta_train"]),
                "coverage_flags": dict(candidate["coverage_flags"]), "fallback": str(fallback),
            } for candidate in candidates]
            return common | {"status": "deferred_missing_posterior_uncertainty", "threshold": threshold, "decisions": decisions}
        uncertainties = posterior_uncertainty_by_task(
            posterior_audit, taskbook_hash=str(descriptor_report["taskbook_hash"]),
            split=str(descriptor_report.get("candidate_split")), shot=int(calibration["shot"]),
        )
        if set(uncertainties) != {str(candidate["task_id"]) for candidate in candidates}:
            raise ValueError("posterior audit and decision candidates differ")
    decisions = []
    for candidate in candidates:
        nearest = dict(candidate["nearest_meta_train"])
        similarity = float(nearest["similarity"])
        variance = None if uncertainties is None else float(uncertainties[str(candidate["task_id"])])
        accepted = similarity >= threshold and (variance is None or variance <= float(uncertainty_threshold))
        if similarity < threshold:
            reason = ["similarity_below_validation_threshold"]
        elif variance is not None and variance > float(uncertainty_threshold):
            reason = ["posterior_uncertainty_above_validation_threshold"]
        else:
            reason = ["similarity_and_posterior_uncertainty_meet_validation_thresholds"]
        decisions.append({
            "task_id": str(candidate["task_id"]),
            "decision": "meta_adapt" if accepted else "defer",
            "allow_meta_adaptation": accepted,
            "reason": reason,
            "nearest_meta_train": nearest,
            "coverage_flags": dict(candidate["coverage_flags"]),
            "posterior_variance_mean": variance,
            "fallback": None if accepted else str(fallback),
        })
    return common | {
        "status": "validation_calibrated_similarity_and_uncertainty_decision" if uncertainty_threshold is not None else "validation_calibrated_similarity_decision",
        "threshold": threshold,
        "leave_one_task_out": dict(leave_one_out),
        "decisions": decisions,
    }
