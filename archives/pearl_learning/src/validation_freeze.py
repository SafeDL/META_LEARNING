"""Validation-to-holdout protocol freeze for formal PEARL comparisons."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .io import content_hash


FREEZE_SCHEMA = "logical_merge_validation_freeze_v1"
_CALIBRATION_STATUSES = {
    "calibrated_validation_only",
    "insufficient_validation_evidence_no_threshold",
}


def _validation_evaluation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Select the deterministic posterior-mean view from the maintained suite."""
    if payload.get("schema") != "pearl_fewshot_evaluation_suite":
        return payload
    regimes = payload.get("evaluation_regimes")
    if not isinstance(regimes, Mapping):
        raise ValueError("validation evaluation suite lacks evaluation regimes")
    validation = regimes.get("validation_known_logical_type")
    if not isinstance(validation, Mapping) or validation.get("split") != "meta_validation":
        raise ValueError("validation evaluation suite lacks the validation regime")
    modes = validation.get("query_modes")
    if not isinstance(modes, Mapping):
        raise ValueError("validation evaluation suite lacks query execution modes")
    selected = modes.get("posterior_mean_deterministic")
    if not isinstance(selected, Mapping):
        raise ValueError("validation freeze requires posterior_mean_deterministic metrics")
    return selected


def _provenance_hash(payload: Mapping[str, Any], taskbook_hash: str) -> str:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("taskbook_hash") != taskbook_hash:
        raise ValueError("validation artifact has an incompatible taskbook provenance")
    checkpoint_hash = str(provenance.get("checkpoint_hash", ""))
    if not checkpoint_hash:
        raise ValueError("validation artifact lacks a checkpoint hash")
    return checkpoint_hash


def freeze_validation_protocol(*, taskbook_hash: str, evaluations: Mapping[str, Mapping[str, Any]],
                               required_policies: list[str],
                               equal_budget: Mapping[str, Mapping[str, Any]] | None = None,
                               representation_audits: list[Mapping[str, Any]] | None = None,
                               calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Freeze validation-only method choices before any holdout evaluation.

    The manifest is deliberately evidence-oriented: it does not inspect test
    results and does not decide that a method is effective.  It merely binds a
    checkpoint, taskbook, support-selection policies, and validation artifacts
    so that later holdout runs cannot silently swap any of them.
    """
    expected = list(required_policies)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("required support policies must be a non-empty unique list")
    if set(evaluations) != set(expected):
        raise ValueError("validation evaluations do not match required support policies")
    checkpoint_hashes: set[str] = set()
    evaluation_hashes: dict[str, str] = {}
    for policy in expected:
        payload = _validation_evaluation(evaluations[policy])
        if payload.get("split") != "meta_validation":
            raise ValueError(f"{policy} evaluation is not validation-only")
        if payload.get("support_selection") != policy:
            raise ValueError(f"{policy} evaluation has a mismatched support-selection policy")
        if not bool(payload.get("no_gradient_adaptation", False)):
            raise ValueError(f"{policy} evaluation lacks the no-gradient invariant")
        if payload.get("parameter_hash_before") != payload.get("parameter_hash_after"):
            raise ValueError(f"{policy} evaluation changed PEARL parameters")
        checkpoint_hashes.add(_provenance_hash(payload, taskbook_hash))
        evaluation_hashes[policy] = content_hash(payload)
    if len(checkpoint_hashes) != 1:
        raise ValueError("support-policy validation evaluations must use one frozen checkpoint")
    budget_hashes: dict[str, str] = {}
    if equal_budget is not None:
        if set(equal_budget) != set(expected):
            raise ValueError("equal-budget reports do not match required support policies")
        for policy, payload in equal_budget.items():
            if payload.get("schema") != "pearl_equal_new_task_budget" or payload.get("taskbook_hash") != taskbook_hash:
                raise ValueError(f"{policy} equal-budget report has an incompatible taskbook")
            if payload.get("split") != "meta_validation" or payload.get("protocol", {}).get("support_selection") != policy:
                raise ValueError(f"{policy} equal-budget report is not the declared validation protocol")
            budget_hashes[policy] = content_hash(payload)
    representation_hashes: list[str] = []
    for payload in representation_audits or []:
        if payload.get("schema") != "logical_merge_task_representation_audit_v1" or payload.get("split") != "meta_validation":
            raise ValueError("representation audit is not validation-only")
        if bool(payload.get("uses_query_cases", True)) or payload.get("parameter_hash_before") != payload.get("parameter_hash_after"):
            raise ValueError("representation audit violates support-only or parameter invariants")
        provenance = payload.get("provenance", {})
        if not isinstance(provenance, Mapping) or provenance.get("taskbook_hash") != taskbook_hash:
            raise ValueError("representation audit has incompatible taskbook provenance")
        representation_hashes.append(content_hash(payload))
    calibration_hash = None
    if calibration is not None:
        if calibration.get("schema") != "logical_merge_transferability_calibration_v1" or calibration.get("taskbook_hash") != taskbook_hash:
            raise ValueError("calibration has an incompatible taskbook")
        if calibration.get("split") != "meta_validation" or calibration.get("status") not in _CALIBRATION_STATUSES:
            raise ValueError("calibration is not a recognized validation-only result")
        calibration_hash = content_hash(calibration)
    return {
        "schema": FREEZE_SCHEMA,
        "status": "validation_frozen_for_holdout",
        "taskbook_hash": taskbook_hash,
        "checkpoint_hash": checkpoint_hashes.pop(),
        "required_support_policies": expected,
        "validation_evaluation_hashes": evaluation_hashes,
        "validation_equal_budget_hashes": budget_hashes,
        "validation_representation_audit_hashes": representation_hashes,
        "validation_calibration_hash": calibration_hash,
        "uses_holdout_query_results": False,
        "purpose": "bind validation-only choices before holdout evaluation; not a performance claim",
    }


def verify_validation_freeze(payload: Mapping[str, Any], *, taskbook_hash: str, checkpoint_hash: str) -> None:
    """Require a compatible frozen validation protocol before a holdout run."""
    if payload.get("schema") != FREEZE_SCHEMA or payload.get("status") != "validation_frozen_for_holdout":
        raise ValueError("unsupported validation-freeze manifest")
    if payload.get("taskbook_hash") != taskbook_hash or payload.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("validation-freeze manifest does not bind this taskbook and checkpoint")
    if bool(payload.get("uses_holdout_query_results", True)):
        raise ValueError("validation-freeze manifest must not contain holdout query results")
