"""Preconditions that prevent formal PEARL runs on un-audited taskbooks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


FORMAL_VALIDATION_SCHEMA = "logical_merge_formal_validation"
REQUIRED_PRETRAIN_BASELINES = {
    "per_task_sac", "cross_task_policy_matrix", "topology_conditioned_pooled_sac", "scratch_sac",
    "pooled_finetune_sac", "oracle_task_conditioned_sac",
}


def verify_formal_validation(path: str | Path | None, taskbook_hash: str) -> None:
    if not path:
        raise RuntimeError("formal PEARL training requires --formal-validation after topology and baseline validation")
    payload: Mapping[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != FORMAL_VALIDATION_SCHEMA or payload.get("taskbook_hash") != taskbook_hash:
        raise RuntimeError("formal validation is incompatible with the frozen taskbook")
    if payload.get("topology_audit") != "pass" or payload.get("integrity_audit") != "pass":
        raise RuntimeError("formal validation lacks a passing all-task topology/integrity audit")
    if payload.get("heterogeneity_audit") != "pass":
        raise RuntimeError("formal validation lacks evidence that pooled SAC underperforms per-task SAC")
    if int(payload.get("baseline_environment_steps", 0)) <= 0:
        raise RuntimeError("formal validation lacks one matched positive environment-step budget for all SAC baselines")
    complete = set(payload.get("completed_baselines", []))
    missing = REQUIRED_PRETRAIN_BASELINES - complete
    if missing:
        raise RuntimeError(f"formal validation lacks required baseline evidence: {sorted(missing)}")
