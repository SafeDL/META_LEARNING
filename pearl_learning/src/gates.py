"""Hard gates that prevent formal PEARL runs on un-audited taskbooks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


GATE_SCHEMA = "logical_merge_formal_gate"
REQUIRED_BASELINES = {
    "per_task_sac", "cross_task_policy_matrix", "topology_conditioned_pooled_sac", "scratch_sac",
    "pooled_finetune_sac", "oracle_task_conditioned_sac", "pearl_no_context",
}


def verify_formal_gate(path: str | Path | None, taskbook_hash: str) -> None:
    if not path:
        raise RuntimeError("formal PEARL training requires --gate-manifest after topology and baseline validation")
    payload: Mapping[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != GATE_SCHEMA or payload.get("taskbook_hash") != taskbook_hash:
        raise RuntimeError("formal gate is incompatible with the frozen taskbook")
    if payload.get("topology_audit") != "pass" or payload.get("integrity_audit") != "pass":
        raise RuntimeError("formal gate lacks a passing all-task topology audit")
    complete = set(payload.get("completed_baselines", []))
    missing = REQUIRED_BASELINES - complete
    if missing:
        raise RuntimeError(f"formal gate lacks required baseline evidence: {sorted(missing)}")
