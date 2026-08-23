"""Offline Gate-3 re-judgement under the v5 Actor-first causal contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from archives.pearl_learning.scripts.audit_gate3_vanilla_pearl_mechanism import (
    gate3_causal_chain_verdict_v5,
)
from archives.pearl_learning.src.io import content_hash, write_json


def recompute(
    suite_path: Path,
    oracle: dict[str, Any] | None,
    output_dir: Path | None,
) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("schema") != "gate3_vanilla_pearl_mechanism_causal_audit_suite_v1":
        raise ValueError(f"{suite_path} is not a Gate 3 causal audit suite")
    gate = gate3_causal_chain_verdict_v5(suite, oracle)
    gate["inputs"] = suite.get("provenance", {})
    gate["recomputed_from"] = {
        "audit_suite_path": str(suite_path.resolve()),
        "audit_suite_hash": content_hash(suite),
        "oracle_audit_hash": None if oracle is None else content_hash(oracle),
        "training_updates": 0,
        "environment_steps": 0,
    }
    target = (
        output_dir if output_dir is not None else suite_path.parent
    ) / "gate3_causal_chain_gate_v5.json"
    write_json(target, gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="append", required=True)
    parser.add_argument("--oracle-audit")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.output_dir and len(args.suite) != 1:
        raise ValueError("--output-dir requires exactly one --suite")
    oracle = (
        None
        if not args.oracle_audit
        else json.loads(Path(args.oracle_audit).read_text(encoding="utf-8"))
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    for raw in args.suite:
        gate = recompute(Path(raw), oracle, output_dir)
        stages = gate["stages"]
        print(
            f"{raw}: Stage A {stages['stage_a_context_to_posterior']['status']}, "
            f"B_Q {stages['stage_b_q_diagnostic_only']['observed_threshold_status']} "
            "(diagnostic only), "
            f"Stage B_pi {stages['stage_b_pi_posterior_to_actor_action']['status']}, "
            f"Stage C {stages['stage_c_actor_to_outcome']['status']}, "
            f"overall {gate['status']}"
        )


if __name__ == "__main__":
    main()
