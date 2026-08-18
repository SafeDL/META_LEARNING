"""Offline re-judgement of existing Gate 3 audit suites under the v3 Stage-A rule.

Round 1 and Round 2 were judged before Stage A required the prior-relative
separation ratio.  Their audit suites already contain every latent, action and
VCSR number the verdict needs, so the v3 verdict is recomputed from those JSONs
without retraining, without new rollouts, and without touching the v2 gate
files.  The recomputed verdict is written to ``gate3_causal_chain_gate_v3.json``
next to each source suite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pearl_learning.src.io import content_hash, write_json

from pearl_learning.scripts.audit_gate3_vanilla_pearl_mechanism import gate3_causal_chain_verdict


def recompute(suite_path: Path, oracle: dict[str, Any] | None, output_dir: Path | None) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("schema") != "gate3_vanilla_pearl_mechanism_causal_audit_suite_v1":
        raise ValueError(f"{suite_path} is not a Gate 3 causal audit suite")
    gate = gate3_causal_chain_verdict(suite, oracle)
    gate["inputs"] = suite.get("provenance", {})
    gate["recomputed_from"] = {
        "audit_suite_path": str(suite_path.resolve()),
        "audit_suite_hash": content_hash(suite),
        "oracle_audit_hash": None if oracle is None else content_hash(oracle),
        "training_updates": 0,
        "environment_steps": 0,
    }
    target = (output_dir if output_dir is not None else suite_path.parent) / "gate3_causal_chain_gate_v3.json"
    write_json(target, gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="append", required=True,
                        help="existing gate3_causal_audit.json; may be repeated")
    parser.add_argument("--oracle-audit",
                        help="single-task SAC query-oracle audit JSON for R_policy and Stage-C gating")
    parser.add_argument("--output-dir", help="override the verdict output directory (single --suite only)")
    args = parser.parse_args()
    oracle = None
    if args.oracle_audit:
        oracle = json.loads(Path(args.oracle_audit).read_text(encoding="utf-8"))
    if args.output_dir and len(args.suite) != 1:
        raise ValueError("--output-dir requires exactly one --suite")
    output_dir = Path(args.output_dir) if args.output_dir else None
    for raw in args.suite:
        gate = recompute(Path(raw), oracle, output_dir)
        print(f"{raw}: Stage A {gate['stages']['stage_a_context_to_posterior']['status']}, "
              f"Stage B {gate['stages']['stage_b_posterior_to_action']['status']}, "
              f"Stage C {gate['stages']['stage_c_action_to_outcome']['status']}, "
              f"passed {gate['passed_stages'] or 'none'}")


if __name__ == "__main__":
    main()
