"""Create formal-training validation from all-task audit and baseline manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pearl_learning.src.baselines import PRETRAIN_BASELINE_NAMES
from pearl_learning.src.formal_validation import FORMAL_VALIDATION_SCHEMA
from pearl_learning.src.io import content_hash, write_json
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskbook", required=True); parser.add_argument("--topology-audit", required=True); parser.add_argument("--integrity-audit", required=True); parser.add_argument("--heterogeneity-audit", required=True); parser.add_argument("--baseline-root", required=True); parser.add_argument("--baseline-env-steps", type=int, required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); taskbook = load_taskbook(args.taskbook); taskbook_hash = content_hash(taskbook_payload(taskbook))
    audit = json.loads(Path(args.topology_audit).read_text(encoding="utf-8"))
    if audit.get("taskbook_hash") != taskbook_hash or audit.get("passed") != audit.get("total"):
        raise SystemExit("topology audit is incomplete or belongs to another taskbook")
    integrity = json.loads(Path(args.integrity_audit).read_text(encoding="utf-8"))
    if integrity.get("taskbook_hash") != taskbook_hash or integrity.get("status") != "pass":
        raise SystemExit("integrity audit is incomplete or belongs to another taskbook")
    heterogeneity = json.loads(Path(args.heterogeneity_audit).read_text(encoding="utf-8"))
    if heterogeneity.get("taskbook_hash") != taskbook_hash or heterogeneity.get("status") != "pass":
        raise SystemExit("task heterogeneity audit is incomplete, failed, or belongs to another taskbook")
    completed = []
    for baseline in PRETRAIN_BASELINE_NAMES:
        manifest = Path(args.baseline_root) / baseline / "baseline_manifest.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("taskbook_hash") == taskbook_hash and payload.get("status") == "completed" and int(payload.get("environment_steps", -1)) == args.baseline_env_steps:
                completed.append(baseline)
    write_json(args.output, {"schema": FORMAL_VALIDATION_SCHEMA, "taskbook_hash": taskbook_hash, "topology_audit": "pass", "integrity_audit": "pass", "heterogeneity_audit": "pass", "baseline_environment_steps": args.baseline_env_steps, "completed_baselines": completed})
    if set(completed) != set(PRETRAIN_BASELINE_NAMES):
        raise SystemExit("formal validation intentionally remains incomplete until all pre-training baselines finish")


if __name__ == "__main__":
    main()
