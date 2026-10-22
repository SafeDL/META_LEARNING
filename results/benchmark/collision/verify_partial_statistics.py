"""Cross-check statistical routines on the1280 independently reconciled runs."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.audit_collision_results import compare
from benchmark.audit_collision_statistics import paired_summary, restricted_records, survival_summary
from benchmark.collision_metrics import lineage_survival, paired_lineage_bootstrap
from benchmark.protocol import DATA_ROOT, DEFAULT, ROOT, method_contract


def main():
    started = time.perf_counter()
    base = ROOT / "results/benchmark/collision"
    receipt = json.loads((base / "fixed_cohorts/front_brake_new/cohort_verification.json").read_text(encoding="utf-8"))
    path = Path(receipt["independent_audit"])
    if not path.is_absolute():
        path = ROOT / path
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    assert fingerprint == receipt["independent_audit_sha256"]
    proof = json.loads(path.read_text(encoding="utf-8"))
    assert proof["verified_runs"] == 1280 and not proof["errors"]
    frozen = json.loads((DATA_ROOT / "frozen_protocol.json").read_text(encoding="utf-8"))
    assert frozen["source_contract"] == method_contract() == proof["method_contract"]
    for row in proof["runs"]:
        assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
    comparisons, survivals = [], []
    for study in ("overall", "strict_family"):
        for condition in ("reuse", "new"):
            rows = [r for r in proof["runs"] if r["study"] == study and r["condition"] == condition]
            assert len(rows) == 320
            for method in sorted({r["method"] for r in rows}):
                group = [r for r in rows if r["method"] == method]
                expected = survival_summary(group, restriction=128)
                compare(lineage_survival(group, restriction=128), expected, "actual lineage survival cross-check")
                survivals.append({"study": study, "condition": condition, "method": method,
                                  "independent_lineages": expected["independent_lineages"],
                                  "censored_runs": expected["censored_runs"],
                                  "restricted_mean_queries": expected["restricted_mean_queries"]})
                if method == "history_response":
                    continue
                for metric in ("unique_collisions", "discovery_auc", "behavior_coverage", "restricted_first_queries"):
                    selected = restricted_records(rows, DEFAULT.budget) if metric == "restricted_first_queries" else rows
                    expected = paired_summary(selected, "history_response", method, metric)
                    compare(paired_lineage_bootstrap(selected, "history_response", method, metric), expected,
                            "actual independent hierarchical bootstrap cross-check")
                    comparisons.append({"study": study, "condition": condition, **expected})
                print("verified actual statistical cell", study, condition, method, flush=True)
    assert len(comparisons) == 112 and len(survivals) == 32
    assert frozen["source_contract"] == method_contract()
    result = {"status": "verified_actual_statistics_crosscheck", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Core statistical routines on the1280 completed front-brake/reuse+new runs only; final full-matrix reports, certified label tables, costs and figure review remain required.",
              "verified_runs": 1280, "verified_intervals": 112, "verified_survival_cells": 32,
              "bootstrap_repetitions": 2000, "bootstrap_seed": 1501,
              "independent_lineages_per_cell": 8, "paired_runs_per_contrast": 40,
              "full_statistical_report_verified": False, "additional_physical_calls": 0,
              "input_raw_audit": str(path), "input_raw_audit_sha256": fingerprint,
              "validator_sha256": hashlib.sha256((ROOT / "benchmark/audit_collision_statistics.py").read_bytes()).hexdigest(),
              "elapsed_s": time.perf_counter() - started, "intervals": comparisons, "survival": survivals}
    (base / "partial_statistics_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("verified112 statistical contrasts and32 survival cells on1280 real runs", flush=True)


if __name__ == "__main__":
    main()
