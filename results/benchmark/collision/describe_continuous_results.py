"""Complete registered equal-count coverage and auxiliary continuous reporting."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmark.audit_collision_statistics import check_coverage
from benchmark.audit_collision_results import require
from benchmark.collision_metrics import equal_count_coverage
from verify_cost_inventory import read, sha


def main():
    output = Path(__file__).parent
    folder = ROOT / "results/gcrt/collision/continuous/report"
    index_path = folder.parent / "run_index.json"
    proof_path = output / "full_continuous_raw_result_audit.json"
    index, proof = read(index_path), read(proof_path)
    require(index["status"] == proof["status"] == "complete" and len(index["runs"]) == proof["verified_runs"] == 100,
            "All actual continuous trajectories must pass before reporting")
    proof_by_path = {str(Path(r["path"]).resolve()): r for r in proof["runs"]}
    rows = index["runs"]
    for row in rows:
        require(sha(row["path"]) == proof_by_path[str(Path(row["path"]).resolve())]["sha256"], "Audited result changed")
    tasks, methods = defaultdict(list), defaultdict(list)
    fields = ("target_id", "scenario", "condition", "environment_seed", "algorithm_seed")
    for row in rows:
        tasks[tuple(row[k] for k in fields)].append(row)
        methods[row["method"]].append(row)
    coverage = [{"task": list(key), "methods": [r["method"] for r in group], **equal_count_coverage(group)}
                for key, group in sorted(tasks.items())]
    require(len(coverage) == 20 and all(len(group) == 5 for group in tasks.values()), "Missing equal-count counterpart")
    check_coverage(coverage, rows)
    coverage_path = folder / "equal_count_behavior_coverage.json"
    coverage_path.write_text(json.dumps({"status": "complete", "groups": coverage,
        "scope": "First prefix reaching each common attainable number of unique collisions. Empty common ranges retained; no equal-count aggregate or superiority claim is inferred."}, indent=2) + "\n", encoding="utf-8")
    auxiliary = {}
    for method, group in sorted(methods.items()):
        violations = Counter()
        for row in group:
            violations.update(row["violation_counts"])
        auxiliary[method] = {"runs": len(group), "paid_queries": sum(r["queries"] for r in group),
                             "violating_execution_counts_by_flag": dict(violations)}
        for key in ("minimum_clearance_m", "minimum_ttc_s", "brier"):
            values = [r[key] for r in group if r[key] is not None]
            require(all(math.isfinite(v) for v in values), "Nonfinite auxiliary measurement")
            auxiliary[method][key] = {"mean_of_measured_run_values": math.fsum(values) / len(values) if values else None,
                                      "measured_runs": len(values), "missing_runs": len(group) - len(values)}
    auxiliary_path = folder / "auxiliary_safety.json"
    auxiliary_path.write_text(json.dumps({"status": "complete", "methods": auxiliary,
        "scope": "Descriptive run safety minima,available prequery-probability Brier values and violation counts. Missing values stay missing. Historical/target-version-new labels require separate real-source certification and are not inferred for proposed continuous points."}, indent=2) + "\n", encoding="utf-8")
    result = {"status": "verified_continuous_reporting_annexes", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "validator_sha256": sha(__file__), "independent_coverage_checker_sha256": sha(ROOT / "benchmark/audit_collision_statistics.py"),
              "raw_audit_sha256": sha(proof_path), "index_sha256": sha(index_path), "additional_physical_calls": 0,
              "coverage_groups": 20, "positive_common_ranges": sum(r["common_maximum"] > 0 for r in coverage),
              "auxiliary_method_groups": len(auxiliary), "coverage_path": str(coverage_path),
              "coverage_sha256": sha(coverage_path), "auxiliary_path": str(auxiliary_path), "auxiliary_sha256": sha(auxiliary_path)}
    (output / "continuous_reporting_annex_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["status"], "20 equal-count groups;", result["positive_common_ranges"], "positive ranges;5 auxiliary method groups", flush=True)


if __name__ == "__main__":
    main()
