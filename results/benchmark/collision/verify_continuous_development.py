"""Audit all actual development calls and their binding to the continuous freeze."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmark.audit_collision_results import compare, require
from benchmark.audit_continuous_results import verify_run
from benchmark.continuous_runs import CONFIGURATIONS, source_contract
from benchmark.continuous_selection import METHODS
from benchmark.develop_continuous import OUTPUT, SETTINGS, TASKS, WORKERS
from benchmark.evaluate_continuous import validate_frozen
from benchmark.protocol import DEFAULT
from controllers.highway.catalog import BY_ID
from verify_cost_inventory import read, sha


def main():
    path = OUTPUT / "report.json"
    report = read(path)
    require(report["status"] == "complete" and report["data_role"] == "development_only", "Incomplete development")
    compare(report["settings"], SETTINGS, "Continuous development settings")
    compare(report["configurations"], CONFIGURATIONS, "Continuous fixed search mechanisms")
    compare(report["source_contract"], source_contract(), "Continuous development producer contract")
    require(report["execution_workers"] == WORKERS == 4 and SETTINGS["ordinary_only"] is True,
            "Development worker count or held-out combinations changed")
    expected = [(sut, kind, method, seed) for sut, kind in TASKS
                for seed in SETTINGS["algorithm_seeds"] for method in METHODS]
    indexed = {str(Path(r["path"]).resolve()): r for r in report["runs"]}
    require(len(expected) == len(indexed) == len(report["runs"]) == 40, "Development duplicates or omissions")
    rows, errors = [], []
    for sut, kind, method, seed in expected:
        run_path = OUTPUT / sut / kind / f"{method}_{seed}.json"
        metadata = {"target_id": sut, "scenario": kind, "method": method, "algorithm_seed": seed,
                    "environment_seed": DEFAULT.environment_seeds[0], "condition": "continuous_development"}
        try:
            require(BY_ID[sut].role == "development", "Final object entered continuous development")
            row = verify_run(run_path, metadata, budget=SETTINGS["budget"],
                             initial_count=SETTINGS["initial_count"], ordinary_only=True)
            compare(indexed[str(run_path.resolve())],
                    {k: v for k, v in row.items() if k not in {"sha256", "raw_evidence_sha256"}},
                    "Actual continuous development report row")
            rows.append(row)
        except Exception as error:
            errors.append({"path": str(run_path), "error": repr(error)})
        print("reconstructed actual continuous development", len(rows), "/40", flush=True)
    require(not errors and len(rows) == 40, repr(errors))
    validate_frozen()
    frozen_path = ROOT / "data/collision/frozen_continuous.json"
    frozen = read(frozen_path)
    require(frozen["development_report_sha256"] == sha(path)
            and frozen["execution_workers"] == {"development": 4, "final": 12}, "Continuous freeze binding differs")
    state = read(ROOT / "results/gcrt/collision/pipeline/state.json")
    stages = state["stages"]
    for name in ("continuous_development", "continuous_freeze"):
        require(stages[name]["status"] == "complete" and stages[name]["native_exit_code"] == 0,
                "A native development/freeze stage did not complete successfully")
    require(stages["continuous_development"]["finished_utc"] <= stages["continuous_freeze"]["started_utc"],
            "Continuous freeze preceded completed development")
    if "continuous_comparisons" in stages:
        require(stages["continuous_freeze"]["finished_utc"] <= stages["continuous_comparisons"]["started_utc"],
                "Final continuous evaluation preceded its freeze")
    result = {"status": "verified_execution", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "All40 actual development runs,1280 paid calls/raw trajectories,ordinary-only coordinates,development roles,paired initializations,32-call budgets,metrics,costs,report and subsequent continuous freeze. Final100 runs and statistics remain separate.",
              "validator_sha256": sha(__file__),
              "raw_validator_sha256": sha(ROOT / "benchmark/audit_continuous_results.py"),
              "metric_validator_sha256": sha(ROOT / "benchmark/audit_collision_results.py"),
              "report_sha256": sha(path), "frozen_continuous_sha256": sha(frozen_path),
              "additional_physical_calls": 0, "verified_runs": 40, "actual_calls": 1280,
              "errors": [], "execution_workers": frozen["execution_workers"], "runs": rows}
    destination = Path(__file__).with_name("continuous_development_verification.json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("verified_execution40 development runs/1280 actual calls and continuous freeze", flush=True)


if __name__ == "__main__":
    main()
