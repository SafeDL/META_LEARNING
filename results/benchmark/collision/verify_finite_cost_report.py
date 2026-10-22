"""Reconcile the actual finite report against paid attempts and saved run clocks."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmark.protocol import method_contract
from verify_cost_inventory import read, require, sha


FIELDS = ("attempted_calls", "successful_calls", "failed_or_interrupted_calls",
          "physical_execution_s", "sut_initialization_s", "total_successful_call_wall_s",
          "failed_call_wall_s", "interrupted_call_duration_unknown")


def close(a, b, context):
    require(math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7), context)


def main():
    output = Path(__file__).parent
    inventory_path = output / "construction_cost_inventory.json"
    inventory = read(inventory_path)
    require(inventory["status"] == "verified_completed_construction_inventory", "Construction not verified")
    require(inventory["validator_sha256"] == sha(output / "verify_cost_inventory.py"), "Inventory checker changed")
    frozen = read(ROOT / "data/collision/frozen_protocol.json")
    require(frozen["source_contract"] == method_contract(), "Frozen finite producer changed")
    proof_path = output / "full_finite_raw_result_audit.json"
    proof = read(proof_path)
    receipt = read(output / "full_finite_completion_verification.json")
    require(proof["status"] == "complete" and proof["verified_runs"] == 10880
            and receipt["full_audit_sha256"] == sha(proof_path), "Full raw evidence missing")
    report_path = ROOT / "results/gcrt/collision/report/report.json"
    report = read(report_path)
    require(report["status"] == "complete" and report["run_count"] == 10880, "Incomplete finite report")
    costs = report["costs"]
    measured = {}
    for row in inventory["physical_banks"]:
        path = ROOT / row["path"]
        require(sha(path) == row["manifest_sha256"], "Construction manifest changed")
        measured[str(path.resolve())] = {key: row[key] for key in FIELDS}
    certificates = []
    for path in sorted((ROOT / "data/collision/certification/shared").glob("*/*/env_*/*/manifest.json")):
        manifest = read(path)
        require(manifest["status"] == "complete", "Incomplete certification bank")
        attempts = [(p, read(p)) for p in sorted((path.parent / "attempts").glob("*.json"))]
        require(len({r["call"] for _, r in attempts}) == len(attempts), "Duplicate certification attempt")
        require(all(r["status"] in {"complete", "failed", "running"} for _, r in attempts), "Unknown attempt")
        good = [r for _, r in attempts if r["status"] == "complete"]
        values = {"attempted_calls": len(attempts), "successful_calls": len(good),
                  "failed_or_interrupted_calls": len(attempts) - len(good),
                  "physical_execution_s": math.fsum(r["execution_s"] for r in good),
                  "total_successful_call_wall_s": math.fsum(r["wall_s"] for r in good),
                  "sut_initialization_s": math.fsum(r["wall_s"] - r["execution_s"] for r in good),
                  "failed_call_wall_s": math.fsum(r.get("wall_s", 0) for _, r in attempts if r["status"] == "failed"),
                  "interrupted_call_duration_unknown": sum(r["status"] == "running" for _, r in attempts)}
        for key in FIELDS:
            close(values[key], manifest[key], "Certification attempts/manifest: " + key)
        measured[str(path.resolve())] = values
        certificates.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), **values,
                             "attempt_files_sha256": hashlib.sha256("".join(
                                 p.name + sha(p) for p, _ in attempts).encode()).hexdigest()})
    require(len(certificates) == 64 and len(measured) == 208, "Wrong complete finite physical matrix")
    reported = {str((ROOT / r["path"]).resolve()): r for r in costs["physical_banks"]}
    require(len(reported) == len(costs["physical_banks"]) and set(reported) == set(measured), "Bank duplication/omission")
    for path, values in measured.items():
        for key in FIELDS:
            close(reported[path][key], values[key], "Reported paid bank: " + key)
    for field in ("offline_model_builds", "policy_construction", "public_interface_qualification"):
        expected = {str((ROOT / r["path"]).resolve()): r for r in inventory[field]}
        actual = {str((ROOT / r["path"]).resolve()): r for r in costs[field]}
        require(len(actual) == len(costs[field]) and set(actual) == set(expected), "Construction report omission: " + field)
        for path, row in expected.items():
            require(sha(path) == row["sha256"], "Construction metadata changed")
            for key in actual[path]:
                if key == "path":
                    continue
                source = read(path)
                value = row[key] if key in row else source[key]
                require(actual[path][key] == value, "Construction reported field changed: " + key)
    index_path = ROOT / "results/gcrt/collision/run_index.json"
    require(sha(index_path) == receipt["run_index_sha256"], "Audited index changed")
    index = read(index_path)
    groups, main_new = defaultdict(list), defaultdict(list)
    keys = ("selection_s", "adaptation_s", "model_construction_s", "selector_process_wall_s", "equivalent_execution_s")
    for row in index["runs"]:
        saved = read(row["path"])
        require(saved["cost"]["execution_mode"] == "replay"
                and saved["cost"]["actual_new_physical_calls"] == 0, "Replay reported new physical calls")
        values = {"selection_s": row["selection_s"], "adaptation_s": row["adaptation_s"],
                  "model_construction_s": saved["cost"]["model_construction_s"] + saved["result"].get("model_construction_s", 0),
                  "selector_process_wall_s": saved["cost"]["selector_process_wall_s"],
                  "equivalent_execution_s": row["equivalent_execution_s"]}
        require(all(math.isfinite(v) and v >= 0 for v in values.values()), "Invalid measured online clock")
        groups[(row["study"], row["method"])].append(values)
        if row["study"] == "overall" and row["condition"] == "new":
            main_new[row["method"]].append(values)
    actual = {(r["study"], r["method"]): r for r in costs["online"]}
    require(len(actual) == len(costs["online"]) == len(groups) == 23, "Online cost groups omitted/duplicated")
    for key, values in groups.items():
        require(actual[key]["runs"] == len(values), "Wrong online cost run count")
        for field in keys:
            close(actual[key]["mean_" + field], math.fsum(v[field] for v in values) / len(values), "Online cost mean: " + field)
    result = {"status": "verified_finite_cost_report", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Actual completed finite report only. Physical calls and time from unique paid attempt ledgers; construction inventory and all10880 saved online clocks. Certification outcome audit, statistics, figures and continuous costs remain separate.",
              "validator_sha256": sha(__file__), "report_sha256": sha(report_path),
              "construction_inventory_sha256": sha(inventory_path), "full_raw_audit_sha256": sha(proof_path),
              "additional_physical_calls": 0, "physical_banks": 208, "unique_physical_calls": sum(v["attempted_calls"] for v in measured.values()),
              "certification_banks": certificates,
              "certification_totals": {field: math.fsum(r[field] for r in certificates) for field in FIELDS},
              "offline_models": 156, "retained_policy_records": 17, "public_qualification_records": 7,
              "online_runs": 10880, "online_groups": 23,
              "main_new_condition_online": {method: {"runs": len(rows), **{
                  "mean_" + field: math.fsum(r[field] for r in rows) / len(rows) for field in keys}}
                  for method, rows in main_new.items()},
              "accounting": [*inventory["accounting"][:-1],
                             "Selector process wall time already contains selector components and replay wait; do not add its components again.",
                             "Equivalent selected physical execution time is not additional simulator wall time or an observed end-to-end speedup.",
                             "Repeated certificate references, methods, scopes and seeds do not multiply a shared physical source bank's cost.",
                             "Continuous development/final physical calls and costs are not part of this finite report receipt."]}
    destination = output / "finite_cost_report_verification.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"], result["unique_physical_calls"], "unique physical calls;10880 online runs;23 groups", flush=True)


if __name__ == "__main__":
    main()
