"""Reconcile completed construction costs; never execute a SUT or infer missing time."""

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
from controllers.highway.catalog import BY_ID


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    data, output = ROOT / "data/collision", Path(__file__).parent
    frozen = read(data / "frozen_protocol.json")
    require(frozen["source_contract"] == method_contract(), "Frozen finite sources changed")
    groups, banks = defaultdict(list), []
    for prefix in ("physical/history", "physical/development", "targets"):
        for path in sorted((data / prefix).rglob("manifest.json")):
            manifest = read(path)
            require(manifest["format"] == "collision-execution-bank" and manifest["status"] == "complete",
                    f"Incomplete construction bank: {path}")
            attempts = [(p, read(p)) for p in sorted((path.parent / "attempts").glob("*.json"))]
            require(len({r["call"] for _, r in attempts}) == len(attempts), "Duplicate physical call")
            good = [r for _, r in attempts if r["status"] == "complete"]
            require(all(r["status"] in {"complete", "failed", "running"} for _, r in attempts),
                    "Unrecognized attempt status")
            values = {
                "attempted_calls": len(attempts), "successful_calls": len(good),
                "failed_or_interrupted_calls": len(attempts) - len(good),
                "physical_execution_s": math.fsum(r["execution_s"] for r in good),
                "total_successful_call_wall_s": math.fsum(r["wall_s"] for r in good),
                "sut_initialization_s": math.fsum(r["wall_s"] - r["execution_s"] for r in good),
                "failed_call_wall_s": math.fsum(r.get("wall_s", 0) for _, r in attempts if r["status"] == "failed"),
                "interrupted_call_duration_unknown": sum(r["status"] == "running" for _, r in attempts),
            }
            for key, value in values.items():
                require(math.isclose(manifest[key], value, rel_tol=1e-10, abs_tol=1e-7),
                        f"Attempt/manifest cost mismatch: {path} {key}")
            receipt = {"path": str(path.relative_to(ROOT)), "manifest_sha256": sha(path), **values,
                       "attempt_files_sha256": hashlib.sha256("".join(
                           p.name + sha(p) for p, _ in attempts).encode()).hexdigest()}
            banks.append(receipt)
            groups[prefix].append(receipt)
    require([len(groups[k]) for k in ("physical/history", "physical/development", "targets")] == [32, 16, 96],
            "Construction matrix is incomplete")
    require(sum(r["attempted_calls"] for r in banks) == 73728, "Unexpected construction calls")
    models = []
    for artifact in frozen["method"]["models"]:
        weights = ROOT / artifact["path"]
        require(sha(weights) == artifact["sha256"], "Frozen weights changed")
        path = weights.with_suffix(".json")
        report = read(path)
        require(report["status"] == "complete" and math.isfinite(report["elapsed_s"]) and report["elapsed_s"] > 0,
                "Missing measured model construction time")
        models.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "elapsed_s": report["elapsed_s"]})
    require(len(models) == len({r["path"] for r in models}) == 156, "Duplicate/missing model build")
    registered = {s.training_run for s in BY_ID.values() if s.family == "learning"}
    paths = sorted((ROOT / "controllers/highway/checkpoints/collision").glob("*/training.json"))
    paths += sorted((ROOT.parent / "META_LEARNING_archive/unqualified_policy_construction").glob("*/training.json"))
    paths += sorted((output / "policy_construction/probes").glob("*/training.json"))
    policies = []
    for path in paths:
        report = read(path)
        policies.append({"path": str(path), "sha256": sha(path), "status": report["status"],
                         "use": "registered_sut" if path.parent.name in registered else "construction_reference",
                         "elapsed_s": report.get("elapsed_s"), "actual_timesteps": report.get("actual_timesteps"),
                         "qualification_executions": report.get("qualification_executions")})
    require(len({p["sha256"] for p in policies}) == len(policies), "Duplicate policy record requires accounting review")
    formal = [p for p in policies if p["use"] == "registered_sut"]
    require(len(formal) == 5 and all(p["actual_timesteps"] == 4194304 for p in formal), "Wrong formal driving runs")
    qualifications = []
    for path in sorted((output / "qualification/public_interface").glob("*/qualification.json")):
        report = read(path)
        qualifications.append({"path": str(path), "sha256": sha(path), "training_run": report["training_run"],
                               "actual_physical_calls": report["actual_physical_calls"], "elapsed_s": report["elapsed_s"],
                               "interrupted_attempt_durations_unknown": report.get("interrupted_attempt_durations_unknown", False)})
    result = {
        "status": "verified_completed_construction_inventory", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Historical/development/target acquisition,156 frozen model builds and retained policy construction records. Final online, certification and continuous costs remain separate and required.",
        "validator_sha256": sha(__file__), "protocol_sha256": sha(data / "frozen_protocol.json"),
        "additional_physical_calls": 0, "physical_banks": banks,
        "acquisition_totals": {key: {"banks": len(rows), **{
            field: math.fsum(r[field] for r in rows) for field in (
                "attempted_calls", "physical_execution_s", "total_successful_call_wall_s", "sut_initialization_s",
                "failed_or_interrupted_calls", "failed_call_wall_s", "interrupted_call_duration_unknown")}}
            for key, rows in groups.items()},
        "offline_model_builds": models, "offline_model_elapsed_s": math.fsum(p["elapsed_s"] for p in models),
        "policy_construction": policies, "policy_elapsed_s": math.fsum(p["elapsed_s"] for p in policies if p["elapsed_s"] is not None),
        "policy_time_records_missing": [p["path"] for p in policies if p["elapsed_s"] is None],
        "policy_actual_timesteps": sum(p["actual_timesteps"] for p in policies if p["actual_timesteps"] is not None),
        "public_interface_qualification": qualifications,
        "accounting": ["Physical bank paths are charged once; attempt records are the source for time and calls.",
                       "Execution time, initialization overhead and successful call wall time are overlapping measures; do not add them together.",
                       "Policy elapsed time already includes training-time qualification; do not add its course counts as another duration.",
                       "Cached general qualification summaries have no independent time records; their duration is unknown, not zero.",
                       "Concurrent task durations are measured worker time and are not end-to-end pipeline elapsed time.",
                       "No final online, certification, continuous or amortization claim is made by this construction inventory."],
    }
    destination = output / "construction_cost_inventory.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(result["status"], len(banks), "banks/73728 calls;156 models;", len(policies), "policy records", flush=True)


if __name__ == "__main__":
    main()
