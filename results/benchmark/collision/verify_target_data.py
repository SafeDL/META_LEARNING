"""Incremental final-bank audit. Reads existing paid traces; never runs a SUT."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from benchmark.audit_collision_results import RawEvidence, compare, require
from benchmark.executions import execution_contract
from benchmark.protocol import DATA_ROOT, DEFAULT, ROOT, method_contract, validate_sut_freeze
from benchmark.scenarios import DOMAINS, sample
from controllers.highway.catalog import SUTS


OUTPUT = Path(__file__).with_name("target_data_audit")
SUMMARY = Path(__file__).with_name("target_data_verification.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def files_digest(paths):
    result = hashlib.sha256()
    for path in sorted(paths):
        result.update(path.name.encode())
        result.update(bytes.fromhex(sha(path)))
    return result.hexdigest()


def bank_paths():
    for kind in DOMAINS:
        for condition in DEFAULT.conditions:
            for environment_seed in DEFAULT.environment_seeds:
                for sut in SUTS:
                    if sut.role == "target":
                        yield sut, kind, condition, environment_seed


def verify_bank(sut, kind, condition, environment_seed, frozen, protocol_sha, validator_sha, metric_sha):
    root = DATA_ROOT / "targets" / sut.sut_id / condition / kind / f"env_{environment_seed}"
    raw = RawEvidence()
    manifest, *_ = raw.bank(root)
    require(manifest["sut_id"] == sut.sut_id and manifest["environment_seed"] == environment_seed,
            "Final physical bank identity differs from its registered task")
    compare(manifest["contract"], frozen["sut_contracts"][sut.sut_id], "frozen target execution contract")
    compare(manifest["contract"], execution_contract(sut), "current target execution contract")
    expected = [asdict(scene) for scene in sample(kind, DEFAULT.candidate_count, condition)]
    compare(manifest["scenarios"], expected, "registered final candidate pool")
    require(manifest["candidate_count"] == DEFAULT.candidate_count, "Final candidate count differs")
    require(len(list((root / "executions").glob("*.json.gz"))) == DEFAULT.candidate_count,
            "Final execution file count differs")
    worlds, responses = hashlib.sha256(), hashlib.sha256()
    retained_calls, retained_s = set(), 0.0
    for index in range(DEFAULT.candidate_count):
        evidence = raw.execution(root, index)
        path = root / "executions" / f"{index:06d}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        compare(payload["scenario"], expected[index], "final raw candidate coordinate")
        require(payload["call"] not in retained_calls, "Final coordinates reused a physical attempt")
        retained_calls.add(payload["call"])
        retained_s += payload["feedback"]["execution_s"]
        trace = payload["trajectory"]
        require(bool(trace), "Missing final trajectory")
        times = np.asarray([frame["time_s"] for frame in trace])
        require(abs(times[0]) < 1e-9 and np.allclose(np.diff(times), 1 / DEFAULT.physics_hz, atol=1e-9, rtol=0)
                and times[-1] <= DEFAULT.episode_duration_s + 1e-9, "Final physical timebase differs")
        if not payload["feedback"]["collision"] and not payload["feedback"]["safety"]["violations"]:
            require(abs(times[-1] - DEFAULT.episode_duration_s) < 1e-9, "Final execution ended before its horizon")
        for position, frame in enumerate(trace):
            command = frame["requested_control"]
            require(-8 - 1e-9 <= command["acceleration"] <= 3 + 1e-9 and abs(command["steering"]) <= .45 + 1e-9,
                    "Final requested control exceeds common limits")
            if position and command != trace[position - 1]["requested_control"]:
                require((position - 1) % (DEFAULT.physics_hz // DEFAULT.decision_hz) == 0,
                        "Final SUT command changed outside the common decision ticks")
        worlds.update(json.dumps(trace[0]["actors"], sort_keys=True).encode())
        responses.update(path.name.encode())
        responses.update(bytes.fromhex(evidence["sha256"]))
    require(len(retained_calls) == 512, "Final retained paid-call count differs")
    return {"status": "complete", "sut_id": sut.sut_id, "lineage": sut.lineage, "family": sut.family,
            "scenario": kind, "condition": condition, "environment_seed": environment_seed,
            "bank": str(root), "responses": DEFAULT.candidate_count,
            "actual_physical_calls": manifest["attempted_calls"], "successful_calls": manifest["successful_calls"],
            "failed_or_interrupted_calls": manifest["failed_or_interrupted_calls"],
            "interrupted_duration_unknown_calls": manifest["interrupted_call_duration_unknown"],
            "successful_calls_without_retained_response": manifest["successful_calls"] - len(retained_calls),
            "physical_execution_s": manifest["physical_execution_s"], "retained_response_execution_s": retained_s,
            "failed_call_wall_s": manifest["failed_call_wall_s"], "initial_worlds_sha256": worlds.hexdigest(),
            "manifest_sha256": sha(root / "manifest.json"), "call_ledger_sha256": sha(root / "calls.jsonl"),
            "attempt_files_sha256": files_digest((root / "attempts").glob("*.json")),
            "raw_execution_files_sha256": responses.hexdigest(), "protocol_sha256": protocol_sha,
            "validator_sha256": validator_sha, "raw_metric_validator_sha256": metric_sha,
            "observed_at_utc": datetime.now(timezone.utc).isoformat()}


def final_integrity(rows):
    for row in rows:
        root = Path(row["bank"])
        require(sha(root / "manifest.json") == row["manifest_sha256"] and sha(root / "calls.jsonl") == row["call_ledger_sha256"],
                "A final audited bank or call ledger changed")
        require(files_digest((root / "attempts").glob("*.json")) == row["attempt_files_sha256"], "Final attempt records changed")
        require(files_digest((root / "executions").glob("*.json.gz")) == row["raw_execution_files_sha256"], "Final raw responses changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    arguments = parser.parse_args()
    started = time.perf_counter()
    protocol_path = DATA_ROOT / "frozen_protocol.json"
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_sut_freeze(frozen)
    protocol_sha, validator_sha = sha(protocol_path), sha(Path(__file__))
    metric_sha = sha(ROOT / "benchmark/audit_collision_results.py")
    require(frozen["source_contract"] == method_contract(), "Method changed after finite freezing")
    plan = list(bank_paths())
    require(len(plan) == 96 and DEFAULT.candidate_count == 512, "Final bank plan changed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "watcher.json", {"pid": __import__("os").getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                                        "protocol_sha256": protocol_sha, "validator_sha256": validator_sha})
    checked = {}
    while True:
        rows, pending, errors = [], [], []
        for sut, kind, condition, env in plan:
            root = DATA_ROOT / "targets" / sut.sut_id / condition / kind / f"env_{env}"
            manifest_path = root / "manifest.json"
            if not manifest_path.exists():
                pending.append(str(root))
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pending.append(str(root))
                continue
            if manifest["status"] != "complete":
                pending.append(str(root))
                continue
            receipt_path = OUTPUT / "banks" / sut.sut_id / condition / kind / f"env_{env}.json"
            try:
                key = str(root)
                cached = checked.get(key)
                if cached is None and receipt_path.exists():
                    cached = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (cached is not None and cached["validator_sha256"] == validator_sha
                        and cached["raw_metric_validator_sha256"] == metric_sha
                        and cached["protocol_sha256"] == protocol_sha
                        and cached["manifest_sha256"] == sha(manifest_path)
                        and cached["call_ledger_sha256"] == sha(root / "calls.jsonl")):
                    row = cached
                else:
                    row = verify_bank(sut, kind, condition, env, frozen, protocol_sha, validator_sha, metric_sha)
                    write_json(receipt_path, row)
                    print("verified final bank", sut.sut_id, condition, kind, flush=True)
                checked[key] = row
                rows.append(row)
            except Exception as error:
                errors.append({"bank": str(root), "error": repr(error)})
        groups = defaultdict(set)
        for row in rows:
            groups[(row["scenario"], row["condition"], row["environment_seed"])].add(row["initial_worlds_sha256"])
        if any(len(values) != 1 for values in groups.values()):
            errors.append({"error": "Same-coordinate target SUTs faced different initial worlds"})
        integrity_checked = False
        if not pending and not errors:
            try:
                final_integrity(rows)
                integrity_checked = True
            except Exception as error:
                errors.append({"error": repr(error)})
        require(sha(protocol_path) == protocol_sha and frozen["source_contract"] == method_contract(), "Frozen inputs changed during target auditing")
        result = {"status": "failed" if errors else "complete" if not pending and integrity_checked else "partial",
                  "scope": "All final target banks and actual raw trajectories/cost logs only; discovery comparisons are separate.",
                  "observed_at_utc": datetime.now(timezone.utc).isoformat(), "expected_banks": 96, "verified_banks": len(rows),
                  "verified_responses": sum(row["responses"] for row in rows), "additional_physical_calls": 0,
                  "protocol_sha256": protocol_sha, "validator_sha256": validator_sha, "raw_metric_validator_sha256": metric_sha,
                  "full_current_content_rehash_checked": integrity_checked, "verification_wall_s": time.perf_counter() - started,
                  "banks": rows, "pending": pending, "errors": errors}
        write_json(SUMMARY, result)
        print(result["status"], len(rows), "/ 96 final banks;", len(errors), "errors", flush=True)
        if errors or result["status"] == "complete" or not arguments.watch:
            raise SystemExit(1 if errors or result["status"] != "complete" else 0)
        time.sleep(15)


if __name__ == "__main__":
    main()
