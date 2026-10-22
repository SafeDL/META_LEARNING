"""Read-only reconstruction of construction archives from their actual execution ledgers."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np

from controllers.highway.catalog import BY_ID, SUTS

from .executions import ExecutionBank, execution_contract
from .protocol import DATA_ROOT, DEFAULT, ROOT
from .scenarios import DOMAINS, sample
from .schema import QueryFeedback
from .storage import load_archive


OUTPUT = ROOT / "results/benchmark/collision/construction_data_audit.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def verify_archive(path, *, data_root=DATA_ROOT, candidate_count=DEFAULT.candidate_count):
    """Check every retained response and account for every measured physical attempt."""
    archive = load_archive(path)
    sut = BY_ID[archive.source_id]
    kind = archive.scenarios[0].kind
    require(archive.role == sut.role and archive.role in ("history", "development"), "Archive data role mismatch")
    require((archive.family, archive.lineage) == (sut.family, sut.lineage), "Archive control lineage mismatch")
    canonical_path = Path(data_root) / archive.role / kind / f"{sut.sut_id}.npz"
    require(Path(path).resolve() == canonical_path.resolve(), "Archive/source identity differs from its canonical archive path")
    expected = sample(kind, candidate_count, archive.role)
    bank_root = Path(data_root) / "physical" / archive.role / kind / sut.sut_id
    require((Path(data_root) / archive.provenance["bank"]).resolve() == bank_root.resolve(), "Archive points outside its canonical physical bank")
    bank = ExecutionBank(bank_root)
    manifest = bank.manifest
    require(archive.scenarios == expected == bank.candidates, "Archive/canonical candidate coordinates differ")
    require(manifest["sut_id"] == sut.sut_id and manifest["environment_seed"] == DEFAULT.environment_seeds[0], "Physical bank execution identity mismatch")
    require(manifest["contract"] == execution_contract(sut) == archive.provenance["contract"], "Physical source implementation or weights changed")
    require(archive.provenance["qualified"] is True and archive.provenance["environment_seed"] == manifest["environment_seed"], "Archive execution provenance mismatch")

    attempts = {}
    known_ids = {s.scenario_id for s in expected}
    for file in sorted((bank_root / "attempts").glob("*.json")):
        record = json.loads(file.read_text(encoding="utf-8"))
        require(record["call"] == int(file.stem) and record["call"] not in attempts, "Duplicate/misidentified physical attempt")
        require(record["sut_id"] == sut.sut_id and record["environment_seed"] == manifest["environment_seed"], "Physical attempt belongs to another execution")
        require(record["scenario_id"] in known_ids, "Physical attempt has a foreign coordinate")
        require(record["status"] in ("complete", "failed", "running"), "Unknown physical attempt status")
        attempts[record["call"]] = record
    completed = [r for r in attempts.values() if r["status"] == "complete"]
    failed = [r for r in attempts.values() if r["status"] == "failed"]
    interrupted = [r for r in attempts.values() if r["status"] == "running"]
    require(manifest["candidate_count"] == candidate_count and manifest["attempted_calls"] == len(attempts), "Physical attempt/candidate count mismatch")
    require(manifest["successful_calls"] == len(completed) and manifest["failed_or_interrupted_calls"] == len(failed) + len(interrupted), "Physical outcome accounting mismatch")
    for key, value in (
        ("physical_execution_s", sum(r["execution_s"] for r in completed)),
        ("total_successful_call_wall_s", sum(r["wall_s"] for r in completed)),
        ("sut_initialization_s", sum(r["wall_s"] - r["execution_s"] for r in completed)),
        ("failed_call_wall_s", sum(r.get("wall_s", 0) for r in failed)),
    ):
        require(close(manifest[key], value), f"Cost ledger mismatch: {key}")
    require(manifest["interrupted_call_duration_unknown"] == len(interrupted), "Unknown-duration calls were hidden")
    require(archive.provenance["physical_calls"] == len(attempts), "Archive hides extra physical attempts")

    ledger = [json.loads(line) for line in (bank_root / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    scheduled = [r for r in ledger if r["status"] == "scheduled"]
    require(len({r["call"] for r in scheduled}) == len(scheduled), "Physical attempt scheduled twice under one call ID")
    for record in scheduled:
        require(0 <= record["index"] < candidate_count and record["scenario_id"] == expected[record["index"]].scenario_id, "Scheduled query/candidate mismatch")
    require(set(attempts) <= {r["call"] for r in scheduled}, "Unscheduled physical attempts")

    labels, safety, retained_calls, retained_execution_s = [], [], set(), 0.0
    initial_worlds, raw_digest = hashlib.sha256(), hashlib.sha256()
    dt = 1 / DEFAULT.physics_hz
    interval = DEFAULT.physics_hz // DEFAULT.decision_hz
    for index, scenario in enumerate(expected):
        file = bank_root / "executions" / f"{index:06d}.json.gz"
        encoded = file.read_bytes()
        raw_digest.update(file.name.encode())
        raw_digest.update(encoded)
        payload = json.loads(gzip.decompress(encoded))
        require(payload["scenario"] == asdict(scenario), "Retained execution has different scenario parameters")
        feedback = QueryFeedback.from_dict(payload["feedback"])
        require(feedback.scenario_id == scenario.scenario_id, "Feedback/candidate identity mismatch")
        attempt = attempts[payload["call"]]
        require(attempt["status"] == "complete" and attempt["scenario_id"] == scenario.scenario_id, "Retained feedback has no matching completed physical call")
        require(close(attempt["execution_s"], feedback.execution_s), "Retained feedback execution cost differs from attempt")
        retained_calls.add(payload["call"])
        retained_execution_s += feedback.execution_s
        trajectory = payload["trajectory"]
        require(bool(trajectory) and close(trajectory[0]["time_s"], 0), "Missing initial world")
        times = np.asarray([r["time_s"] for r in trajectory])
        require(np.allclose(np.diff(times), dt, atol=1e-9) and times[-1] <= DEFAULT.episode_duration_s + 1e-9, "Physical timebase/horizon mismatch")
        require(feedback.collision == any(r["collision"] for r in trajectory), "Collision label disagrees with logged simulator events")
        if not feedback.collision and not feedback.safety.violations:
            require(close(times[-1], DEFAULT.episode_duration_s), "Collision-free execution stopped before the registered horizon")
        for position, frame in enumerate(trajectory):
            requested = frame["requested_control"]
            require(-8 - 1e-9 <= requested["acceleration"] <= 3 + 1e-9 and abs(requested["steering"]) <= 0.45 + 1e-9, "Requested control exceeds common actuator limits")
            if position and requested != trajectory[position - 1]["requested_control"]:
                require((position - 1) % interval == 0, "SUT command changed outside the common decision ticks")
        initial_worlds.update(json.dumps(trajectory[0]["actors"], sort_keys=True).encode())
        labels.append(feedback.collision)
        safety.append([feedback.safety.min_clearance_m, feedback.safety.min_ttc_s])
    require(len(retained_calls) == candidate_count, "Retained coordinates reused a physical attempt")
    require(np.array_equal(labels, archive.collision), "Archive collision labels disagree with paid physical feedback")
    require(np.array_equal(safety, archive.safety), "Archive safety measurements disagree with paid physical feedback")
    return {"source_id": sut.sut_id, "family": sut.family, "role": archive.role, "scenario": kind,
            "responses": candidate_count, "collisions": sum(labels), "actual_physical_calls": len(attempts),
            "successful_physical_calls": len(completed), "successful_calls_without_retained_response": len(completed) - len(retained_calls),
            "failed_physical_calls": len(failed), "interrupted_unknown_duration_calls": len(interrupted),
            "scheduled_without_started_attempt": len(scheduled) - len(attempts),
            "physical_execution_s": manifest["physical_execution_s"], "retained_response_execution_s": retained_execution_s,
            "initial_worlds_sha256": initial_worlds.hexdigest(), "raw_execution_files_sha256": raw_digest.hexdigest(),
            "archive_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    verified, pending, errors = [], [], []
    for sut in SUTS:
        if sut.role not in ("history", "development"):
            continue
        for kind in DOMAINS:
            path = DATA_ROOT / sut.role / kind / f"{sut.sut_id}.npz"
            if not path.exists():
                pending.append(str(path.relative_to(ROOT)))
                continue
            if not zipfile.is_zipfile(path) and time.time() - path.stat().st_mtime < 60:
                pending.append(str(path.relative_to(ROOT)))
                continue
            try:
                verified.append(verify_archive(path))
            except (zipfile.BadZipFile, EOFError, OSError) as error:
                # Archive writers may have created the file without closing its ZIP directory yet.
                if time.time() - path.stat().st_mtime < 60:
                    pending.append(str(path.relative_to(ROOT)))
                else:
                    errors.append({"path": str(path.relative_to(ROOT)), "error": repr(error)})
            except Exception as error:
                errors.append({"path": str(path.relative_to(ROOT)), "error": repr(error)})
    groups = defaultdict(set)
    for row in verified:
        groups[(row["role"], row["scenario"])].add(row["initial_worlds_sha256"])
    if any(len(values) != 1 for values in groups.values()):
        errors.append({"error": "Same-role same-coordinate SUTs faced different initial worlds"})
    report = {"status": "failed" if errors else "partial" if pending else "complete",
              "scope": "Historical/development archives and actual raw execution/cost logs only; final comparisons are separate.",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "additional_physical_calls": 0,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "verified_archives": len(verified), "expected_archives": 48,
              "verified_responses": sum(r["responses"] for r in verified),
              "banks": verified, "pending": pending, "errors": errors}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(report["status"], len(verified), "/ 48 archives;", report["verified_responses"], "reconstructed responses", flush=True)
    if errors or arguments.require_complete and pending:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
