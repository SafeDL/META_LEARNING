"""Apply the full independent raw-result verifier to a completed registered cohort."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.audit_collision_results import RawEvidence, require, verify_run
from benchmark.evaluate_collision import task_path
from benchmark.protocol import DATA_ROOT, ROOT, method_contract, validate_sut_freeze
from benchmark.studies import Task, tasks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.perf_counter()
    registration = json.loads(arguments.registration.read_text(encoding="utf-8"))
    frozen_path = DATA_ROOT / "frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    validate_sut_freeze(frozen)
    contract = method_contract()
    require(frozen["source_contract"] == registration["source_contract"] == contract,
            "Cohort uses another frozen evaluator")
    require(registration["protocol_sha256"] == sha(frozen_path), "Cohort protocol changed")
    require(registration["original_api"] == "benchmark.evaluate_collision.evaluate", "Unexpected cohort execution API")
    plan = tasks()
    require(frozen["study_tasks"] == [asdict(task) for task in plan], "Full10880-task matrix changed")
    selected = [Task(**row) for row in registration["tasks"]]
    require(len(set(selected)) == len(selected) == registration["expected_runs"] and set(selected) <= set(plan),
            "Cohort duplicates tasks or differs from the full frozen matrix")
    for artifact in frozen["method"]["models"]:
        require(sha(ROOT / artifact["path"]) == artifact["sha256"], "A frozen model changed")
    raw, verified, pending, errors = RawEvidence(), [], [], []
    for task in selected:
        path = task_path(task)
        if not path.exists():
            pending.append(str(path))
            continue
        try:
            verified.append(verify_run(path, task, raw))
        except Exception as error:
            errors.append({"path": str(path), "error": repr(error)})
        if (len(verified) + len(errors)) % 24 == 0:
            print("independently verified registered cohort", len(verified), "/", len(selected), flush=True)
    require(frozen["source_contract"] == method_contract(), "Frozen evaluator changed during cohort auditing")
    result = {"status": "failed" if errors else "verified_complete_registered_cohort" if not pending else "partial",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "expected_runs": len(selected),
              "verified_runs": len(verified), "whole_native_matrix_runs": len(plan), "full_native_matrix_complete": False,
              "registration": str(arguments.registration), "registration_sha256": sha(arguments.registration),
              "protocol_sha256": sha(frozen_path), "source_contract": contract,
              "validator_sha256": sha(ROOT / "benchmark/audit_collision_results.py"),
              "cohort_validator_sha256": sha(Path(__file__)), "model_freeze_checked": True,
              "unique_raw_executions_checked": len(raw.executions), "additional_physical_calls": 0,
              "elapsed_s": time.perf_counter() - started, "runs": verified, "pending": pending, "errors": errors}
    output = arguments.registration.with_name("raw_cohort_verification.json")
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(result["status"], len(verified), "/", len(selected), ";", len(errors), "errors", flush=True)
    raise SystemExit(0 if not errors and not pending else 1)


if __name__ == "__main__":
    main()
