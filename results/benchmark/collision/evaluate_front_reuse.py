"""Execute all 640 registered front-brake/reuse comparisons without changing them."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.evaluate_collision import evaluate, task_path
from benchmark.protocol import DATA_ROOT, RESULT_ROOT, ROOT, method_contract, validate_sut_freeze
from benchmark.studies import tasks


OUTPUT = Path(__file__).with_name("front_reuse_comparisons")


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    previous_root = Path(__file__).with_name("first_bank_comparisons")
    previous = json.loads((previous_root / "progress.json").read_text(encoding="utf-8"))
    audit = json.loads((previous_root / "raw_result_audit.json").read_text(encoding="utf-8"))
    if previous["status"] != "complete" or previous["completed_runs"] != 80 or audit["verified_runs"] != 80 or audit["errors"]:
        raise ValueError("The first 80 actual final runs must pass independent raw-result verification")
    frozen_path = DATA_ROOT / "frozen_protocol.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    validate_sut_freeze(frozen)
    contract = method_contract()
    if frozen["source_contract"] != contract:
        raise ValueError("Current final evaluator differs from the frozen source inputs")
    for model in frozen["method"]["models"]:
        if hashlib.sha256((ROOT / model["path"]).read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError("A frozen model changed")
    plan = tasks()
    if frozen["study_tasks"] != [asdict(task) for task in plan]:
        raise ValueError("The full registered task matrix changed")
    cohort = [task for task in plan if task.scenario == "front_brake" and task.condition == "reuse"]
    if len(cohort) != 640:
        raise ValueError("The fixed full front-brake/reuse cohort changed")
    existing = {str(task_path(task)) for task in cohort if task_path(task).exists()}
    previous_paths = {row["path"] for row in previous["runs"]}
    if existing != previous_paths:
        raise ValueError("Only the verified first 80 runs may precede this cohort")
    state = json.loads((RESULT_ROOT / "pipeline/state.json").read_text(encoding="utf-8"))
    if state["status"] != "running" or state["stages"]["targets"]["status"] != "running" or "finite_comparisons" in state["stages"]:
        raise ValueError("The native final evaluator must not yet be scheduled")
    if any(DATA_ROOT.glob("targets/*/*/cut_out/env_*/manifest.json")):
        raise ValueError("Native acquisition is too advanced for early cohort admission")
    for task in cohort:
        bank = DATA_ROOT / "targets" / task.target_id / task.condition / task.scenario / f"env_{task.environment_seed}/manifest.json"
        if json.loads(bank.read_text(encoding="utf-8"))["status"] != "complete":
            raise ValueError("A cohort physical bank is incomplete")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registration = {"status": "admitted", "pid": os.getpid(), "started_utc": now(),
                    "protocol_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
                    "dispatcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "source_contract": contract, "workers": 4, "original_api": "benchmark.evaluate_collision.evaluate",
                    "additional_physical_calls": 0, "previous_verified_runs_reused": 80,
                    "tasks": [asdict(task) for task in cohort],
                    "note": "Full fixed cohort selected by scenario/condition before looking at discovery outcomes; all 10880 native tasks remain registered."}
    with (OUTPUT / "registration.json").open("x", encoding="utf-8") as stream:
        json.dump(registration, stream, indent=2, allow_nan=False)
    rows, started = [], time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(evaluate, task): task for task in cohort}
            for future in as_completed(pending):
                row = future.result()
                row["artifact_sha256"] = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
                rows.append(row)
                write(OUTPUT / "progress.json", {"status": "running", "pid": os.getpid(), "observed_at_utc": now(),
                                                "completed_runs": len(rows), "expected_runs": 640, "runs": rows})
                if len(rows) % 16 == 0 or len(rows) == 640:
                    print("front-brake/reuse final comparisons", len(rows), "/ 640", flush=True)
        if frozen_path.read_bytes() != frozen_bytes or method_contract() != contract:
            raise ValueError("Frozen inputs changed during the final cohort")
        write(OUTPUT / "progress.json", {"status": "complete", "pid": os.getpid(), "finished_utc": now(),
                                        "completed_runs": len(rows), "expected_runs": 640,
                                        "elapsed_s": time.perf_counter() - started, "runs": rows})
    except Exception as error:
        write(OUTPUT / "progress.json", {"status": "failed", "pid": os.getpid(), "finished_utc": now(),
                                        "completed_runs": len(rows), "expected_runs": 640,
                                        "error": repr(error), "runs": rows})
        raise


if __name__ == "__main__":
    main()
