"""Precompute180 frozen late-ordered source-swap tasks, without changing any task."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.evaluate_collision import evaluate, task_path
from benchmark.protocol import DATA_ROOT, RESULT_ROOT, ROOT, method_contract, validate_sut_freeze
from benchmark.studies import tasks
from evaluate_fixed_cohort import write_json


OUTPUT = Path(__file__).with_name("late_source_swaps") / "front_brake_learning"


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    predecessor_path = Path(__file__).with_name("fixed_cohorts") / "cut_in_reuse/progress.json"
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor["status"] != "complete" or predecessor["completed_runs"] != 640 or predecessor["failed_runs"]:
        raise ValueError("The previous640-run supplementary cohort must finish first")
    frozen_path = DATA_ROOT / "frozen_protocol.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    validate_sut_freeze(frozen)
    contract = method_contract()
    if frozen["source_contract"] != contract:
        raise ValueError("Frozen evaluator source inputs changed")
    for artifact in frozen["method"]["models"]:
        if hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("A frozen model changed")
    plan = tasks()
    if frozen["study_tasks"] != [asdict(task) for task in plan]:
        raise ValueError("The full registered10880-run matrix changed")
    cohort = [task for task in plan if task.study == "source_swap" and task.scenario == "front_brake"
              and task.condition == "new" and task.family == "learning"]
    if len(cohort) != 180 or any(task_path(task).exists() for task in cohort):
        raise ValueError("Admission requires the untouched fixed180-task late source-swap cohort")
    first = min(plan.index(task) for task in cohort)
    missing_before = sum(not task_path(task).exists() for task in plan[:first])
    if missing_before < 4000:
        raise ValueError("Native evaluation is too close to the assigned late task ordinals")
    state = json.loads((RESULT_ROOT / "pipeline/state.json").read_text(encoding="utf-8"))
    if state["status"] != "running" or state["stages"]["targets"]["status"] != "running" or "finite_comparisons" in state["stages"]:
        raise ValueError("Early source-swap admission requires native target acquisition to remain active")
    for task in cohort:
        root = DATA_ROOT / "targets" / task.target_id / task.condition / task.scenario / f"env_{task.environment_seed}"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["contract"] != frozen["sut_contracts"][task.target_id]:
            raise ValueError("A required paid target bank is incomplete or has another contract")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registration = {"status": "admitted", "pid": os.getpid(), "started_utc": now(),
                    "protocol_sha256": hashlib.sha256(frozen_bytes).hexdigest(), "source_contract": contract,
                    "dispatcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "original_api": "benchmark.evaluate_collision.evaluate", "workers": 4,
                    "expected_runs": 180, "first_registered_task_ordinal": first + 1,
                    "unexecuted_earlier_native_tasks_at_admission": missing_before,
                    "predecessor": str(predecessor_path), "predecessor_sha256": hashlib.sha256(predecessor_path.read_bytes()).hexdigest(),
                    "tasks": [asdict(task) for task in cohort], "additional_physical_calls": 0,
                    "selection_basis": "Fixed late registry order; not selected using discovery outcomes.",
                    "full_native_matrix_unchanged": True}
    with (OUTPUT / "registration.json").open("x", encoding="utf-8") as stream:
        json.dump(registration, stream, indent=2, allow_nan=False)
    rows, errors, started = [], [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(evaluate, task): task for task in cohort}
        for future in as_completed(pending):
            task = pending[future]
            try:
                row = future.result()
                row["artifact_sha256"] = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
                rows.append(row)
            except Exception as error:
                failure = {"task": asdict(task), "error": repr(error), "traceback": traceback.format_exc()}
                errors.append(failure)
                print("FAILED late source swap", failure, flush=True)
            write_json(OUTPUT / "progress.json", {"status": "running", "pid": os.getpid(), "observed_at_utc": now(),
                       "completed_runs": len(rows), "failed_runs": len(errors), "expected_runs": 180,
                       "runs": rows, "errors": errors})
            if (len(rows) + len(errors)) % 12 == 0:
                print("late source swaps", len(rows), "/180 completed;", len(errors), "failures", flush=True)
    if frozen_path.read_bytes() != frozen_bytes or method_contract() != contract:
        raise ValueError("Frozen evaluator inputs changed during source replacement evaluation")
    write_json(OUTPUT / "progress.json", {"status": "failed" if errors else "complete", "pid": os.getpid(),
               "finished_utc": now(), "elapsed_s": time.perf_counter() - started,
               "completed_runs": len(rows), "failed_runs": len(errors), "expected_runs": 180,
               "runs": rows, "errors": errors}, critical=True)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
