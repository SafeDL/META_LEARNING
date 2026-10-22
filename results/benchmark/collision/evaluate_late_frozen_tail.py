"""Precompute the untouched fixed late matrix with the original evaluator."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
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

START_ORDINAL = 8001
WORKERS = 4


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base = Path(__file__).parent
    with (base / "late_frozen_tail.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        execute(base)


def execute(base):
    predecessors = []
    for name in ("front_brake_learning", "cut_in_learning", "lane_change_rear_learning",
                 "front_brake_search", "cut_out_search", "lane_change_rear_search", "cut_out_learning"):
        path = base / "late_source_swaps" / name / "progress.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["status"] != "complete" or report["completed_runs"] != 180 or report["failed_runs"]:
            raise ValueError("All preceding auxiliary cohorts must be terminal and complete")
        predecessors.append({"path": str(path), "sha256": sha(path)})
    protocol_path = DATA_ROOT / "frozen_protocol.json"
    frozen_bytes = protocol_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    validate_sut_freeze(frozen)
    contract = method_contract()
    if frozen["source_contract"] != contract:
        raise ValueError("Frozen evaluator changed")
    for artifact in frozen["method"]["models"]:
        if sha(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError("Frozen model changed")
    plan = tasks()
    if frozen["study_tasks"] != [asdict(task) for task in plan] or len(plan) != 10880:
        raise ValueError("Full registered matrix changed")
    selected = [task for task in plan[START_ORDINAL - 1:] if not task_path(task).exists()]
    if not selected:
        raise ValueError("No untouched fixed late tasks remain")
    first = min(plan.index(task) for task in selected)
    missing_before = sum(not task_path(task).exists() for task in plan[:first])
    if missing_before < 4000:
        raise ValueError("Native evaluation is too close to the fixed late matrix")
    state = json.loads((RESULT_ROOT / "pipeline/state.json").read_text(encoding="utf-8"))
    if state["status"] != "running" or state["stages"]["finite_comparisons"]["status"] != "running":
        raise ValueError("The native finite evaluator must already be running")
    for task in selected:
        bank = DATA_ROOT / "targets" / task.target_id / task.condition / task.scenario / f"env_{task.environment_seed}"
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["contract"] != frozen["sut_contracts"][task.target_id]:
            raise ValueError("A target bank is incomplete or changed")
    output = base / "late_frozen_tail"
    output.mkdir(exist_ok=True)
    registration = {
        "status": "admitted", "pid": os.getpid(), "started_utc": now(), "workers": WORKERS,
        "protocol_sha256": hashlib.sha256(frozen_bytes).hexdigest(), "source_contract": contract,
        "dispatcher_sha256": sha(__file__), "original_api": "benchmark.evaluate_collision.evaluate",
        "fixed_tail_start_ordinal": START_ORDINAL, "first_registered_task_ordinal": first + 1,
        "unexecuted_earlier_native_tasks_at_admission": missing_before,
        "expected_runs": len(selected), "tasks": [asdict(task) for task in selected],
        "predecessors": predecessors, "additional_physical_calls": 0,
        "selection_basis": "Every untouched task at or after fixed ordinal8001; completion status only, no outcome selection.",
        "full_native_matrix_unchanged": True,
    }
    with (output / "registration.json").open("x", encoding="utf-8") as stream:
        json.dump(registration, stream, indent=2, allow_nan=False)
    started, rows, errors = time.perf_counter(), [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        pending = {pool.submit(evaluate, task): task for task in selected}
        for future in as_completed(pending):
            task = pending[future]
            try:
                row = future.result()
                row["artifact_sha256"] = sha(row["path"])
                rows.append(row)
            except Exception as error:
                failure = {"task": asdict(task), "error": repr(error), "traceback": traceback.format_exc()}
                errors.append(failure)
                print("FAILED fixed late task", failure, flush=True)
            write_json(output / "progress.json", {
                "status": "running", "pid": os.getpid(), "observed_at_utc": now(),
                "completed_runs": len(rows), "failed_runs": len(errors), "expected_runs": len(selected),
                "runs": rows, "errors": errors})
            if (len(rows) + len(errors)) % 16 == 0:
                print("fixed late matrix", len(rows), "/", len(selected), ";", len(errors), "failures", flush=True)
    if protocol_path.read_bytes() != frozen_bytes or method_contract() != contract:
        raise ValueError("Frozen evaluator changed during late matrix execution")
    write_json(output / "progress.json", {
        "status": "failed" if errors else "complete", "pid": os.getpid(), "finished_utc": now(),
        "elapsed_s": time.perf_counter() - started, "completed_runs": len(rows), "failed_runs": len(errors),
        "expected_runs": len(selected), "runs": rows, "errors": errors}, critical=True)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
