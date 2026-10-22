"""Schedule a complete frozen overall/strict-family cohort with per-failure receipts."""

import argparse
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
from benchmark.protocol import DATA_ROOT, DEFAULT, RESULT_ROOT, ROOT, method_contract, validate_sut_freeze
from benchmark.scenarios import DOMAINS
from benchmark.studies import tasks


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, *, critical=False):
    temporary = path.with_suffix(".tmp")
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    for attempt in range(25):
        try:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)
            return True
        except PermissionError as error:
            if attempt == 24:
                if critical:
                    raise
                print("auxiliary progress write failed", str(path), repr(error), flush=True)
                return False
            time.sleep(.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=DOMAINS, required=True)
    parser.add_argument("--condition", choices=DEFAULT.conditions, required=True)
    arguments = parser.parse_args()
    frozen_path = DATA_ROOT / "frozen_protocol.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    validate_sut_freeze(frozen)
    contract = method_contract()
    if frozen["source_contract"] != contract or frozen["protocol"] != json.loads(json.dumps(asdict(DEFAULT))):
        raise ValueError("Final configuration or evaluator changed after protocol freezing")
    for artifact in frozen["method"]["models"]:
        if hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("A frozen model changed")
    plan = tasks()
    if frozen["study_tasks"] != [asdict(task) for task in plan]:
        raise ValueError("The full final task matrix changed")
    cohort = [task for task in plan if task.scenario == arguments.scenario and task.condition == arguments.condition
              and task.study in ("overall", "strict_family")]
    if len(cohort) != 640:
        raise ValueError("A complete frozen overall/strict-family cohort requires640 runs")
    state = json.loads((RESULT_ROOT / "pipeline/state.json").read_text(encoding="utf-8"))
    if state["status"] != "running" or state["stages"]["targets"]["status"] != "running" or "finite_comparisons" in state["stages"]:
        raise ValueError("Native final comparison execution must not have started")
    if any(DATA_ROOT.glob("targets/*/*/cut_out/env_*/manifest.json")):
        raise ValueError("Native target collection is too advanced for early-cohort admission")
    for task in cohort:
        root = DATA_ROOT / "targets" / task.target_id / task.condition / task.scenario / f"env_{task.environment_seed}"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["contract"] != frozen["sut_contracts"][task.target_id]:
            raise ValueError("A required physical target bank is unfinished or belongs to another frozen object")
    # Earlier condition cohorts remain cached inputs to the untouched full native
    # runner. This helper never produces or claims the global10880-run index.
    output = Path(__file__).with_name("fixed_cohorts") / f"{arguments.scenario}_{arguments.condition}"
    output.mkdir(parents=True, exist_ok=True)
    cached = [str(task_path(task)) for task in cohort if task_path(task).exists()]
    registration = {"status": "admitted", "pid": os.getpid(), "started_utc": now(),
                    "protocol_sha256": hashlib.sha256(frozen_bytes).hexdigest(), "source_contract": contract,
                    "dispatcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "original_api": "benchmark.evaluate_collision.evaluate", "workers": 4,
                    "expected_runs": 640, "tasks": [asdict(task) for task in cohort],
                    "cached_paths_at_admission": cached, "additional_physical_calls": 0,
                    "full_native_matrix_unchanged": True}
    with (output / "registration.json").open("x", encoding="utf-8") as stream:
        json.dump(registration, stream, indent=2, allow_nan=False)
    rows, errors, started = [], [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(evaluate, task): (index, task) for index, task in enumerate(cohort)}
        for future in as_completed(pending):
            index, task = pending[future]
            try:
                row = future.result()
                row["artifact_sha256"] = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
                rows.append(row)
            except Exception as error:
                failure = {"task": asdict(task), "observed_at_utc": now(), "error": repr(error),
                           "traceback": traceback.format_exc()}
                errors.append(failure)
                print("FAILED frozen task", index, repr(error), failure["traceback"], flush=True)
                (output / "errors").mkdir(exist_ok=True)
                write_json(output / "errors" / f"{index:06d}.json", failure)
            write_json(output / "progress.json", {"status": "running", "pid": os.getpid(),
                       "observed_at_utc": now(), "completed_runs": len(rows), "expected_runs": 640,
                       "failed_runs": len(errors), "runs": rows, "errors": errors})
            if (len(rows) + len(errors)) % 16 == 0:
                print("fixed cohort", arguments.scenario, arguments.condition,
                      len(rows), "/640 completed;", len(errors), "failures", flush=True)
    if frozen_path.read_bytes() != frozen_bytes or method_contract() != contract:
        raise ValueError("Frozen inputs changed during the final cohort")
    write_json(output / "progress.json", {"status": "failed" if errors else "complete", "pid": os.getpid(),
               "finished_utc": now(), "elapsed_s": time.perf_counter() - started,
               "completed_runs": len(rows), "expected_runs": 640, "failed_runs": len(errors),
               "runs": rows, "errors": errors}, critical=True)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
