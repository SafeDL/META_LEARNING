"""Collect the fixed ordinal81-88 final banks via the unchanged paid executor."""

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

from benchmark.collect import require_learning_freeze
from benchmark.executions import collect_bank
from benchmark.protocol import DATA_ROOT, DEFAULT, method_contract, validate_sut_freeze
from benchmark.scenarios import DOMAINS, sample
from controllers.highway.catalog import SUTS
from evaluate_fixed_cohort import write_json


OUTPUT = Path(__file__).with_name("parallel_target_batches") / "lane_change_rear_new"


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    require_learning_freeze()
    frozen_path = DATA_ROOT / "frozen_protocol.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    validate_sut_freeze(frozen)
    contract = method_contract()
    if frozen["source_contract"] != contract or frozen["protocol"] != json.loads(json.dumps(asdict(DEFAULT))):
        raise ValueError("Final execution configuration differs from the frozen protocol")
    plan = [(sut, kind, condition, env) for kind in DOMAINS for condition in DEFAULT.conditions
            for env in DEFAULT.environment_seeds for sut in SUTS
            if sut.role == "target" and kind in sut.supported_scenarios]
    assigned = plan[80:88]
    if len(plan) != 96 or len(assigned) != 8 or any(kind != "lane_change_rear" or condition != "new"
                                                   for _, kind, condition, _ in assigned):
        raise ValueError("The fixed ordinal81-88 assignment changed")
    def root_for(task):
        sut, kind, condition, env = task
        return DATA_ROOT / "targets" / sut.sut_id / condition / kind / f"env_{env}"
    if any(root_for(task).exists() for task in assigned):
        raise ValueError("All assigned banks must be untouched at admission")
    prefix_stop = next((i for i, task in enumerate(plan) if not (root_for(task) / "manifest.json").exists()), len(plan))
    if prefix_stop > 48:
        raise ValueError("Native acquisition is too advanced for this future batch")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registration = {"status": "admitted", "pid": os.getpid(), "started_utc": now(),
                    "protocol_sha256": hashlib.sha256(frozen_bytes).hexdigest(), "source_contract": contract,
                    "dispatcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "metadata_writer_sha256": hashlib.sha256(Path(__file__).with_name("evaluate_fixed_cohort.py").read_bytes()).hexdigest(),
                    "original_api": "benchmark.executions.collect_bank", "workers_per_bank": 12,
                    "assigned_ordinals": list(range(81, 89)), "contiguous_started_prefix_at_admission": prefix_stop,
                    "all_assigned_paths_absent_at_admission": True,
                    "tasks": [{"sut_id": sut.sut_id, "kind": kind, "condition": condition,
                               "environment_seed": env, "bank": str(root_for(task))}
                              for task in assigned for sut, kind, condition, env in [task]]}
    with (OUTPUT / "registration.json").open("x", encoding="utf-8") as stream:
        json.dump(registration, stream, indent=2, allow_nan=False)
    rows, started = [], time.perf_counter()
    try:
        for task in assigned:
            sut, kind, condition, env = task
            if frozen_path.read_bytes() != frozen_bytes or method_contract() != contract:
                raise ValueError("Frozen inputs changed during physical acquisition")
            if any((root_for(guard) / "manifest.json").exists() for guard in plan[72:80]):
                raise ValueError("Native acquisition entered the eight-bank guard region")
            destination = root_for(task)
            if destination.exists():
                raise ValueError("Another owner created an assigned bank")
            bank_started = time.perf_counter()
            bank = collect_bank(sut.sut_id, sample(kind, DEFAULT.candidate_count, condition), destination,
                                environment_seed=env, workers=12)
            if bank.manifest["candidate_count"] != 512 or bank.manifest["failed_or_interrupted_calls"]:
                raise ValueError("A batch bank is incomplete or contains an unsuccessful attempt")
            rows.append({"sut_id": sut.sut_id, "bank": str(destination), "finished_utc": now(),
                         "elapsed_s": time.perf_counter() - bank_started,
                         "actual_physical_calls": bank.manifest["attempted_calls"],
                         "physical_execution_s": bank.manifest["physical_execution_s"],
                         "manifest_sha256_at_completion": hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest()})
            write_json(OUTPUT / "progress.json", {"status": "running", "pid": os.getpid(), "observed_at_utc": now(),
                       "completed_banks": len(rows), "expected_banks": 8, "banks": rows})
            print("fixed rear/new acquisition", len(rows), "/8 banks", flush=True)
        write_json(OUTPUT / "progress.json", {"status": "complete", "pid": os.getpid(), "finished_utc": now(),
                   "completed_banks": len(rows), "expected_banks": 8, "elapsed_s": time.perf_counter() - started,
                   "banks": rows}, critical=True)
    except Exception as error:
        print("FAILED fixed rear/new acquisition", repr(error), traceback.format_exc(), flush=True)
        write_json(OUTPUT / "progress.json", {"status": "failed", "pid": os.getpid(), "finished_utc": now(),
                   "completed_banks": len(rows), "expected_banks": 8, "error": repr(error), "banks": rows})
        raise


if __name__ == "__main__":
    main()
