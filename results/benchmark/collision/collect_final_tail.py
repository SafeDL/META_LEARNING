"""Collect the fixed last eight final banks with the unchanged paid executor."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.collect import require_learning_freeze
from benchmark.executions import collect_bank
from benchmark.protocol import DATA_ROOT, DEFAULT, ROOT, method_contract, validate_sut_freeze
from benchmark.scenarios import DOMAINS, sample
from controllers.highway.catalog import SUTS


OUTPUT = Path(__file__).with_name("parallel_target_tail")


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    require_learning_freeze()
    frozen_path = DATA_ROOT / "frozen_protocol.json"
    protocol_bytes = frozen_path.read_bytes()
    frozen = json.loads(protocol_bytes)
    validate_sut_freeze(frozen)
    contract = method_contract()
    if frozen["source_contract"] != contract or frozen["protocol"] != json.loads(json.dumps(asdict(DEFAULT))):
        raise ValueError("Final protocol differs from the native collector")
    plan = [(sut, kind, condition, env)
            for kind in DOMAINS for condition in DEFAULT.conditions for env in DEFAULT.environment_seeds
            for sut in SUTS if sut.role == "target" and kind in sut.supported_scenarios]
    tail = plan[-8:]
    if len(plan) != 96 or any(kind != "lane_change_rear" or condition != "hard" for _, kind, condition, _ in tail):
        raise ValueError("The fixed final tail differs from the preregistered eight banks")
    def root_for(task):
        sut, kind, condition, env = task
        return DATA_ROOT / "targets" / sut.sut_id / condition / kind / f"env_{env}"
    if any(root_for(task).exists() for task in tail):
        raise ValueError("Tail admission requires every assigned bank to be untouched")
    existing = [i for i, task in enumerate(plan) if (root_for(task) / "manifest.json").exists()]
    if not existing or max(existing) >= 24:
        raise ValueError("Native collection is too close to the assigned tail")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registration = {"status": "admitted", "pid": os.getpid(), "started_utc": now(),
                    "source_contract": contract, "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                    "dispatcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "native_last_started_ordinal_at_admission": max(existing) + 1,
                    "assigned_ordinals": list(range(89, 97)), "workers_per_bank": 12,
                    "execution_api": "benchmark.executions.collect_bank", "physical_calls_per_bank": 512,
                    "all_assigned_paths_absent_at_admission": True,
                    "tasks": [{"sut_id": sut.sut_id, "kind": kind, "condition": condition,
                               "environment_seed": env, "bank": str(root_for(task))}
                              for task in tail for sut, kind, condition, env in [task]]}
    path = OUTPUT / "registration.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(registration, stream, indent=2, allow_nan=False)
    rows = []
    started = time.perf_counter()
    try:
        for task in tail:
            sut, kind, condition, env = task
            if frozen_path.read_bytes() != protocol_bytes or method_contract() != contract:
                raise ValueError("Frozen source inputs changed during final tail acquisition")
            # Keep a large separation from the native sequential collector. This
            # helper never claims an already-created path or executes a bank twice.
            if any((root_for(front) / "manifest.json").exists() for front in plan[72:88]):
                raise ValueError("Native collector entered the tail guard region")
            root = root_for(task)
            if root.exists():
                raise ValueError("Assigned bank was created by another owner")
            bank_started = time.perf_counter()
            bank = collect_bank(sut.sut_id, sample(kind, DEFAULT.candidate_count, condition), root,
                                environment_seed=env, workers=12)
            if bank.manifest["candidate_count"] != 512 or bank.manifest["failed_or_interrupted_calls"]:
                raise ValueError("Tail bank is incomplete or contains unsuccessful attempts")
            rows.append({"sut_id": sut.sut_id, "bank": str(root), "finished_utc": now(),
                         "elapsed_s": time.perf_counter() - bank_started,
                         "actual_physical_calls": bank.manifest["attempted_calls"],
                         "physical_execution_s": bank.manifest["physical_execution_s"],
                         "manifest_sha256_at_completion": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()})
            write(OUTPUT / "progress.json", {"status": "running", "pid": os.getpid(), "completed_banks": len(rows),
                                            "expected_banks": 8, "observed_at_utc": now(), "banks": rows})
            print("native final tail", len(rows), "/ 8", flush=True)
        write(OUTPUT / "progress.json", {"status": "complete", "pid": os.getpid(), "completed_banks": len(rows),
                                        "expected_banks": 8, "finished_utc": now(),
                                        "elapsed_s": time.perf_counter() - started, "banks": rows})
    except Exception as error:
        write(OUTPUT / "progress.json", {"status": "failed", "pid": os.getpid(), "completed_banks": len(rows),
                                        "expected_banks": 8, "finished_utc": now(), "error": repr(error), "banks": rows})
        raise


if __name__ == "__main__":
    main()
