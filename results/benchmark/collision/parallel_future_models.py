"""Populate independent future-scenario model caches using the original trainers.

The original pipeline keeps owning front_brake. Typed reservations prevent its
builder from starting a duplicate future model. Incomplete reservations are
deliberately rejected by the existing builder and final freeze gates.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from threadpoolctl import threadpool_limits

from benchmark.build_collision_models import model_path
from benchmark.construction import load_construction
from benchmark.protocol import ROOT, method_contract
from benchmark.scenarios import DOMAINS
from benchmark.studies import SCOPES
from methods.baselines.fsbo.method import CONFIG as FSBO_CONFIG, train as train_fsbo
from methods.baselines.pooled_ft.method import CONFIG as POOLED_CONFIG, train as train_pooled
from methods.common.model_contract import validate_model_contract
from methods.common.response import archive_fingerprints
from methods.gcrt.config import CONFIG
from methods.gcrt.train import train as train_main


OUTPUT = Path(__file__).with_name("parallel_models")
CONFIGURATIONS = {"history_response": asdict(CONFIG), "fsbo": FSBO_CONFIG, "pooled_ft": POOLED_CONFIG}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def methods(scope):
    return ["history_response"] if scope.name.startswith("swap_") else ["history_response", "fsbo", "pooled_ft"]


def front_pending():
    missing = []
    for scope in SCOPES.values():
        for method in methods(scope):
            path = model_path(method, scope.name, "front_brake").with_suffix(".json")
            complete = False
            if path.exists():
                try:
                    complete = json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
                except json.JSONDecodeError:
                    pass
            if not complete:
                missing.append(str(path))
    return missing


def check_model(path, method, scope, history, development):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    validate_model_contract(checkpoint, method, configuration=CONFIGURATIONS[method])
    assert report["status"] == "complete" and report["curve"][-1]["step"] == CONFIGURATIONS[method]["steps"] == 1500
    assert checkpoint["held_family"] == scope.held_family
    assert checkpoint["history_fingerprints"] == archive_fingerprints(history)
    assert checkpoint["development_fingerprints"] == archive_fingerprints(development)
    assert set(checkpoint["history_ids"]) == {a.source_id for a in history}
    assert set(checkpoint["development_ids"]) == {a.source_id for a in development}
    return checkpoint, report


def train_job(job):
    started = time.perf_counter()
    attempt_path = OUTPUT / "attempts" / f"{job['id']}.json"
    receipt = {"job": job, "pid": os.getpid(), "status": "running", "started_utc": utc(),
               "additional_physical_calls": 0}
    write_json(attempt_path, receipt)
    log_path = OUTPUT / "logs" / f"{job['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(1)
        assert method_contract() == job["source_contract"]
        scope = SCOPES[job["scope"]]
        all_history, all_development = load_construction(job["scenario"])
        history = [a for a in all_history if a.source_id in scope.history_ids]
        development = [a for a in all_development if a.source_id in scope.development_ids]
        destination = OUTPUT / "staging" / job["scope"] / job["method"] / f"{job['scenario']}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        trainer = {"history_response": train_main, "fsbo": train_fsbo, "pooled_ft": train_pooled}[job["method"]]
        with log_path.open("w", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream), threadpool_limits(1):
            trainer(history, development, destination, held_family=scope.held_family)
        checkpoint, report = check_model(destination, job["method"], scope, history, development)
        assert method_contract() == job["source_contract"]
        final = Path(job["destination"])
        marker = torch.load(final, weights_only=True, map_location="cpu")
        assert marker == {"format": "collision-model-reservation", "dispatcher_id": job["dispatcher_id"], "job_id": job["id"]}
        assert not final.with_suffix(".json").exists()
        assert final.resolve().is_relative_to((ROOT / "results").resolve())
        assert destination.resolve().is_relative_to((OUTPUT / "staging").resolve())
        # Both files are individually atomic. The pipeline refuses an intermediate
        # state and can resume after this dispatcher finishes if it catches up.
        destination.with_suffix(".json").replace(final.with_suffix(".json"))
        destination.replace(final)
        receipt.update(status="complete", finished_utc=utc(), wall_s=time.perf_counter() - started,
                       training_elapsed_s=report["elapsed_s"], steps=1500,
                       checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                       training_report_sha256=hashlib.sha256(final.with_suffix(".json").read_bytes()).hexdigest(),
                       history_ids=checkpoint["history_ids"], development_ids=checkpoint["development_ids"],
                       log=str(log_path))
    except Exception as error:
        receipt.update(status="failed", finished_utc=utc(), wall_s=time.perf_counter() - started,
                       error=repr(error), traceback=traceback.format_exc(), log=str(log_path))
    write_json(attempt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    assert 1 <= arguments.workers <= 8
    started = time.perf_counter()
    state = json.loads((ROOT / "results/gcrt/collision/pipeline/state.json").read_text(encoding="utf-8"))
    assert state["status"] == "running" and state["stages"]["all_models"]["status"] == "running"
    assert state["contract"]["method"] == method_contract()
    assert list(DOMAINS)[0] == "front_brake" and front_pending(), "Native builder must still own unfinished front_brake models"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registration_path = OUTPUT / "registration.json"
    assert not registration_path.exists(), "A dispatcher registration already exists; inspect its actual process before recovery"
    dispatcher_id = str(uuid.uuid4())
    jobs, skipped = [], []
    contract = method_contract()
    for kind in list(DOMAINS)[1:]:
        historical, validation = load_construction(kind)
        for scope in SCOPES.values():
            history = [a for a in historical if a.source_id in scope.history_ids]
            development = [a for a in validation if a.source_id in scope.development_ids]
            for method in methods(scope):
                destination = model_path(method, scope.name, kind)
                if destination.exists():
                    assert destination.with_suffix(".json").exists(), "Unexpected existing in-progress future model"
                    check_model(destination, method, scope, history, development)
                    skipped.append(str(destination))
                    continue
                assert not destination.with_suffix(".json").exists()
                jobs.append({"id": f"{kind}__{scope.name}__{method}", "scenario": kind,
                             "scope": scope.name, "method": method, "destination": str(destination),
                             "dispatcher_id": dispatcher_id, "source_contract": contract})
    assert len(jobs) == 108 and len(skipped) == 9
    assert len({job["destination"] for job in jobs}) == 108
    registration = {"status": "reserving", "dispatcher_id": dispatcher_id, "pid": os.getpid(),
                    "started_utc": utc(), "workers": arguments.workers, "cpu_threads_per_worker": 1,
                    "scope": "Only the three future scenario kinds; original native builder owns front_brake.",
                    "steps_per_model": 1500, "original_training_functions": True,
                    "source_contract": contract, "jobs": jobs, "already_complete_models": skipped,
                    "dispatcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "reservation_semantics": "Typed incomplete caches; native build and freeze gates reject them. Resume the original pipeline after this dispatcher completes if it catches up.",
                    "additional_physical_calls": 0}
    with registration_path.open("x", encoding="utf-8") as stream:
        json.dump(registration, stream, indent=2, allow_nan=False)
    for job in jobs:
        final = Path(job["destination"])
        assert final.resolve().is_relative_to((ROOT / "results").resolve())
        final.parent.mkdir(parents=True, exist_ok=True)
        with final.open("xb") as stream:
            torch.save({"format": "collision-model-reservation", "dispatcher_id": dispatcher_id,
                        "job_id": job["id"]}, stream)
    # Completion is monotone in the original first-kind loop. Remaining first-kind
    # models after all reservations prove that it has not entered a reserved kind.
    pending_front = front_pending()
    assert pending_front, "Original builder crossed the dispatch boundary during reservation"
    registration.update(status="running", reservations=108, front_pending_after_reservations=pending_front)
    write_json(registration_path, registration)
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    progress = {"status": "running", "dispatcher_id": dispatcher_id, "pid": os.getpid(),
                "expected_models": 108, "completed_models": 0, "failed_models": 0, "models": []}
    write_json(OUTPUT / "progress.json", progress)
    with ProcessPoolExecutor(max_workers=arguments.workers, mp_context=mp.get_context("spawn")) as pool:
        pending = [pool.submit(train_job, job) for job in jobs]
        for future in as_completed(pending):
            receipt = future.result()
            progress["models"].append(receipt)
            progress["completed_models"] += receipt["status"] == "complete"
            progress["failed_models"] += receipt["status"] == "failed"
            progress["observed_at_utc"] = utc()
            write_json(OUTPUT / "progress.json", progress)
            print("parallel models", progress["completed_models"], "/ 108; failed", progress["failed_models"], flush=True)
    assert method_contract() == contract
    progress.update(status="complete" if progress["completed_models"] == 108 and not progress["failed_models"] else "failed",
                    finished_utc=utc(), wall_s=time.perf_counter() - started, additional_physical_calls=0)
    write_json(OUTPUT / "progress.json", progress)
    if progress["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
