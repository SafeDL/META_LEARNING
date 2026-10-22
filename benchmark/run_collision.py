"""Run the approved rebuild stages, preserving native failures and resumable ledgers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from controllers.highway.catalog import BY_ID
from controllers.highway.factory import create
from controllers.highway.learning import CHECKPOINT_ROOT
from controllers.highway.train import TRAINING, freeze_training, training_contract

from .executions import execution_contract
from .protocol import RESULT_ROOT, ROOT, method_contract
from .qualification import QUALIFICATION_ROOT, qualify


STAGES = (
    ("history", "benchmark.collect", "history"),
    ("core_models", "benchmark.build_collision_models", "core"),
    ("development", "benchmark.develop_collision"),
    ("all_models", "benchmark.build_collision_models", "complete"),
    ("finite_freeze", "benchmark.freeze_collision"),
    ("targets", "benchmark.collect", "targets"),
    ("finite_comparisons", "benchmark.evaluate_collision"),
    ("new_collision_certification", "benchmark.certify_collisions"),
    ("finite_report", "benchmark.report_collision"),
    ("continuous_development", "benchmark.develop_continuous"),
    ("continuous_freeze", "benchmark.freeze_continuous"),
    ("continuous_comparisons", "benchmark.evaluate_continuous"),
    ("continuous_report", "benchmark.report_continuous"),
)
OUTPUT = RESULT_ROOT / "pipeline"
STATE_PATH = OUTPUT / "state.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def acquire_owner(path):
    owner = Path(path).open("a+b")
    if owner.seek(0, os.SEEK_END) == 0:
        owner.write(b"0")
        owner.flush()
    owner.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(owner.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        owner.close()
        raise RuntimeError("Another pipeline process already owns this execution ledger") from error
    return owner


def policy_report(name):
    sut = BY_ID[name]
    report = json.loads((CHECKPOINT_ROOT / sut.training_run / "training.json").read_text(encoding="utf-8"))
    if (report["source_contract"] != training_contract() or report["configuration"] != TRAINING
            or report["object"] != asdict(sut)):
        raise ValueError(f"{name} does not match the registered shared training construction")
    if report["status"] not in ("running", "qualified"):
        raise ValueError(f"{name} terminated with status {report['status']}")
    if report["status"] == "qualified" and report["actual_timesteps"] < TRAINING["total_timesteps"]:
        raise ValueError(f"{name} has not completed the full registered training budget")
    return report


def verify_public_qualification(name):
    """Independently replay factory-built policies, accounting for interrupted attempts."""
    sut = BY_ID[name]
    report = policy_report(name)
    if report["status"] != "qualified":
        raise ValueError("Public qualification requires completed, qualified policy construction")
    root = QUALIFICATION_ROOT / "public_interface" / sut.training_run
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "qualification.json"
    contract = execution_contract(sut)
    if destination.exists():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if saved["contract"] != contract or not saved["qualified"]:
            raise ValueError("Existing public-interface qualification needs explicit construction review")
        return saved
    ledger = root / "attempts.jsonl"
    previous = ledger.read_text(encoding="utf-8").splitlines() if ledger.exists() else []
    calls = len(previous)
    started = time.perf_counter()

    def factory():
        nonlocal calls
        calls += 1
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"attempt": calls, "started_utc": utc_now(), "sut_id": name}) + "\n")
        return create(name)

    result = qualify(factory, save_root=root / f"attempt_{len(previous) + 1}")
    saved = {"training_run": sut.training_run, "contract": contract, **result,
             "actual_physical_calls": calls, "previous_interrupted_attempts": len(previous),
             "elapsed_s": time.perf_counter() - started,
             "interrupted_attempt_durations_unknown": bool(previous)}
    destination.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    if not saved["qualified"]:
        raise ValueError(f"{name} failed public-interface capability replay")
    return saved


def audit_completion():
    from .continuous_selection import METHODS
    from .protocol import DATA_ROOT, DEFAULT, validate_sut_freeze
    from .studies import CONTINUOUS_TASKS, tasks

    frozen = json.loads((DATA_ROOT / "frozen_protocol.json").read_text(encoding="utf-8"))
    validate_sut_freeze(frozen)
    if frozen["source_contract"] != method_contract() or len(frozen["method"]["models"]) != 156:
        raise ValueError("Completed workflow does not retain the frozen full model construction")
    finite = len(tasks())
    continuous = len(CONTINUOUS_TASKS) * len(METHODS) * len(DEFAULT.algorithm_seeds) * len(DEFAULT.environment_seeds)
    for path, field, expected in (
        (RESULT_ROOT / "run_index.json", "runs", finite),
        (RESULT_ROOT / "new_collision_metrics.json", "runs", finite),
        (RESULT_ROOT / "report/report.json", "run_count", finite),
        (RESULT_ROOT / "continuous/run_index.json", "runs", continuous),
        (RESULT_ROOT / "continuous/report/report.json", "runs", continuous),
    ):
        saved = json.loads(path.read_text(encoding="utf-8"))
        count = len(saved[field]) if isinstance(saved[field], list) else saved[field]
        if saved["status"] != "complete" or count != expected:
            raise ValueError(f"Missing preregistered results in {path}")
    return {"finite_runs": finite, "continuous_runs": continuous, "frozen_models": 156}


class Pipeline:
    def __init__(self, *, resume=False):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.contract = {"method": method_contract(), "training": training_contract(),
                         "objects": [asdict(s) for s in BY_ID.values()]}
        self.owner = acquire_owner(OUTPUT / "owner.lock")
        try:
            if STATE_PATH.exists():
                self.state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                if not resume or self.state["contract"] != self.contract:
                    raise ValueError("Existing pipeline needs --resume and identical code/registered objects")
            else:
                self.state = {"status": "running", "started_utc": utc_now(), "contract": self.contract,
                              "stages": {}}
            self.state.update(pid=os.getpid(), status="running")
            self.save()
        except Exception:
            self.close()
            raise

    def close(self):
        self.owner.close()

    def save(self):
        temporary = STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        temporary.replace(STATE_PATH)

    def check_contract(self):
        if self.contract != {"method": method_contract(), "training": training_contract(),
                             "objects": [asdict(s) for s in BY_ID.values()]}:
            raise ValueError("Pipeline inputs changed while stages were running")

    def command(self, stage, module, *arguments):
        command = [sys.executable, "-B", "-u", "-m", module, *arguments]
        with self.lock:
            self.check_contract()
            old = self.state["stages"].get(stage, {})
            if old.get("status") == "complete":
                if old["command"] != command:
                    raise ValueError("Completed stage command changed")
                return
            attempt = old.get("attempt", 0) + 1
            log = OUTPUT / f"{stage}_{attempt}.log"
            self.state["stages"][stage] = {"status": "running", "command": command,
                                          "attempt": attempt, "log": str(log), "started_utc": utc_now()}
            self.save()
        print("starting", stage, flush=True)
        started = time.perf_counter()
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           PYTHONIOENCODING="utf-8")
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        with self.lock:
            self.state["stages"][stage].update(status="complete" if result.returncode == 0 else "failed",
                                               native_exit_code=result.returncode, finished_utc=utc_now(),
                                               elapsed_s=time.perf_counter() - started)
            self.save()
        if result.returncode:
            raise RuntimeError(f"{stage} failed with native exit {result.returncode}; see {log}")
        print("completed", stage, flush=True)

    def run(self):
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(self.command, name, "controllers.highway.train", name)
                           for name in ("history_ppo", "history_dqn", "development_ppo")
                           if not (CHECKPOINT_ROOT / BY_ID[name].training_run / "training.json").exists()]
                for future in futures:
                    future.result()
            while True:
                self.check_contract()
                reports = {name: policy_report(name) for name in (
                    "history_ppo", "history_dqn", "development_ppo")}
                if all(r["status"] == "qualified" for r in reports.values()):
                    break
                self.state["waiting_for"] = {n: r["status"] for n, r in reports.items()}
                self.state["checked_utc"] = utc_now()
                self.save()
                time.sleep(30)
            for name in reports:
                verify_public_qualification(name)
            frozen = CHECKPOINT_ROOT / "training_protocol.json"
            if not frozen.exists():
                freeze_training()
            protocol = json.loads(frozen.read_text(encoding="utf-8"))
            if protocol["source_contract"] != training_contract() or protocol["configuration"] != TRAINING:
                raise ValueError("Target policy training protocol differs from this construction")
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for name in ("target_ppo", "target_dqn"):
                    path = CHECKPOINT_ROOT / BY_ID[name].training_run / "training.json"
                    if path.exists():
                        if policy_report(name)["status"] != "qualified":
                            raise ValueError("An existing target policy run needs construction review")
                    else:
                        futures.append(executor.submit(self.command, name, "controllers.highway.train", name))
                for future in futures:
                    future.result()
            for name in ("target_ppo", "target_dqn"):
                verify_public_qualification(name)
            self.state.pop("waiting_for", None)
            self.save()
            for stage, module, *arguments in STAGES:
                self.command(stage, module, *arguments)
            self.state.update(status="complete", finished_utc=utc_now(), completed_matrix=audit_completion())
            self.save()
        except Exception as error:
            self.state.update(status="failed", error=repr(error), stopped_utc=utc_now())
            self.save()
            raise
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    Pipeline(resume=parser.parse_args().resume).run()


if __name__ == "__main__":
    main()
