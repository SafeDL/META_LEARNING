"""Evaluator-owned physical archives with resumable, individually metered executions."""

import copy
import gzip
import hashlib
import json
import multiprocessing as mp
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import cached_property
from importlib.metadata import version
from pathlib import Path

from controllers.highway.catalog import BY_ID, CURRENT_BY_ID
from controllers.highway.contracts import runtime_contract
from controllers.highway.factory import create
from controllers.highway.learning import CHECKPOINT_ROOT
from simulator.highway.collision_env import execute

from .domain import Domain
from .execution_events import analyze_events
from .parallel import COLLECTION_WORKERS, initialize_worker, prepare_processes
from .protocol import ROOT, episode_duration
from .qualification import QUALIFICATION_ROOT, verify_course_evidence
from .query import PaidExecution
from .schema import QueryFeedback, Scenario
from .source_registration import maintained_path, specification_matches

_LEARNING_SUT = None
_LEARNING_TEMPLATE = None


def _initialize_collection(sut_id):
    global _LEARNING_SUT, _LEARNING_TEMPLATE
    initialize_worker()
    _LEARNING_SUT = sut_id if BY_ID[sut_id].family == "learning" else None
    _LEARNING_TEMPLATE = None


def _execution_policy(sut_id):
    global _LEARNING_TEMPLATE
    if sut_id != _LEARNING_SUT:
        return create(sut_id)
    if _LEARNING_TEMPLATE is None:
        _LEARNING_TEMPLATE = create(sut_id)
    # Only immutable inference weights are shared. Each episode gets a pristine
    # driver and wrapper, including lane/speed targets and any wrapper state.
    model = _LEARNING_TEMPLATE.policy.model
    return copy.deepcopy(_LEARNING_TEMPLATE, {id(model): model})


def execution_contract(sut, *, kind=None):
    """Record the simulator, controller and policy checkpoint used for an execution."""
    paths = [
        "simulator/highway/collision_env.py",
        "simulator/highway/observation.py",
        "controllers/highway/control.py",
        "controllers/highway/factory.py",
        "controllers/highway/decision_interface.py",
    ]
    module = {
        "rule": "rule",
        "optimization": "optimization",
        "search": "search",
        "learning": "learning",
    }[sut.family]
    paths.append(f"controllers/highway/{module}.py")
    if sut.family in ("optimization", "search"):
        paths.append("controllers/highway/planning.py")
    paths.append("simulator/highway/timing.py")
    fingerprints = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    if sut.family == "learning":
        path = CHECKPOINT_ROOT / sut.training_run / "best.zip"
        fingerprints["checkpoint"] = hashlib.sha256(path.read_bytes()).hexdigest()
    packages = {"highway_env": version("highway-env"), "numpy": version("numpy")}
    if sut.family == "learning":
        packages.update(stable_baselines3=version("stable-baselines3"), torch=version("torch"))
    return {
        "runtime_contract": runtime_contract(sut),
        "source_files": fingerprints,
        "object": asdict(sut),
        "episode_duration_s": episode_duration(kind),
        **packages,
    }


def require_qualified(sut):
    if sut.family == "learning":
        path = CHECKPOINT_ROOT / sut.training_run / "training.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report["status"] != "qualified"
            or report["training_seed"] != sut.training_seed
            or report["sut_id"] != sut.sut_id
        ):
            raise ValueError("Learning object has not independently qualified")
    else:
        path = QUALIFICATION_ROOT / sut.sut_id / "qualification.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report["qualified"] or report["parameters"] != sut.parameters:
            raise ValueError("Controller has not qualified under its registered parameters")
    # A trained checkpoint and an old environment's qualification are not evidence
    # of capability after the execution stack changes. All ADS need a fresh gate.
    gate = QUALIFICATION_ROOT / sut.sut_id / "qualification.json"
    current = json.loads(gate.read_text(encoding="utf-8"))
    if not current.get("qualified") or current.get("runtime_contract") != runtime_contract(sut):
        raise ValueError(
            "ADS requires successful qualification under the current native highway-env contract"
        )
    evidence = verify_course_evidence(gate.parent)
    if evidence["courses"] != current.get("courses"):
        raise ValueError("Qualification summary differs from accepted capability trajectories")
    return str(gate.relative_to(ROOT))


def _execute_task(sut_id, scenario, environment_seed, attempt_path, call):
    attempt_path = Path(attempt_path)
    record = {
        "call": call,
        "sut_id": sut_id,
        "scenario_id": scenario.scenario_id,
        "environment_seed": environment_seed,
        "status": "running",
        "started": time.time(),
    }
    attempt_path.write_text(json.dumps(record), encoding="utf-8")
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as notices:
        try:
            policy = _execution_policy(sut_id)
            execution = execute(
                policy,
                scenario,
                environment_seed=environment_seed,
                duration_s=episode_duration(scenario.kind),
            )
        except Exception as error:
            record.update(status="failed", wall_s=time.perf_counter() - started, error=repr(error))
            raise
        finally:
            counts = Counter((n.category.__name__, str(n.message)) for n in notices)
            record["warnings"] = [
                {"category": category, "message": message, "count": count}
                for (category, message), count in sorted(counts.items())
            ]
            if record["status"] == "failed":
                attempt_path.write_text(json.dumps(record), encoding="utf-8")
    record.update(
        status="complete",
        execution_s=execution.feedback.execution_s,
        wall_s=time.perf_counter() - started,
    )
    attempt_path.write_text(json.dumps(record), encoding="utf-8")
    return execution


def _collect_task(sut_id, scenario, environment_seed, attempt_path, call, execution_path):
    execution = _execute_task(sut_id, scenario, environment_seed, attempt_path, call)
    payload = {
        "call": call,
        "scenario": asdict(scenario),
        "feedback": execution.feedback.to_dict(),
        "trajectory": execution.trajectory,
        "events": analyze_events(scenario, execution.trajectory),
    }
    path = Path(execution_path)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False)
    temporary.replace(path)
    # Do not serialize full trajectories back through the multiprocessing pipe.
    return execution.feedback.execution_s


class ExecutionBank:
    """Complete outcomes stay here in the evaluator, outside selector processes."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["format"] != "collision-execution-bank":
            raise ValueError("Not a new-protocol physical execution bank")
        if self.manifest["status"] != "complete":
            raise ValueError("Incomplete physical banks cannot enter evaluations")
        self.candidates = [Scenario(**s) for s in self.manifest["scenarios"]]
        definition = self.manifest.get("parameter_definition")
        self.domain = None if definition is None else Domain.from_dict(definition)
        if self.domain is not None:
            self.domain.features(self.candidates)
        self.indices = {s.scenario_id: i for i, s in enumerate(self.candidates)}

    def execution_path(self, index):
        references = self.manifest.get("execution_refs")
        if references is not None:
            return ROOT / references[index]
        return self.root / "executions" / f"{index:06d}.json.gz"

    @cached_property
    def snapshot_fingerprint(self):
        """Seal the current complete bank once per evaluator instance."""
        digest = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode())
        for index in range(len(self.candidates)):
            digest.update(hashlib.sha256(self.execution_path(index).read_bytes()).digest())
        return digest.hexdigest()

    def __call__(self, scenario):
        index = self.indices[scenario.scenario_id]
        if scenario != self.candidates[index]:
            raise ValueError("Cached coordinate differs from the requested scene")
        with gzip.open(self.execution_path(index), "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload["scenario"] != asdict(scenario):
            raise ValueError("Cached execution is attached to a different coordinate")
        feedback = QueryFeedback.from_dict(payload["feedback"])
        return PaidExecution(feedback, payload["trajectory"])


def collect_bank(
    sut_id,
    scenarios,
    destination,
    *,
    environment_seed=81001,
    workers=COLLECTION_WORKERS,
    current_registration=False,
):
    if not scenarios or len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError("A physical bank requires nonempty, uniquely identified scenes")
    sut = (CURRENT_BY_ID if current_registration else BY_ID)[sut_id]
    if current_registration and (
        len({s.kind for s in scenarios}) != 1
        or any(s.kind not in sut.supported_scenarios for s in scenarios)
    ):
        raise ValueError("Current execution banks require one registered functional scenario")
    qualification = require_qualified(sut)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "executions").mkdir(exist_ok=True)
    (destination / "attempts").mkdir(exist_ok=True)
    manifest_path = destination / "manifest.json"
    specification = {
        "format": "collision-execution-bank",
        "sut_id": sut_id,
        "environment_seed": environment_seed,
        "qualification": qualification,
        "contract": execution_contract(sut, kind=scenarios[0].kind),
        "scenarios": [asdict(s) for s in scenarios],
    }
    specification = json.loads(json.dumps(specification))
    pending = [
        i
        for i in range(len(scenarios))
        if not (destination / "executions" / f"{i:06d}.json.gz").exists()
    ]
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(
            not specification_matches(existing[key], value)
            if key == "contract"
            else maintained_path(existing[key]) != value.replace("\\", "/")
            if key == "qualification"
            else existing.get(key) != value
            for key, value in specification.items()
        ):
            raise ValueError(
                "Existing bank belongs to a different scene/controller execution contract"
            )
        if existing["status"] == "complete" and not pending:
            return ExecutionBank(destination)
    manifest = {**specification, "status": "running"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ledger_path = destination / "calls.jsonl"
    previous_calls = (
        [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
        if ledger_path.exists()
        else []
    )
    next_call = max((r["call"] for r in previous_calls), default=0) + 1
    started = time.perf_counter()
    errors = []
    # Only the parent writes the ledger, including failed/incomplete attempts on restart.
    prepare_processes()
    with ledger_path.open("a", encoding="utf-8", buffering=1) as ledger:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_collection,
            initargs=(sut_id,),
        ) as pool:
            futures = {}
            for index in pending:
                call = next_call
                next_call += 1
                ledger.write(
                    json.dumps(
                        {
                            "call": call,
                            "index": index,
                            "scenario_id": scenarios[index].scenario_id,
                            "status": "scheduled",
                            "time": time.time(),
                        }
                    )
                    + "\n"
                )
                attempt = destination / "attempts" / f"{call:06d}.json"
                futures[
                    pool.submit(
                        _collect_task,
                        sut_id,
                        scenarios[index],
                        environment_seed,
                        str(attempt),
                        call,
                        str(destination / "executions" / f"{index:06d}.json.gz"),
                    )
                ] = (index, call)
            for completed, future in enumerate(as_completed(futures), 1):
                index, call = futures[future]
                try:
                    execution_s = future.result()
                except Exception as error:
                    ledger.write(
                        json.dumps(
                            {"call": call, "index": index, "status": "failed", "error": repr(error)}
                        )
                        + "\n"
                    )
                    errors.append({"index": index, "call": call, "error": repr(error)})
                    continue
                ledger.write(
                    json.dumps(
                        {
                            "call": call,
                            "index": index,
                            "status": "complete",
                            "execution_s": execution_s,
                        }
                    )
                    + "\n"
                )
                if completed % 32 == 0 or completed == len(pending):
                    print(sut_id, scenarios[0].kind, completed, "/", len(pending), flush=True)
    attempts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((destination / "attempts").glob("*.json"))
    ]
    completed_calls = [r for r in attempts if r["status"] == "complete"]
    manifest.update(
        status="failed" if errors else "complete",
        candidate_count=len(scenarios),
        attempted_calls=len(attempts),
        successful_calls=len(completed_calls),
        physical_execution_s=sum(r["execution_s"] for r in completed_calls),
        total_successful_call_wall_s=sum(r["wall_s"] for r in completed_calls),
        sut_initialization_s=sum(r["wall_s"] - r["execution_s"] for r in completed_calls),
        failed_or_interrupted_calls=sum(r["status"] != "complete" for r in attempts),
        failed_call_wall_s=sum(r.get("wall_s", 0) for r in attempts if r["status"] == "failed"),
        interrupted_call_duration_unknown=sum(r["status"] == "running" for r in attempts),
        last_collection_wall_s=time.perf_counter() - started,
    )
    if errors:
        manifest["errors"] = errors
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(f"Physical executions failed; completed calls preserved: {errors}")
    return ExecutionBank(destination)
