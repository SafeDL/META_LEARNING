"""Parent-owned continuous execution and separate extension experiment contracts."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from methods.baselines.indago.search import CONFIG as INDAGO_CONFIG
from methods.baselines.samota.search import CONFIG as SAMOTA_CONFIG
from methods.baselines.deep_hyperion.search import CONFIG as HYPERION_CONFIG
from methods.gcrt.continuous import CONFIG as MAIN_CONFIG
from methods.gcrt.train import load_encoder
from methods.baselines.pooled_ft.method import load as load_classifier
from methods.common.response import archive_fingerprints

from .collision_metrics import summarize
from .continuous_execution import ContinuousExecutor
from .continuous_selection import METHODS, select
from .continuous_space import initial_scenarios
from .protocol import ROOT, method_contract
from .query import QueryLedger, run_isolated


CONFIGURATIONS = {"history_response": MAIN_CONFIG, "indago": INDAGO_CONFIG,
                  "samota": SAMOTA_CONFIG, "deep_hyperion": HYPERION_CONFIG, "random": {}}


def source_contract():
    paths = ("benchmark/continuous_runs.py", "benchmark/continuous_execution.py",
             "benchmark/continuous_selection.py", "benchmark/continuous_space.py",
             "methods/gcrt/continuous.py", "methods/baselines/indago/search.py",
             "methods/baselines/samota/search.py", "methods/baselines/deep_hyperion/search.py",
             "benchmark/develop_continuous.py", "benchmark/freeze_continuous.py",
             "benchmark/evaluate_continuous.py", "benchmark/report_continuous.py")
    return {**method_contract(), **{p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}}


def run_task(sut_id, kind, history, *, method, algorithm_seed, destination, metadata,
             checkpoint=None, budget=128, initial_count=8, environment_seed=81001,
             configuration=None, ordinary_only=False):
    if method not in METHODS:
        raise ValueError("Unknown continuous algorithm")
    if not 0 < initial_count <= budget:
        raise ValueError("Continuous initialization must fit the actual-call budget")
    history = tuple(history)
    if method not in ("history_response", "indago") and history:
        raise ValueError("This continuous comparator has no historical input")
    configuration = CONFIGURATIONS[method] if configuration is None else configuration
    initial = initial_scenarios(kind, algorithm_seed, initial_count, ordinary_only=ordinary_only)
    identity = {"metadata": metadata, "method": method, "algorithm_seed": algorithm_seed,
                "budget": budget, "initial_count": initial_count, "configuration": configuration,
                "ordinary_only": ordinary_only,
                "source_contract": source_contract(), "history_fingerprints": archive_fingerprints(history),
                "initial_scenarios": [asdict(s) for s in initial]}
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        identity["checkpoint"] = {"path": str(checkpoint),
                                  "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    identity = json.loads(json.dumps(identity))
    destination = Path(destination)
    executor = ContinuousExecutor(sut_id, destination.with_suffix(""), identity,
                                  environment_seed=environment_seed)
    if destination.exists():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if saved["specification"] != identity or saved["result"]["actual_queries"] != budget:
            raise ValueError("Existing continuous result belongs to another registered experiment")
        return saved
    started = time.perf_counter()
    public = {"initial": initial, "method": method, "seed": algorithm_seed, "budget": budget,
              "ordinary_only": ordinary_only}
    if method in ("history_response", "indago"):
        model, evidence = (load_encoder if method == "history_response" else load_classifier)(checkpoint)
        if (set(evidence["history_ids"]) != {a.source_id for a in history}
                or evidence["history_fingerprints"] != archive_fingerprints(history)):
            raise ValueError("Continuous prior contains different or disallowed historical responses")
        if method == "history_response":
            public.update(encoder=model, history=history,
                          target_weight=evidence["config"]["target_weight"])
        else:
            public["classifier"] = model
    if configuration:
        public["configuration"] = configuration
    construction_s = time.perf_counter() - started
    ledger = QueryLedger(initial, budget, executor)
    started = time.perf_counter()
    result = run_isolated(select, public, ledger)
    wall_s = time.perf_counter() - started
    metrics = summarize(ledger.candidates, result["queries"], ledger.trajectories)
    attempts = [json.loads(p.read_text(encoding="utf-8")) for p in executor.root.glob("attempt_*.json")]
    if len(attempts) != budget or any(r["status"] != "complete" for r in attempts):
        raise ValueError("Continuous physical call ledger is incomplete")
    physical_s = sum(r["execution_s"] for r in attempts)
    output = {"specification": identity, "result": result, "metrics": metrics,
              "candidates": [asdict(s) for s in ledger.candidates], "query_ledger": ledger.records,
              "cost": {"execution_mode": "actual_on_demand", "actual_physical_calls": len(attempts),
                       "physical_execution_s": physical_s,
                       "sut_initialization_s": sum(r["wall_s"] for r in attempts) - physical_s,
                       "model_construction_s": construction_s, "selector_process_wall_s": wall_s,
                       "physical_log": str(executor.root)}}
    temporary = destination.with_suffix(".tmp")
    encoded = json.dumps(output, indent=2, allow_nan=False)
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return json.loads(encoded)
