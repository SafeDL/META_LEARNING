"""One isolated paid-query run, with evaluator-only identity, metrics and cache evidence."""

import hashlib
import json
from pathlib import Path
import time

from benchmark.initialization import initial_indices
from methods.gcrt.prior import posterior
from methods.gcrt.selection import select
from methods.gcrt.train import load_encoder
from methods.common.response import archive_fingerprints

from methods.baselines.response_baselines.collision import METHODS, run as run_baseline
from .collision_metrics import summarize
from .query import QueryLedger, run_isolated
from .scenarios import features


MAIN_METHODS = ("history_response", "no_response", "fixed_weights", "no_target", "greedy")


def run_task(bank, history, *, method, algorithm_seed, budget, initial_count,
             destination, metadata, checkpoint=None, horizon=5):
    if method not in (*MAIN_METHODS, *METHODS):
        raise ValueError(f"Unknown registered method: {method}")
    history = tuple(history)
    candidates = bank.candidates
    construction = {"bank_contract": bank.manifest["contract"],
                    "candidate_ids": [s.scenario_id for s in candidates],
                    "history_ids": [a.source_id for a in history],
                    "history_contracts": [a.provenance.get("contract") for a in history],
                    "history_fingerprints": archive_fingerprints(history),
                    "method": method, "algorithm_seed": algorithm_seed,
                    "budget": budget, "initial_count": initial_count, "horizon": horizon}
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        construction["checkpoint"] = {"path": str(checkpoint),
                                      "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    specification = json.loads(json.dumps({"metadata": metadata, "construction": construction}))
    destination = Path(destination)
    if destination.exists():
        cached = json.loads(destination.read_text(encoding="utf-8"))
        if cached["specification"] != specification:
            raise ValueError("Existing run belongs to a different paid-query experiment")
        if cached["result"]["actual_queries"] != budget:
            raise ValueError("Existing run did not complete its registered budget")
        return cached
    started = time.perf_counter()
    public = {"candidates": candidates, "initial_indices": initial_indices(
        len(candidates), initial_count, algorithm_seed), "budget": budget}
    if method in MAIN_METHODS:
        encoder, model_metadata = load_encoder(checkpoint)
        if set(model_metadata["history_ids"]) != {a.source_id for a in history}:
            raise ValueError("Main representation and prior must use the same allowed sources")
        if model_metadata["history_fingerprints"] != construction["history_fingerprints"]:
            raise ValueError("Representation was trained on different historical responses")
        public.update(model=posterior(encoder, features(candidates), history,
                                      response=method != "no_response", target=method != "no_target",
                                      fixed_weights=method == "fixed_weights",
                                      target_weight=model_metadata["config"]["target_weight"]),
                      horizon=horizon, greedy=method == "greedy")
        selector = select
    else:
        public.update(method=method, seed=algorithm_seed)
        if method in ("history_ranking", "nearest_residual"):
            public["history"] = history
        if method in ("pooled_ft", "fsbo"):
            from methods.baselines.fsbo.method import load as load_fsbo
            from methods.baselines.pooled_ft.method import load as load_pooled
            base_model, model_metadata = (load_fsbo if method == "fsbo" else load_pooled)(checkpoint)
            if set(model_metadata["history_ids"]) != {a.source_id for a in history}:
                raise ValueError("Baseline checkpoint includes disallowed historical objects")
            if model_metadata["history_fingerprints"] != construction["history_fingerprints"]:
                raise ValueError("Baseline checkpoint was trained on different historical responses")
            public["pretrained"] = base_model
            public["adaptation_config"] = {key: model_metadata["config"][key]
                                            for key in ("adaptation_steps", "adaptation_rate")}
        selector = run_baseline
    model_construction_s = time.perf_counter() - started
    ledger = QueryLedger(candidates, budget, bank)
    started = time.perf_counter()
    result = run_isolated(selector, public, ledger)
    process_wall_s = time.perf_counter() - started
    metrics = summarize(candidates, result["queries"], ledger.trajectories)
    output = {"specification": specification, "result": result, "metrics": metrics,
              "cost": {"model_construction_s": model_construction_s,
                       "selector_process_wall_s": process_wall_s, "execution_mode": "replay",
                       "actual_new_physical_calls": 0,
                       "equivalent_selected_execution_s": metrics["equivalent_execution_s"],
                       "physical_bank": str(bank.root),
                       "physical_bank_calls": bank.manifest["attempted_calls"],
                       "physical_bank_execution_s": bank.manifest["physical_execution_s"]},
              "query_ledger": ledger.records}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    encoded = json.dumps(output, indent=2, allow_nan=False)
    output = json.loads(encoded)
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return output
