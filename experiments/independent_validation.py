"""Metered independent discovery with final allowed historical models."""

import argparse
import hashlib
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import torch
from threadpoolctl import threadpool_limits

from testing_methods.baselines.av_bo_rf_ei import method as rf_ei
from testing_methods.baselines.indago import method as indago_model
from testing_methods.baselines.indago import pool as indago
from testing_methods.baselines.paired_system_active_evaluation import method as paired
from testing_methods.baselines.response_baselines.collision import run as random_run
from testing_methods.baselines.samota import pool as samota
from testing_methods.baselines.scenario_fuzz import model as graph_model
from testing_methods.baselines.scenario_fuzz import pool as scenario_fuzz
from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.selection import aggregate_history, select

from .collision_metrics import collision_recall, summarize
from .construction import load_archives
from .discovery_config import read_json, write_json
from .domain_paired_validation import build_proxy
from .executions import ExecutionBank
from .independent_preparation import OUTPUT
from .independent_protocol import (
    CONTROLS,
    METHODS,
    REUSED_METHODS,
    WORKERS,
    canonical_task,
    held_family,
    jobs,
    reuse_jobs,
    run_path,
    target_bank_path,
    tasks,
    validate_protocol,
)
from .initialization import initial_indices
from .protocol import ROOT
from .query import PaidExecution, QueryLedger, run_isolated
from .recorded_models import load_model, load_sources, model_root
from .scenarios import sample
from .schema import QueryFeedback


def bank_audit_path(task):
    return OUTPUT / "evaluation_banks" / task["target_id"] / f"{task['scenario']}.json"


def public_task(task, method, candidates, history):
    public = {
        "candidates": candidates,
        "initial_indices": initial_indices(512, 8, task["algorithm_seed"]),
        "budget": 128,
    }
    held = held_family(task)
    if method in ("gcrt", *CONTROLS):
        base, _ = load_model(model_root("gcrt", task["scenario"], held) / "target_kernel.pt")
        sources = load_sources(task["scenario"], held)
        if sources["names"] != [a.source_id for a in history]:
            raise ValueError("A main selector received a different allowed historical source group")
        if method.startswith("H0"):
            sources = aggregate_history(sources)
        return select, {
            **public,
            "pretrained": base,
            "sources": sources,
            "mode": "H1F1" if method == "gcrt" else method,
            "device": "cpu",
        }
    public["seed"] = task["algorithm_seed"]
    if method == "random":
        return random_run, {**public, "method": "random"}
    if method == "av_bo_rf_ei":
        return rf_ei.run, public
    if method == "samota":
        return samota.run, public
    if method == "paired_system_active_evaluation":
        proxy, _ = build_proxy(candidates, history)
        return paired.run, {**public, "proxy": proxy}
    module, selector, filename = (
        (indago_model, indago.run, "classifier.pt")
        if method == "indago"
        else (graph_model, scenario_fuzz.run, "risk.pt")
    )
    classifier, metadata = module.load(model_root(method, task["scenario"], held) / filename)
    if (
        metadata["history_fingerprints"] != archive_fingerprints(history)
        or metadata["held_family"] != held
    ):
        raise ValueError("A baseline classifier contains a different allowed source scope")
    return selector, {**public, "classifier": classifier}


def execute(task, method):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        started = time.perf_counter()
        protocol = read_json(OUTPUT / "protocol.json")
        bank = ExecutionBank(target_bank_path(task))
        if (
            bank.manifest["contract"]
            != protocol["sut_contracts"][task["target_id"]][task["scenario"]]
        ):
            raise ValueError("The independent bank differs from its frozen physical object")
        private = read_json(bank_audit_path(task))
        if private["status"] != "passed" or private["candidate_ids"] != [
            s.scenario_id for s in bank.candidates
        ]:
            raise ValueError(
                "The evaluator cache differs from the audited complete independent pool"
            )
        setting = {
            "task": task,
            "method": method,
            "protocol_sha256": hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest(),
            "bank_contract": bank.manifest["contract"],
            "candidate_ids": private["candidate_ids"],
            "bank_audit_sha256": hashlib.sha256(bank_audit_path(task).read_bytes()).hexdigest(),
        }
        destination = run_path(task, method)
        initial = initial_indices(512, 8, task["algorithm_seed"])
        if destination.exists():
            saved = read_json(destination)
            if saved["specification"] != setting or saved["result"]["actual_queries"] != 128:
                raise ValueError("A saved independent run differs from its frozen paid task")
        else:
            history, _ = load_archives(task["scenario"], held_family(task))
            selector, public = public_task(task, method, bank.candidates, history)
            construction_s = time.perf_counter() - started

            def paid_execution(scene):
                raw = bank(scene)
                feedback = QueryFeedback.from_dict(
                    private["feedback"][bank.indices[scene.scenario_id]]
                )
                if (
                    raw.feedback.collision != feedback.collision
                    or raw.feedback.execution_s != feedback.execution_s
                ):
                    raise ValueError(
                        "The evaluator's paid scalar cache differs from its raw execution"
                    )
                return PaidExecution(feedback, raw.trajectory)

            ledger = QueryLedger(bank.candidates, 128, paid_execution)
            try:
                result = run_isolated(selector, public, ledger)
            except Exception:
                write_json(
                    OUTPUT
                    / "failed_runs"
                    / method
                    / task["target_id"]
                    / task["scenario"]
                    / task["condition"]
                    / f"{task['algorithm_seed']}_{time.time_ns()}.json",
                    {
                        "specification": setting,
                        "query_ledger": ledger.records,
                        "error": traceback.format_exc(),
                    },
                )
                raise
            labels = [f["collision"] for f in private["feedback"]]
            metrics = {
                **summarize(bank.candidates, result["queries"], ledger.trajectories),
                **collision_recall(bank.candidates, labels, result["queries"]),
            }
            saved = {
                "specification": setting,
                "initial_indices": initial,
                "result": result,
                "query_ledger": ledger.records,
                "metrics": metrics,
                "cost": {
                    "wall_s": time.perf_counter() - started,
                    "construction_and_loading_s": construction_s,
                    "model_inference_s": result.get("model_inference_s", 0),
                    "equivalent_queries": 128,
                    "new_physical_calls": 0,
                    "cached_scalar_derivation": "audited same paid raw execution; never sent before its metered query",
                },
            }
            write_json(destination, saved)
        return {
            **task,
            "method": method,
            "path": str(destination.relative_to(ROOT)).replace("\\", "/"),
            **saved["metrics"],
        }


def first_jobs():
    opened = next(
        t
        for t in tasks()
        if t["target_id"] == "independent_idm_mobil"
        and t["scenario"] == "cut_in_interaction"
        and t["condition"] == "all_sources"
        and t["algorithm_seed"] == 11
    )
    strict = next(
        t
        for t in tasks()
        if t["target_id"] == "independent_ppo"
        and t["scenario"] == "front_brake"
        and t["condition"] == "strict_leave_family"
        and t["algorithm_seed"] == 11
    )
    full = {**strict, "condition": "all_sources"}
    return (
        [(opened, method) for method in (*METHODS, *CONTROLS)]
        + [(full, method) for method in METHODS]
        + [(strict, method) for method in ("gcrt", "scenario_fuzz")]
    )


def selected_reuse_jobs(first_only=False):
    if not first_only:
        return reuse_jobs()
    strict = {
        tuple(task.items())
        for task, _ in first_jobs()
        if task["condition"] == "strict_leave_family"
    }
    return [(task, method) for task, method in reuse_jobs() if tuple(task.items()) in strict]


def reference_record(task, method):
    if task["condition"] != "strict_leave_family" or method not in REUSED_METHODS:
        raise ValueError("Only source-agnostic selectors have a cross-condition reuse reference")
    source = canonical_task(task, method)
    path = run_path(source, method)
    saved = read_json(path)
    if (
        saved["specification"]["task"] != source
        or saved["specification"]["method"] != method
        or saved["result"]["actual_queries"] != 128
        or len(saved["query_ledger"]) != 128
    ):
        raise ValueError("A reuse reference requires its completed, exact full-source run")
    candidates = sample(task["scenario"], 512, "new")
    first, public_first = public_task(source, method, candidates, [])
    second, public_second = public_task(task, method, candidates, [])
    if first is not second or public_first != public_second:
        raise ValueError(
            "Removing a historical family changed a supposedly source-agnostic selector"
        )
    public_serialized = json.dumps(
        {**public_first, "candidates": [asdict(scene) for scene in candidates]},
        sort_keys=True,
        allow_nan=False,
    ).encode()
    return {
        "task": task,
        "method": method,
        "source_task": source,
        "source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selector": first.__module__ + "." + first.__name__,
        "public_inputs_sha256": hashlib.sha256(public_serialized).hexdigest(),
        "protocol_sha256": saved["specification"]["protocol_sha256"],
        "bank_audit_sha256": saved["specification"]["bank_audit_sha256"],
        "source_queries": 128,
        "new_queries": 0,
        "new_physical_calls": 0,
    }


def run(first_only=False):
    validate_protocol()
    if read_json(OUTPUT / "evaluation_bank_audit.json")["status"] != "passed":
        raise ValueError("Audit the final raw pools before their independent metered replays")
    if not first_only and read_json(OUTPUT / "first_audit.json")["status"] != "passed":
        raise ValueError(
            "Independently check both full-source interaction and strict-family first tasks"
        )
    selected, rows = first_jobs() if first_only else jobs(), []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(execute, task, method) for task, method in selected]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(
                OUTPUT / ("first.json" if first_only else "run_index.json"),
                {
                    "status": "complete" if len(rows) == len(selected) else "running",
                    "runs": rows,
                    "equivalent_queries": 128 * len(rows),
                    "new_physical_calls": 0,
                },
            )
            print(
                "independent_first" if first_only else "independent_matrix",
                len(rows),
                "/",
                len(selected),
                row["method"],
                flush=True,
            )
    references = [
        reference_record(task, method) for task, method in selected_reuse_jobs(first_only)
    ]
    write_json(
        OUTPUT / ("first_reuse_index.json" if first_only else "reuse_index.json"),
        {"status": "complete", "references": references, "new_queries": 0, "new_physical_calls": 0},
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("banks", "first", "audit_first", "run", "audit", "report")
    )
    stage = parser.parse_args().stage
    if stage in ("first", "run"):
        run(first_only=stage == "first")
    elif stage in ("banks", "audit_first", "audit"):
        from .independent_validation_audit import audit, prepare_banks

        prepare_banks() if stage == "banks" else audit(first_only=stage == "audit_first")
    else:
        from .independent_validation_report import report

        report()


if __name__ == "__main__":
    main()
