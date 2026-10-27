"""Freeze and run the domain comparison and its matched H-by-F development controls."""

import argparse
import hashlib
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from importlib.metadata import version

import torch
from threadpoolctl import threadpool_limits

from controllers.highway.catalog import BY_ID
from testing_methods.baselines.av_bo_rf_ei import method as rf_ei
from testing_methods.baselines.indago import method as indago_model
from testing_methods.baselines.indago import pool as indago
from testing_methods.baselines.paired_system_active_evaluation import method as paired
from testing_methods.baselines.response_baselines.collision import run as run_random
from testing_methods.baselines.samota import pool as samota
from testing_methods.baselines.scenario_fuzz import model as graph_model
from testing_methods.baselines.scenario_fuzz import pool as scenario_fuzz
from testing_methods.common.fsbo.method import load as load_target
from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.domain_build import OUTPUT as MODELS
from testing_methods.history_transfer.domain_build import load_sources
from testing_methods.history_transfer.domain_discovery import aggregate_history, select
from testing_methods.history_transfer.joint_training import load_model

from .collision_metrics import collision_recall, summarize
from .construction import load_construction
from .discovery_config import development_tasks, read_json, write_json
from .domain_paired_validation import build_proxy
from .domain_risk import disclose_driving_quality, disclose_impact
from .executions import ExecutionBank, execution_contract
from .initialization import initial_indices
from .protocol import ROOT, TEST_RECORDS_ROOT, runtime_environment
from .query import QueryLedger, run_isolated
from .source_registration import maintained_path, specification_matches

OUTPUT = ROOT / "experiment_outputs/benchmark/collision/discovery/domain_comparison"
REFERENCES = (
    "random",
    "av_bo_rf_ei",
    "samota",
    "scenario_fuzz",
    "paired_system_active_evaluation",
    "indago",
)
METHODS = ("gcrt", *REFERENCES, "H0F0", "H1F0", "H0F1")
WORKERS = 8


def run_path(task, method):
    owner = "gcrt" if method in ("gcrt", "H0F0", "H1F0", "H0F1") else method
    return (
        ROOT
        / "experiment_outputs"
        / owner
        / "collision/discovery/domain_comparison/runs"
        / method
        / task["target_id"]
        / task["scenario"]
        / f"{task['algorithm_seed']}.json"
    )


def source_contract():
    paths = {"environment.yml"}
    for directory in (
        "testing_methods/common",
        "testing_methods/baselines/av_bo_rf_ei",
        "testing_methods/baselines/samota",
        "testing_methods/baselines/scenario_fuzz",
        "testing_methods/baselines/indago",
        "testing_methods/baselines/paired_system_active_evaluation",
    ):
        paths.update(
            str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / directory).glob("*.py")
        )
    paths.update(
        (
            "testing_methods/baselines/response_baselines/collision.py",
            "testing_methods/common/fsbo/method.py",
        )
    )
    paths.update(
        "testing_methods/history_transfer/" + p + ".py"
        for p in (
            "domain_build",
            "domain_discovery",
            "joint_training",
            "joint_response",
            "converged_response",
            "residual_scale",
        )
    )
    paths.update(
        "experiments/" + p + ".py"
        for p in (
            "domain_comparison",
            "domain_comparison_audit",
            "domain_comparison_report",
            "domain_risk",
            "domain_paired_validation",
            "query",
            "schema",
            "scenarios",
            "collision_metrics",
            "initialization",
            "executions",
            "construction",
            "storage",
            "protocol",
            "discovery_config",
            "domain_main_validation",
            "inference_reference",
            "statistics",
        )
    )
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(paths)}


def model_paths(kind):
    return (
        MODELS / kind / "target_kernel.pt",
        MODELS / kind / "history_response.pt",
        MODELS / kind / "source_priors.npz",
        ROOT
        / f"experiment_outputs/indago/collision/discovery/domain_comparison/stage_a/models/{kind}/classifier.pt",
        ROOT
        / f"experiment_outputs/scenario_fuzz/collision/discovery/domain_comparison/stage_a/models/{kind}/risk.pt",
    )


def baseline_configuration():
    return {
        "av_bo_rf_ei": asdict(rf_ei.DEFAULT),
        "samota": {**asdict(samota.DEFAULT), "thresholds": list(samota.DEFAULT.thresholds)},
        "scenario_fuzz": asdict(scenario_fuzz.DEFAULT),
        "scenario_fuzz_model": asdict(graph_model.DEFAULT),
        "paired_system_active_evaluation": asdict(paired.DEFAULT),
        "indago": asdict(indago.DEFAULT),
        "indago_model": asdict(indago_model.DEFAULT),
    }


def protocol():
    construction = {}
    models = {}
    for kind in ("cut_in", "cut_out"):
        history, development = load_construction(kind)
        construction[kind] = {
            "history": archive_fingerprints(history),
            "development": archive_fingerprints(development),
        }
        paths = model_paths(kind)
        _, target_metadata = load_target(paths[0])
        _, response_metadata = load_model(paths[1])
        for metadata in (target_metadata, response_metadata["specification"]):
            if (
                metadata["history_fingerprints"] != construction[kind]["history"]
                or metadata["development_fingerprints"] != construction[kind]["development"]
            ):
                raise ValueError(
                    "Main construction differs from the allowed history/development split"
                )
        for loader, path in ((indago_model.load, paths[3]), (graph_model.load, paths[4])):
            _, metadata = loader(path)
            if metadata["history_fingerprints"] != construction[kind]["history"]:
                raise ValueError(
                    "A frozen classifier differs from the allowed historical responses"
                )
        models.update(
            {
                str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in model_paths(kind)
            }
        )
    return {
        "purpose": "domain failure-discovery comparison and matched global H-by-F controls",
        "main": "gcrt",
        "methods": list(METHODS),
        "references": list(REFERENCES),
        "tasks": development_tasks(),
        "candidate_count": 512,
        "budget": 128,
        "initial_count": 8,
        "checkpoints": [16, 32, 64, 128],
        "unique_runs": 400,
        "equivalent_queries": 51200,
        "mechanism_alias": {"H1F1": "gcrt; identical paid run, not an additional execution"},
        "main_config": {
            "beta": 1.0,
            "source_prior_mass": 0.75,
            "independent_target_mass": 0.25,
            "physical_mass": 0.1,
            "reliability": "global prequery likelihood",
            "selection": "greedy",
            "target_adaptation": {"steps": 10, "learning_rate": 0.003},
            "physical_adaptation": {"steps": 10, "learning_rate": 0.01},
            "H0_teacher": "same-source moment-matched latent mean and total covariance",
            "F0": "initial and later labels never condition, adapt kernels or update weights",
            "mode_gradient_tolerance": 1e-10,
            "online_feature_precision": "float64",
            "conditional_inference_device": "cpu",
            "kernel_and_adaptation_device": "cuda",
        },
        "baseline_config": baseline_configuration(),
        "adaptations": {
            "samota": "restart covered objectives; explicitly confirmed by user",
            "scenario_fuzz": "public planned graph, pool projection, native paid quality and threshold completion",
            "paired_system_active_evaluation": "equal mean of 8 allowed historical predictors; fixed bandwidth 0.25",
            "indago": "fixed historical classifier; no target policy training failures",
            "av_bo_rf_ei": "measured equal-mass impact delta-V; 2D horizon-censored TTC; finite EI maximization",
        },
        "feedback": "same paid collision, clearance, TTC, impact delta-V and safe driving-quality scalar interface",
        "used_information": {
            "gcrt": "paid Boolean",
            "H0F1": "paid Boolean",
            "H0F0": "queried indices",
            "H1F0": "queried indices",
            "random": "queried indices",
            "av_bo_rf_ei": "paid collision, impact delta-V and TTC",
            "samota": "paid collision, clearance and TTC",
            "scenario_fuzz": "paid collision and driving quality",
            "paired_system_active_evaluation": "paid Boolean and fixed historical proxy",
            "indago": "queried indices",
        },
        "statistics": {
            "aggregation": "equal implementation/training lineage",
            "bootstrap_repetitions": 2000,
            "bootstrap_seed": 1501,
            "seed_is_independent_sut": False,
        },
        "gate": {
            "reference_order": [
                "recall",
                "count",
                "AUC",
                "method identifier (lexicographic maximum)",
            ],
            "remaining_miss_reduction": 0.5,
            "count_not_lower": True,
            "minimum_positive_lineages": 2,
        },
        "construction": construction,
        "models": models,
        "source_files": source_contract(),
        "runtime": runtime_environment(),
        "extra_packages": {p: version(p) for p in ("optuna", "torch-geometric", "scikit-fuzzy")},
        "workers": WORKERS,
        "new_physical_calls": 0,
        "independent_targets_accessed": False,
        "old_runs_reused": 0,
    }


def freeze():
    required = [
        OUTPUT / "data_audit.json",
        OUTPUT / "driving_quality_audit.json",
        ROOT / "experiment_outputs/gcrt/collision/discovery/domain_comparison/stage_a/audit.json",
    ]
    required += [
        ROOT / f"experiment_outputs/{m}/collision/discovery/domain_comparison/stage_a/audit.json"
        for m in REFERENCES
        if m != "random"
    ]
    for path in required:
        if read_json(path)["status"] != "passed":
            raise ValueError(f"Stage A did not pass: {path}")
    if read_json(MODELS / "construction.json")["status"] != "complete":
        raise ValueError("Complete the historical main-method construction before freezing")
    regression = OUTPUT / "regression.xml"
    suites = ET.parse(regression).getroot().findall("testsuite")
    if (
        not suites
        or sum(int(s.get("tests", 0)) for s in suites) == 0
        or any(int(s.get("failures", 0)) or int(s.get("errors", 0)) for s in suites)
    ):
        raise ValueError("The full engineering and MetaDrive regression must pass before freezing")
    required.append(regression)
    specification = protocol()
    specification["stage_a_evidence"] = {
        str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in required
    }
    path = OUTPUT / "protocol.json"
    if path.exists() and read_json(path) != specification:
        raise ValueError("The domain protocol was already frozen differently")
    write_json(path, specification)
    return specification


def validate_protocol(specification):
    current = protocol()
    if any(current[k] != specification[k] for k in current):
        raise ValueError("Domain runtime, construction or source code changed after freezing")
    for path, digest in specification["stage_a_evidence"].items():
        if hashlib.sha256((ROOT / maintained_path(path)).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen Stage A evidence changed")


def public_task(task, method, candidates, history):
    public = {
        "candidates": candidates,
        "initial_indices": initial_indices(512, 8, task["algorithm_seed"]),
        "budget": 128,
    }
    if method in ("gcrt", "H0F0", "H1F0", "H0F1"):
        base, _ = load_target(MODELS / task["scenario"] / "target_kernel.pt")
        sources = load_sources(task["scenario"])
        if method.startswith("H0"):
            sources = aggregate_history(sources)
        return select, {
            **public,
            "pretrained": base,
            "sources": sources,
            "mode": "H1F1" if method == "gcrt" else method,
            "device": "cuda",
        }
    public["seed"] = task["algorithm_seed"]
    if method == "random":
        return run_random, {**public, "method": "random"}
    if method == "av_bo_rf_ei":
        return rf_ei.run, public
    if method == "samota":
        return samota.run, public
    if method == "paired_system_active_evaluation":
        proxy, _ = build_proxy(candidates, history)
        return paired.run, {**public, "proxy": proxy}
    model_path = model_paths(task["scenario"])[3 if method == "indago" else 4]
    loader, selector = (
        (indago_model.load, indago.run)
        if method == "indago"
        else (graph_model.load, scenario_fuzz.run)
    )
    model, metadata = loader(model_path)
    if metadata["history_fingerprints"] != archive_fingerprints(history):
        raise ValueError("A baseline classifier differs from the allowed historical responses")
    return selector, {**public, "classifier": model}


def execute(task, method, specification):
    torch.set_num_threads(1)
    if specification != read_json(OUTPUT / "protocol.json"):
        raise ValueError("The submitted task differs from the frozen domain protocol")
    with threadpool_limits(1):
        started = time.perf_counter()
        history, development = load_construction(task["scenario"])
        target = next(a for a in development if a.source_id == task["target_id"])
        bank = ExecutionBank(
            TEST_RECORDS_ROOT / "physical/development" / task["scenario"] / task["target_id"]
        )
        if bank.candidates != target.scenarios or not specification_matches(
            bank.manifest["contract"], execution_contract(BY_ID[task["target_id"]])
        ):
            raise ValueError("The development execution bank differs from its frozen object")
        setting = {
            "task": task,
            "method": method,
            "protocol_sha256": hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest(),
            "bank_contract": bank.manifest["contract"],
            "candidate_ids": [s.scenario_id for s in bank.candidates],
        }
        destination = run_path(task, method)
        if destination.exists():
            saved = read_json(destination)
            if saved["specification"] != setting or saved["result"]["actual_queries"] != 128:
                raise ValueError("Saved domain run differs from the frozen paid task")
        else:
            selector, public = public_task(task, method, bank.candidates, history)
            construction_s = time.perf_counter() - started
            ledger = QueryLedger(
                bank.candidates,
                128,
                lambda scene: disclose_driving_quality(disclose_impact(bank(scene))),
            )
            try:
                result = run_isolated(selector, public, ledger)
            except Exception:
                write_json(
                    OUTPUT
                    / "failed_runs"
                    / method
                    / task["target_id"]
                    / task["scenario"]
                    / f"{task['algorithm_seed']}_{time.time_ns()}.json",
                    {
                        "specification": setting,
                        "query_ledger": ledger.records,
                        "error": traceback.format_exc(),
                        "new_physical_calls": 0,
                    },
                )
                raise
            metrics = {
                **summarize(bank.candidates, result["queries"], ledger.trajectories),
                **collision_recall(bank.candidates, target.collision.tolist(), result["queries"]),
            }
            saved = {
                "specification": setting,
                "initial_indices": public["initial_indices"],
                "result": result,
                "query_ledger": ledger.records,
                "metrics": metrics,
                "cost": {
                    "wall_s": time.perf_counter() - started,
                    "construction_and_loading_s": construction_s,
                    "model_inference_s": result.get("model_inference_s", 0),
                    "equivalent_queries": 128,
                    "new_physical_calls": 0,
                },
            }
            write_json(destination, saved)
    return {
        **task,
        "method": method,
        "path": str(destination.relative_to(ROOT)).replace("\\", "/"),
        **saved["metrics"],
    }


def first():
    specification = read_json(OUTPUT / "protocol.json")
    validate_protocol(specification)
    task = development_tasks()[0]
    rows = []
    for method in METHODS:
        rows.append(execute(task, method, specification))
        write_json(
            OUTPUT / "first.json",
            {"status": "complete" if len(rows) == len(METHODS) else "running", "runs": rows},
        )
        print("domain_first", method, "complete", flush=True)


def matrix():
    specification = read_json(OUTPUT / "protocol.json")
    validate_protocol(specification)
    if read_json(OUTPUT / "first_audit.json")["status"] != "passed":
        raise ValueError("Independently audit the actual first task before the full domain matrix")
    jobs = [(task, method) for task in development_tasks() for method in METHODS]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(execute, t, m, specification): (t, m) for t, m in jobs}
        for future in as_completed(futures):
            completed = future.result()
            rows.append(completed)
            rows.sort(
                key=lambda r: (r["target_id"], r["scenario"], r["algorithm_seed"], r["method"])
            )
            write_json(
                OUTPUT / "run_index.json",
                {
                    "status": "complete" if len(rows) == len(jobs) else "running",
                    "runs": rows,
                    "new_physical_calls": 0,
                    "unique_runs": len(rows),
                    "equivalent_queries": 128 * len(rows),
                },
            )
            print("domain_matrix", len(rows), "/", len(jobs), completed["method"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("freeze", "first", "audit_first", "run", "audit", "report")
    )
    stage = parser.parse_args().stage
    if stage == "freeze":
        freeze()
    elif stage == "first":
        first()
    elif stage == "run":
        matrix()
    elif stage in ("audit_first", "audit"):
        from .domain_comparison_audit import audit

        audit(first_only=stage == "audit_first")
    else:
        from .domain_comparison_report import report

        report()


if __name__ == "__main__":
    main()
