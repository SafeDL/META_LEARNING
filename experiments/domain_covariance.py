"""Matched Stage C covariance control using the frozen domain construction."""

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from testing_methods.common.fsbo.method import load as load_target
from testing_methods.history_transfer.converged_search import select
from testing_methods.history_transfer.domain_build import OUTPUT as MODELS
from testing_methods.history_transfer.domain_build import load_sources

from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery_config import development_tasks, read_json, write_json
from .domain_comparison import OUTPUT as DOMAIN
from .domain_comparison import WORKERS, validate_protocol
from .domain_comparison import run_path as domain_path
from .domain_main_validation import reconstruct
from .domain_risk import disclose_driving_quality, disclose_impact
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import ROOT, TEST_RECORDS_ROOT
from .query import QueryLedger, run_isolated
from .statistics import method_summary

OUTPUT = DOMAIN / "covariance"
RUNS = ROOT / "experiment_outputs/gcrt/collision/discovery/domain_comparison/covariance"
VARIANTS = ("full", "diagonal")


def diagonal_sources(sources):
    covariance = np.asarray(sources["covariances"])
    return {**sources, "covariances": covariance * np.eye(covariance.shape[-1])[None]}


def design():
    return {
        "purpose": "matched historical posterior covariance control; global greedy method finalization",
        "domain_protocol_sha256": hashlib.sha256(
            (DOMAIN / "protocol.json").read_bytes()
        ).hexdigest(),
        "variants": list(VARIANTS),
        "tasks": development_tasks(),
        "budget": 128,
        "initial_count": 8,
        "controlled_difference": "historical posterior off-diagonal covariance; preserve means and diagonal",
        "shared": {
            "beta": 1.0,
            "history_mass": 0.75,
            "independent_target_mass": 0.25,
            "physical_mass": 0.1,
            "global_reliability": True,
            "greedy": True,
            "online_features": "float64",
            "local_weights": False,
            "lookahead": False,
        },
        "selection_rule": "retain full unless diagonal is no worse in count and recall and better in either; use AUC only to break equality",
        "validation": "full replay must match every Stage B query, prediction, likelihood and weight",
        "experimental_runs": 40,
        "validation_replay_runs": 40,
        "physical_calls": 0,
        "source_files": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "experiments/domain_covariance.py",
                "testing_methods/history_transfer/converged_search.py",
                "testing_methods/history_transfer/converged_response.py",
                "testing_methods/history_transfer/residual_scale.py",
                "testing_methods/common/gp_classifier.py",
                "testing_methods/common/gp_laplace.py",
            )
        },
    }


def freeze():
    validate_protocol(read_json(DOMAIN / "protocol.json"))
    if (
        read_json(DOMAIN / "audit.json")["status"] != "passed"
        or read_json(DOMAIN / "gate.json")["status"] != "passed"
    ):
        raise ValueError("Complete and audit the admitted domain comparison before Stage C")
    specification = design()
    path = OUTPUT / "protocol.json"
    if path.exists() and read_json(path) != specification:
        raise ValueError("The covariance control was already frozen differently")
    write_json(path, specification)
    return specification


def run_path(task, variant):
    return RUNS / variant / task["target_id"] / task["scenario"] / f"{task['algorithm_seed']}.json"


def compare_full_replay(actual, expected):
    if actual["selected_indices"] != expected["selected_indices"]:
        raise ValueError("The reusable global engine differs from the frozen Stage B sequence")
    maximum = 0.0
    for a, b in zip(actual["queries"], expected["queries"]):
        for key in (
            "prequery_probability",
            "prequery_component_probabilities",
            "prequery_component_weights",
        ):
            difference = float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key]))))
            maximum = max(maximum, difference)
            if difference > 1e-10:
                raise ValueError(f"Stage C global replay differs in {key}: {difference}")
        np.testing.assert_allclose(
            a["prequery_group_log_likelihood"], b["prequery_log_likelihood"], atol=1e-10
        )
        np.testing.assert_allclose(
            np.r_[0.9 * np.asarray(a["group_weights_after_query"]), 0.1],
            b["component_weights_after_query"],
            atol=1e-10,
        )
    return maximum


def choose_covariance(stats):
    full, diagonal = stats["full"], stats["diagonal"]
    count_difference = diagonal["unique_collisions"]["mean"] - full["unique_collisions"]["mean"]
    recall_difference = diagonal["collision_recall"]["mean"] - full["collision_recall"]["mean"]
    dominates = (
        count_difference >= -1e-12
        and recall_difference >= -1e-12
        and (count_difference > 1e-12 or recall_difference > 1e-12)
    )
    tied = abs(count_difference) <= 1e-12 and abs(recall_difference) <= 1e-12
    return (
        "diagonal"
        if dominates
        or (tied and diagonal["discovery_auc"]["mean"] > full["discovery_auc"]["mean"] + 1e-12)
        else "full"
    )


def execute(task, variant, *, verify_acquisition=False):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        started = time.perf_counter()
        _, development = load_construction(task["scenario"])
        archive = next(a for a in development if a.source_id == task["target_id"])
        bank = ExecutionBank(
            TEST_RECORDS_ROOT / "physical/development" / task["scenario"] / task["target_id"]
        )
        base, _ = load_target(MODELS / task["scenario"] / "target_kernel.pt")
        sources = load_sources(task["scenario"])
        if variant == "diagonal":
            sources = diagonal_sources(sources)
        path = run_path(task, variant)
        setting = {
            "task": task,
            "variant": variant,
            "protocol_sha256": hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest(),
            "candidate_ids": [s.scenario_id for s in bank.candidates],
        }
        if path.exists():
            saved = read_json(path)
            if saved["specification"] != setting:
                raise ValueError("A covariance run differs from its frozen task")
        else:
            ledger = QueryLedger(
                bank.candidates,
                128,
                lambda scene: disclose_driving_quality(disclose_impact(bank(scene))),
            )
            result = run_isolated(
                select,
                {
                    "candidates": bank.candidates,
                    "pretrained": base.double(),
                    "sources": sources,
                    "mode": "converged_global",
                    "beta": 1.0,
                    "initial_indices": initial_indices(512, 8, task["algorithm_seed"]),
                    "budget": 128,
                    "device": "cuda",
                },
                ledger,
            )
            metrics = {
                **summarize(bank.candidates, result["queries"], ledger.trajectories),
                **collision_recall(bank.candidates, archive.collision.tolist(), result["queries"]),
            }
            saved = {
                "specification": setting,
                "result": result,
                "metrics": metrics,
                "query_ledger": ledger.records,
                "cost": {"wall_s": time.perf_counter() - started, "new_physical_calls": 0},
            }
            write_json(path, saved)
        error = None
        if variant == "full":
            error = compare_full_replay(
                saved["result"], read_json(domain_path(task, "gcrt"))["result"]
            )
        checks = []
        if verify_acquisition:
            for number in (9, 64, 128):
                row = saved["result"]["queries"][number - 1]
                index, probability, weights, _ = reconstruct(
                    bank.candidates,
                    base,
                    sources,
                    saved["result"]["queries"][: number - 1],
                    "H1F1",
                    initial_indices(512, 8, task["algorithm_seed"]),
                )
                if index != row["index"]:
                    raise ValueError(
                        "Independent covariance acquisition selected a different public candidate"
                    )
                np.testing.assert_allclose(
                    probability[index], row["prequery_probability"], atol=1e-7
                )
                np.testing.assert_allclose(weights, row["prequery_component_weights"], atol=1e-10)
                checks.append(number)
        return {
            **task,
            "method": variant,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            **saved["metrics"],
            "full_replay_maximum_error": error,
            "independent_acquisition_checks": checks,
        }


def run(*, first_only=False):
    if read_json(OUTPUT / "protocol.json") != design():
        raise ValueError("Covariance code or domain construction changed after freezing")
    if not first_only and read_json(OUTPUT / "first.json")["status"] != "passed":
        raise ValueError("Check the two covariance first runs before the full control")
    tasks = development_tasks()[:1] if first_only else development_tasks()
    rows = []
    with ProcessPoolExecutor(max_workers=2 if first_only else WORKERS) as pool:
        futures = [
            pool.submit(execute, task, variant, verify_acquisition=True)
            for task in tasks
            for variant in VARIANTS
        ]
        for future in as_completed(futures):
            rows.append(future.result())
            print("covariance", len(rows), "/", 2 * len(tasks), flush=True)
            write_json(
                OUTPUT / ("first.json" if first_only else "run_index.json"),
                {"status": "passed" if len(rows) == 2 * len(tasks) else "running", "runs": rows},
            )
    return rows


def report():
    index = read_json(OUTPUT / "run_index.json")
    if index["status"] != "passed" or len(index["runs"]) != 80:
        raise ValueError("Complete every paired covariance run and its numerical checks")
    rows = index["runs"]
    stats = method_summary(rows)
    effects = [
        paired_lineage_bootstrap(
            [r for r in rows if r[metric] is not None], "full", "diagonal", metric
        )
        for metric in ("unique_collisions", "collision_recall", "discovery_auc")
    ]
    selected = choose_covariance(stats)
    decision = {
        "status": "complete_matched_covariance_control",
        "selected_covariance": selected,
        "global_reliability": True,
        "greedy": True,
        "local_weights": False,
        "lookahead": False,
        "summary": stats,
        "comparisons": effects,
        "full_replays": 40,
        "diagonal_experimental_runs": 40,
        "validation_replay_equivalent_queries": 5120,
        "experimental_queries": 5120,
        "new_physical_calls": 0,
        "maximum_replay_error": max(
            r["full_replay_maximum_error"] for r in rows if r["method"] == "full"
        ),
        "independent_acquisition_checks": sum(
            len(r["independent_acquisition_checks"]) for r in rows
        ),
        "next_action": "finalize consistent historical construction"
        if selected == "full"
        else "complete matching diagonal-teacher feedback control before finalization",
    }
    write_json(OUTPUT / "decision.json", decision)
    print(
        {"selected_covariance": selected, "maximum_replay_error": decision["maximum_replay_error"]},
        flush=True,
    )
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "first", "run", "report"))
    stage = parser.parse_args().stage
    if stage == "freeze":
        freeze()
    elif stage == "report":
        report()
    else:
        run(first_only=stage == "first")


if __name__ == "__main__":
    main()
