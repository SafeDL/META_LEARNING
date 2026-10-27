"""Complete matched H-by-F controls for the admitted diagonal historical prior."""

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from testing_methods.common.fsbo.method import load as load_target
from testing_methods.history_transfer.domain_build import OUTPUT as MODELS
from testing_methods.history_transfer.domain_build import load_sources
from testing_methods.history_transfer.domain_discovery import aggregate_history, select

from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery_config import development_tasks, read_json, write_json
from .domain_comparison import OUTPUT as DOMAIN
from .domain_comparison import run_path as original_path
from .domain_comparison_report import METRICS, gate, interaction
from .domain_covariance import diagonal_sources
from .domain_main_validation import reconstruct
from .domain_risk import disclose_driving_quality, disclose_impact
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import ROOT, TEST_RECORDS_ROOT
from .query import QueryLedger, run_isolated
from .statistics import method_summary

OUTPUT = DOMAIN / "diagonal_controls"
RUNS = ROOT / "experiment_outputs/gcrt/collision/discovery/domain_comparison/diagonal_controls"


def design():
    return {
        "purpose": "complete the matched aggregate-teacher feedback control after covariance selection",
        "covariance_decision_sha256": hashlib.sha256(
            (DOMAIN / "covariance/decision.json").read_bytes()
        ).hexdigest(),
        "cpu_validation_sha256": hashlib.sha256(
            (DOMAIN / "cpu_replay.json").read_bytes()
        ).hexdigest(),
        "tasks": development_tasks(),
        "new_runs": 40,
        "budget": 128,
        "initial_count": 8,
        "mode": "H0F1",
        "diagonalize_before_moment_matching": True,
        "reuse": {
            "H1F1": "the 40 audited diagonal covariance runs",
            "H0F0_and_H1F0": "prove identical complete marginal predictions before reusing 80 Stage B runs",
        },
        "device": "cpu",
        "workers": 8,
        "physical_calls": 0,
        "source_files": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "experiments/domain_diagonal_controls.py",
                "testing_methods/history_transfer/domain_discovery.py",
            )
        },
    }


def freeze():
    if (
        read_json(DOMAIN / "covariance/decision.json")["selected_covariance"] != "diagonal"
        or read_json(DOMAIN / "cpu_replay.json")["status"] != "passed"
    ):
        raise ValueError("Finish the diagonal decision and CPU numerical validation first")
    path = OUTPUT / "protocol.json"
    spec = design()
    if path.exists() and read_json(path) != spec:
        raise ValueError("The matching diagonal control was already frozen differently")
    write_json(path, spec)
    return spec


def run_path(task):
    return RUNS / task["target_id"] / task["scenario"] / f"{task['algorithm_seed']}.json"


def execute(task):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        _, development = load_construction(task["scenario"])
        archive = next(a for a in development if a.source_id == task["target_id"])
        bank = ExecutionBank(
            TEST_RECORDS_ROOT / "physical/development" / task["scenario"] / task["target_id"]
        )
        base, _ = load_target(MODELS / task["scenario"] / "target_kernel.pt")
        sources = aggregate_history(diagonal_sources(load_sources(task["scenario"])))
        initial = initial_indices(512, 8, task["algorithm_seed"])
        setting = {
            "task": task,
            "method": "H0F1",
            "protocol_sha256": hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest(),
        }
        path = run_path(task)
        if path.exists():
            saved = read_json(path)
            if saved["specification"] != setting:
                raise ValueError("The diagonal teacher run belongs to another paid task")
        else:
            started = time.perf_counter()
            ledger = QueryLedger(
                bank.candidates,
                128,
                lambda scene: disclose_driving_quality(disclose_impact(bank(scene))),
            )
            result = run_isolated(
                select,
                {
                    "candidates": bank.candidates,
                    "pretrained": base,
                    "sources": sources,
                    "mode": "H0F1",
                    "initial_indices": initial,
                    "budget": 128,
                    "device": "cpu",
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
                "query_ledger": ledger.records,
                "metrics": metrics,
                "cost": {"wall_s": time.perf_counter() - started, "physical_calls": 0},
            }
            write_json(path, saved)
        result = saved["result"]
        if (
            len(result["queries"]) != 128
            or len(set(result["selected_indices"])) != 128
            or result["selected_indices"][:8] != initial
        ):
            raise ValueError("The matching teacher did not spend the exact common budget")
        for query, record in zip(result["queries"], saved["query_ledger"]):
            execution = disclose_driving_quality(
                disclose_impact(bank(bank.candidates[query["index"]]))
            )
            if (
                query["feedback"] != execution.feedback.to_dict()
                or record["feedback"] != query["feedback"]
            ):
                raise ValueError(
                    "The diagonal teacher used a scalar not attached to its paid execution"
                )
        for number in (9, 64, 128):
            row = result["queries"][number - 1]
            index, probability, weights, components = reconstruct(
                bank.candidates, base, sources, result["queries"][: number - 1], "H0F1", initial
            )
            if index != row["index"]:
                raise ValueError("Independent diagonal-teacher acquisition differs")
            np.testing.assert_allclose(probability[index], row["prequery_probability"], atol=1e-7)
            np.testing.assert_allclose(
                components[:, index], row["prequery_component_probabilities"], atol=1e-7
            )
            np.testing.assert_allclose(weights, row["prequery_component_weights"], atol=1e-12)
        return {
            **task,
            "method": "H0F1",
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            **saved["metrics"],
            "independent_acquisition_checks": [9, 64, 128],
        }


def run(first_only=False):
    if read_json(OUTPUT / "protocol.json") != design():
        raise ValueError("The matching diagonal construction changed after registration")
    if not first_only and read_json(OUTPUT / "first.json")["status"] != "passed":
        raise ValueError("Check the actual matching teacher first before the complete matrix")
    tasks = development_tasks()[:1] if first_only else development_tasks()
    rows = []
    with ProcessPoolExecutor(max_workers=1 if first_only else 8) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            write_json(
                OUTPUT / ("first.json" if first_only else "run_index.json"),
                {"status": "passed" if len(rows) == len(tasks) else "running", "runs": rows},
            )
            print("diagonal_teacher", len(rows), "/", len(tasks), flush=True)
    return rows


def prove_f0_reuse():
    checks = []
    with threadpool_limits(1), torch.no_grad():
        for kind in ("cut_in", "cut_out"):
            base, _ = load_target(MODELS / kind / "target_kernel.pt")
            base = base.double()
            _, development = load_construction(kind)
            original = load_sources(kind)
            diagonal = diagonal_sources(original)
            for mode in ("H0F0", "H1F0"):
                old = aggregate_history(original) if mode.startswith("H0") else original
                new = aggregate_history(diagonal) if mode.startswith("H0") else diagonal
                np.testing.assert_array_equal(new["means"], old["means"])
                np.testing.assert_allclose(
                    np.diagonal(new["covariances"], axis1=1, axis2=2),
                    np.diagonal(old["covariances"], axis1=1, axis2=2),
                    atol=1e-14,
                )
                for archive in development:
                    for seed in (11, 23, 37, 53, 71):
                        task = next(
                            t
                            for t in development_tasks()
                            if t["scenario"] == kind
                            and t["target_id"] == archive.source_id
                            and t["algorithm_seed"] == seed
                        )
                        expected = read_json(original_path(task, mode))["result"]
                        initial = initial_indices(512, 8, seed)
                        _, probabilities, weights, components = reconstruct(
                            archive.scenarios, base, new, [], mode, initial
                        )
                        remaining, error = set(range(512)), 0.0
                        for number, row in enumerate(expected["queries"]):
                            best = max(probabilities[i] for i in remaining)
                            index = (
                                initial[number]
                                if number < 8
                                else min(
                                    (i for i in remaining if probabilities[i] >= best - 1e-12),
                                    key=lambda i: archive.scenarios[i].scenario_id,
                                )
                            )
                            if index != row["index"]:
                                raise ValueError(
                                    "Diagonalization changed a purportedly reusable F0 acquisition"
                                )
                            remaining.remove(index)
                            error = max(
                                error, abs(probabilities[index] - row["prequery_probability"])
                            )
                            np.testing.assert_allclose(
                                components[:, index],
                                row["prequery_component_probabilities"],
                                atol=1e-7,
                            )
                            np.testing.assert_allclose(
                                weights, row["prequery_component_weights"], atol=1e-12
                            )
                        if error > 1e-7:
                            raise ValueError(
                                "Diagonalization changed the fixed marginal probability"
                            )
                        checks.append(
                            {
                                "task": task,
                                "mode": mode,
                                "identical_paid_indices": 128,
                                "maximum_probability_error": error,
                            }
                        )
    return checks


def report():
    torch.set_num_threads(1)
    index = read_json(OUTPUT / "run_index.json")
    if index["status"] != "passed" or len(index["runs"]) != 40:
        raise ValueError("Complete and independently check all 40 matched diagonal teachers")
    equivalences = prove_f0_reuse()
    old = read_json(DOMAIN / "run_index.json")["runs"]
    covariance = read_json(DOMAIN / "covariance/run_index.json")["runs"]
    rows = [r for r in old if r["method"] not in ("gcrt", "H0F1")]
    rows += [{**r, "method": "gcrt"} for r in covariance if r["method"] == "diagonal"]
    rows += index["runs"]
    effects = [
        paired_lineage_bootstrap(
            [r for r in rows if r[metric] is not None], "gcrt", control, metric
        )
        for control in ("H0F0", "H1F0", "H0F1")
        for metric in METRICS
    ]
    result = {
        "status": "complete_finalized_development_method",
        "covariance_mode": "diagonal",
        "global_reliability": True,
        "greedy": True,
        "local_weights": False,
        "lookahead": False,
        "gate": gate(rows),
        "summary": method_summary(rows),
        "mechanism_comparisons": effects,
        "interaction": {metric: interaction(rows, metric) for metric in METRICS},
        "f0_equivalences": equivalences,
        "independent_teacher_acquisition_checks": 120,
        "new_matching_teacher_runs": 40,
        "new_equivalent_queries": 5120,
        "new_physical_calls": 0,
        "historical_construction": "testing_methods.history_transfer.domain_training; fresh allowed histories and Boolean-consistent episodes",
        "next_stage": "D independent scenario and system preparation",
    }
    write_json(OUTPUT / "finalization.json", result)
    write_json(OUTPUT / "matched_run_index.json", {"status": "complete", "runs": rows})
    print({"status": result["status"], "gate": result["gate"]["status"]}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "first", "run", "report"))
    stage = parser.parse_args().stage
    if stage in ("freeze", "report"):
        {"freeze": freeze, "report": report}[stage]()
    else:
        run(first_only=stage == "first")


if __name__ == "__main__":
    main()
