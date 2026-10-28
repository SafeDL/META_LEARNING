"""Analyze preserved mechanism controls and equivalent static predictions."""

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from testing_methods.history_transfer.selection import aggregate_history

from .analyze_covariance import diagonal_sources
from .collision_metrics import paired_lineage_bootstrap
from .construction import load_archives
from .development import OUTPUT as DOMAIN
from .development import run_path as original_path
from .discovery_config import development_tasks, read_json, write_json
from .initialization import initial_indices
from .protocol import ROOT
from .recorded_models import DEVELOPMENT_MODELS as MODELS
from .recorded_models import load_development_sources as load_sources
from .recorded_models import load_model as load_target
from .report_development import METRICS, gate, interaction
from .statistics import method_summary
from .verify_selection import reconstruct

OUTPUT = DOMAIN / "diagonal_controls"

RUNS = ROOT / "experiment_outputs/gcrt/collision/discovery/domain_comparison/diagonal_controls"


def run_path(task):
    return RUNS / task["target_id"] / task["scenario"] / f"{task['algorithm_seed']}.json"


def prove_f0_reuse():
    checks = []
    with threadpool_limits(1), torch.no_grad():
        for kind in ("cut_in", "cut_out"):
            base, _ = load_target(MODELS / kind / "target_kernel.pt")
            base = base.double()
            _, development = load_archives(kind)
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
        "historical_construction": "testing_methods.history_transfer.training; fresh allowed histories and Boolean-consistent episodes",
        "next_stage": "D independent scenario and system preparation",
    }
    write_json(OUTPUT / "finalization.json", result)
    write_json(OUTPUT / "matched_run_index.json", {"status": "complete", "runs": rows})
    print({"status": result["status"], "gate": result["gate"]["status"]}, flush=True)
    return result
