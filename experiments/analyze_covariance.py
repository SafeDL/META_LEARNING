"""Analyze the preserved full/diagonal covariance comparison."""

import numpy as np

from .collision_metrics import paired_lineage_bootstrap
from .development import OUTPUT as DOMAIN
from .discovery_config import read_json, write_json
from .protocol import ROOT
from .statistics import method_summary

OUTPUT = DOMAIN / "covariance"

RUNS = ROOT / "experiment_outputs/gcrt/collision/discovery/domain_comparison/covariance"

VARIANTS = ("full", "diagonal")


def diagonal_sources(sources):
    covariance = np.asarray(sources["covariances"])
    return {**sources, "covariances": covariance * np.eye(covariance.shape[-1])[None]}


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
