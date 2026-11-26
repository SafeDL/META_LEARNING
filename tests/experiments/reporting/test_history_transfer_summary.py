"""Final summaries retain the full grid, valid metric counts and declared budgets."""

import copy
from itertools import product

import numpy as np
import pytest

from experiments.history_transfer_summary import (
    budget_summary,
    prediction_summary,
    register_statistics,
)
from experiments.k_shot import METRICS, prediction_metrics
from tests.fixtures.predictions import DESIGN, prediction_rows


def summarize_predictions(rows):
    return prediction_summary(rows, **DESIGN, checkpoints=(0, 1))


def test_prediction_summary_preserves_zero_tasks_and_reports_metric_specific_ads_counts():
    result = summarize_predictions(prediction_rows())
    assert result["scopes"]["all"]["design"]["registered_ads"] == 2
    assert result["scopes"]["all"]["design"]["registered_seed_runs"] == 8
    assert result["scopes"]["zero"]["paired"]["valid_ads"][1][0] == [2, 2, 0, 0, 0, 2]
    assert result["scopes"]["positive"]["paired"]["valid_ads"][1][0] == [2] * 6
    assert result["scopes"]["zero"]["unpaired"]["mean"][1][0][2] is None
    assert result["scopes"]["zero"]["unpaired"]["mean"][1][0][1] == pytest.approx(0.01)
    assert result["undefined_metric_rows"] and not result["failed_rows"]


def test_k_shot_averages_four_feedback_levels_without_counting_them_as_ads():
    checkpoints = (0, 5, 10, 20, 50, 100, 200)
    methods = ("target_only", "pooled_history", "full")
    labels = np.zeros(400, dtype=bool)
    labels[300:] = True
    rows = []
    quality = {5: 0.55, 10: 0.65, 20: 0.75, 50: 0.85, 100: 0.9, 200: 0.95}
    for ads, method, k in product(("a", "b"), methods, checkpoints):
        probability = np.full(400, 0.2 if method == "target_only" else 0.4)
        if method == "full" and k:
            probability = np.where(labels, quality[k], 1 - quality[k])
        metrics, _ = prediction_metrics(probability, labels, [f"{i:03}" for i in range(400)], list(range(200, 400)))
        rows.append({"sut_id": ads, "scenario": "scene", "seed": 11, "method": method, "K": k, "status": "complete", "candidate_count": 400, "pool_collisions": 100, **metrics})
    result = prediction_summary(rows, ads={"a": "rule", "b": "learning"}, scenes=("scene",), seeds=(11,), methods=methods, checkpoints=checkpoints, candidate_count=400, replicates=100)
    summary = result["few_feedback_analysis"]["scopes"]["scene"]
    expected = -0.5 * np.log(0.2) - 0.5 * np.log(0.8) + np.mean(np.log([0.55, 0.65, 0.75, 0.85]))
    assert summary["paired"]["gain"][2][METRICS.index("nll")] == pytest.approx(expected)
    assert summary["design"]["registered_ads"] == 2
    assert summary["design"]["registered_seed_runs"] == 2
    assert result["few_feedback_analysis"]["support"]["scene"]["both_supported"]
    self_gain = result["adaptation_from_own_k0"]["scopes"]["scene"]["paired"]["gain"][1]
    assert self_gain[0] == [0] * len(METRICS)


@pytest.mark.parametrize("change", ("missing", "duplicate", "pending"))
def test_incomplete_or_duplicate_prediction_grid_cannot_be_presented_as_final(change):
    predictions = prediction_rows()
    if change == "missing":
        predictions.pop()
    elif change == "duplicate":
        predictions.append(copy.deepcopy(predictions[0]))
    else:
        predictions[0]["status"] = "pending"
    with pytest.raises(ValueError):
        summarize_predictions(predictions)


def test_zero_positive_recall_cannot_be_filled_with_zero_for_a_nicer_curve():
    predictions = prediction_rows()
    predictions[0]["recall_100"] = 0
    with pytest.raises(ValueError, match="Undefined ranking metrics"):
        summarize_predictions(predictions)


def test_methods_and_k_must_keep_the_same_fixed_evaluation_truth():
    predictions = prediction_rows()
    row = next(r for r in predictions if r["scenario"] == "positive" and r["method"] == "full")
    row["evaluation_collisions"] = 2
    with pytest.raises(ValueError, match="Evaluation truth differs"):
        summarize_predictions(predictions)


def test_failed_prediction_is_explicitly_retained_without_zero_metrics():
    predictions = prediction_rows()
    for metric in METRICS:
        predictions[0][metric] = None
    predictions[0].update(status="failed", failure_reason="recorded numerical failure")
    result = summarize_predictions(predictions)
    assert len(result["failed_rows"]) == 1
    predictions[0]["brier"] = 0
    with pytest.raises(ValueError, match="failed prediction"):
        summarize_predictions(predictions)


@pytest.fixture
def budgets():
    rows = []
    for sut, scene, seed, method in product(
        DESIGN["ads"], DESIGN["scenes"], DESIGN["seeds"], DESIGN["methods"]
    ):
        pool = 0 if scene == "zero" else 2
        curve = (
            [0, 0, 0, 0]
            if pool == 0
            else ([0, 0, 1, 1] if method == "target_only" else [0, 1, 1, 2])
        )
        recall = [value / pool if pool else None for value in curve]
        rows.append(
            {
                "sut_id": sut,
                "scenario": scene,
                "seed": seed,
                "method": method,
                "status": "complete",
                "candidate_count": 10,
                "pool_collisions": pool,
                "metrics": {
                    "discovery_curve": curve,
                    "queries": 4,
                    "initial_count": 1,
                    "candidate_count": 10,
                    "pool_unique_collisions": pool,
                    "unique_collisions": curve[-1],
                    "initial_collisions": curve[0],
                    "post_initial_collisions": curve[-1] - curve[0],
                    "discovery_auc": float(np.mean(curve)),
                    "recall_curve": recall,
                    "collision_recall": recall[-1],
                },
            }
        )
    return rows


def summarize_budgets(rows):
    return budget_summary(rows, **DESIGN, budget=4, initial_count=1)


def test_budget_auc_includes_initialization_and_keeps_macro_counts_separate_from_coverage(budgets):
    result = summarize_budgets(budgets)
    assert result["endpoints"]["all"]["paired"]["gain"][1] == pytest.approx([0.25, 0.5, 0.5])
    assert result["pooled_run_coverage"]["full_pool_collision_denominator"] == 8
    assert result["pooled_run_coverage"]["coverage_curve"][1][-1] == 1
    assert result["curves"]["zero"]["unpaired"]["mean"][1][-1] == [0, None]
    assert result["endpoints"]["all"]["design"]["registered_ads"] == 2


def test_pooled_coverage_uses_the_same_complete_logical_runs_for_every_method(budgets):
    row = next(r for r in budgets if r["scenario"] == "positive" and r["method"] == "target_only")
    row.update(status="failed", failure_reason="recorded failure", metrics=None)
    result = summarize_budgets(budgets)
    assert len(result["failed_runs"]) == 1
    assert result["pooled_run_coverage"]["common_completed_seed_tasks"] == 7
    assert result["pooled_run_coverage"]["full_pool_collision_denominator"] == 6
    assert result["pooled_run_coverage"]["discovery_numerator"][0][-1] == 3
    assert result["pooled_run_coverage"]["discovery_numerator"][1][-1] == 6


def test_different_common_initialization_cannot_create_an_auc_advantage(budgets):
    row = next(r for r in budgets if r["scenario"] == "positive" and r["method"] == "full")
    row["metrics"].update(
        discovery_curve=[1, 1, 1, 2],
        initial_collisions=1,
        post_initial_collisions=1,
        discovery_auc=1.25,
        recall_curve=[0.5, 0.5, 0.5, 1.0],
    )
    with pytest.raises(ValueError, match="same initial discovery curve"):
        summarize_budgets(budgets)


def test_common_initialization_checks_order_even_when_its_final_count_matches(budgets):
    for row in budgets:
        curve = (
            [0, 0, 0, 0]
            if row["pool_collisions"] == 0
            else ([0, 1, 1, 1] if row["method"] == "target_only" else [1, 1, 1, 2])
        )
        row["metrics"].update(
            discovery_curve=curve,
            initial_count=2,
            initial_collisions=curve[1],
            post_initial_collisions=curve[-1] - curve[1],
            discovery_auc=float(np.mean(curve)),
            recall_curve=[v / 2 if row["pool_collisions"] else None for v in curve],
        )
    with pytest.raises(ValueError, match="same initial discovery curve"):
        budget_summary(budgets, **DESIGN, budget=4, initial_count=2)


@pytest.mark.parametrize("change", ("jump", "auc", "coverage", "truth", "budget"))
def test_corrupted_discovery_or_wrong_budget_cannot_enter_final_statistics(budgets, change):
    row = next(r for r in budgets if r["scenario"] == "positive")
    if change == "jump":
        row["metrics"]["discovery_curve"] = [0, 2, 2, 2]
    elif change == "auc":
        row["metrics"]["discovery_auc"] = float("nan")
    elif change == "coverage":
        budgets[0]["metrics"]["collision_recall"] = 0
    elif change == "truth":
        row["pool_collisions"] = 3
    else:
        row["metrics"]["queries"] = 3
    with pytest.raises(ValueError):
        summarize_budgets(budgets)


def test_small_validation_fixture_cannot_be_relabelled_as_the_formal_budget(budgets):
    with pytest.raises(ValueError):
        budget_summary(
            budgets,
            ads=DESIGN["ads"],
            scenes=DESIGN["scenes"],
            seeds=DESIGN["seeds"],
            methods=DESIGN["methods"],
        )


def test_formal_statistics_registration_uses_four_scenes_and_one_fixed_seed(tmp_path):
    protocol = register_statistics(tmp_path)
    assert len(protocol["ads"]) == 8
    assert set(protocol["ads"].values()) == {"rule", "optimization", "search", "learning"}
    assert protocol["feedback_seeds"] == [11]
    assert protocol["scenarios"] == ["car_following_braking_onset", "cut_in_onset", "cut_out_rear_gap", "side_encroachment"]
    assert protocol["prediction"]["trajectories"] == 96
    assert protocol["prediction"]["metric_rows"] == 672
    assert protocol["prediction"]["evaluation_count"] == 1800
    assert "budget" not in protocol
    assert protocol["statistics"]["replicates"] == 10000
    assert not protocol["formal_target_labels_read"] and protocol["new_physical_executions"] == 0
