"""Saved records must agree with real native traces and an external task approval."""

import copy
import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from controllers.highway.control import Control
from experiments.domain_risk import disclose_driving_quality, disclose_impact
from experiments.failure_metrics import summarize
from experiments.history_transfer_experiment import prediction_specification
from experiments.initialization import initial_indices
from experiments.k_shot import prediction_metrics, split_candidates
from experiments.query import QueryLedger
from experiments.record_audit import BankTruth, budget_record, prediction_record, prediction_scores
from experiments.scenarios import DOMAINS, sample
from experiments.storage import compact_feedback, compact_result
from simulator.highway.collision_env import execute
from testing_methods.baselines.response_baselines.collision import run as random_select
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


def test_independent_rank_formulas_handle_probability_ties_without_using_label_order():
    p = np.full(120, 0.5)
    y = np.array([True] * 20 + [False] * 100)
    ids = [f"id-{119 - i:03d}" for i in range(120)]
    scores, bins = prediction_scores(p, y, ids, list(range(120)))
    assert scores["auroc"] == 0.5
    assert scores["auprc_ap"] == pytest.approx(1 / 6)
    assert scores["recall_100"] == 0
    assert scores["ece"] == pytest.approx(1 / 3)
    assert bins[5]["count"] == 120
    reference, reference_bins = prediction_metrics(p, y, ids, list(range(120)))
    assert reference == scores and reference_bins == bins


@pytest.mark.parametrize("positives", (False, True))
def test_single_class_metrics_and_probability_boundaries_are_preserved(positives):
    labels = np.full(6, positives)
    p = np.array([0, 1, 0.2, 0.2, 0.8, 0.8])
    actual, bins = prediction_scores(p, labels, list("abcdef"), list(range(6)))
    expected, expected_bins = prediction_metrics(p, labels, list("abcdef"), list(range(6)))
    for key in expected:
        if isinstance(expected[key], float):
            assert actual[key] == pytest.approx(expected[key], abs=1e-12)
        else:
            assert actual[key] == expected[key]
    assert bins == expected_bins


@pytest.mark.parametrize("change", ("nonbinary", "empty", "duplicate", "range", "probability"))
def test_invalid_prediction_input_cannot_produce_valid_metrics(change):
    p, labels, evaluation = [0.1, 0.2], [0, 1], [0, 1]
    if change == "nonbinary":
        labels = [0, 0.5]
    elif change == "empty":
        evaluation = []
    elif change == "duplicate":
        evaluation = [0, 0]
    elif change == "range":
        evaluation = [0, 2]
    else:
        p[1] = float("nan")
    with pytest.raises(ValueError):
        prediction_scores(p, labels, ["a", "b"], evaluation)


class Coast:
    def act(self, _):
        return Control(0.0, 0.0)


class NativeBank:
    def __init__(self):
        self.candidates = sample("car_following", 8, "new", replicate=989)
        bounds = DOMAINS["car_following"].bounds
        index = initial_indices(8, 2, 11)[0]
        self.candidates[index] = replace(
            self.candidates[index],
            parameters={
                "ego_speed_mps": bounds["ego_speed_mps"][1],
                "lead_speed_mps": bounds["lead_speed_mps"][0],
                "gap_m": bounds["gap_m"][0],
            },
        )
        self.manifest = {
            "sut_id": "reader_fixture",
            "environment_seed": 81001,
            "contract": {"episode_duration_s": 0.4},
        }
        self.domain = None
        self.executions = {
            s.scenario_id: execute(Coast(), s, duration_s=0.4) for s in self.candidates
        }

    @property
    def snapshot_fingerprint(self):
        return hashlib.sha256(
            json.dumps(
                [
                    {"feedback": p.feedback.to_dict(), "trajectory": p.trajectory}
                    for p in self.executions.values()
                ],
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def __call__(self, scene):
        return self.executions[scene.scenario_id]


@pytest.fixture(scope="module")
def bank():
    torch.set_num_threads(1)
    return NativeBank()


@pytest.fixture
def prediction(bank):
    truth = BankTruth(bank)
    split = split_candidates(bank.candidates, 11, budget=4)
    spec = prediction_specification(bank, split, 11, "target_only", None, {}, None, (0, 2, 4))
    ledger = QueryLedger(bank.candidates, 4, bank, joint_feedback=True, allow_registration=False)
    result = select(
        bank.candidates,
        ledger,
        query_order=split["query_order"],
        initial_indices=split["query_order"][:1],
        budget=4,
        snapshot_counts=(0, 2, 4),
        raw_kernel_type=SharedPrivateKernel,
    )
    metrics = []
    for snapshot in result["prediction_snapshots"]:
        values, calibration = prediction_metrics(
            snapshot["probabilities"],
            truth.labels,
            split["candidate_ids"],
            split["evaluation_indices"],
        )
        metrics.append({"K": snapshot["paid_queries"], **values, "calibration": calibration})
    return (
        {
            "specification": spec,
            "result": compact_result(result),
            "metrics": metrics,
            "query_ledger": compact_feedback(ledger.records),
        },
        copy.deepcopy(spec),
        truth,
    )


def make_budget(bank, *, reverse_initial=False):
    truth = BankTruth(bank)
    initial = initial_indices(8, 2, 11)
    if reverse_initial:
        initial.reverse()
    ledger = QueryLedger(
        bank.candidates,
        4,
        lambda scene: disclose_driving_quality(disclose_impact(bank(scene))),
        joint_feedback=True,
        allow_registration=False,
    )
    result = random_select(bank.candidates, ledger, initial_indices=initial, budget=4, seed=11)
    spec = {
        "bank_contract": bank.manifest["contract"],
        "candidates": [asdict(s) for s in bank.candidates],
        "budget": 4,
        "initial_count": 2,
        "seed": 11,
        "task_snapshot": {
            "bank_snapshot": bank.snapshot_fingerprint,
            "method_contract": {"method": "random"},
        },
    }
    metrics = summarize(
        bank.candidates,
        truth.labels.tolist(),
        result["queries"],
        ledger.trajectories,
        initial_count=2,
        query_ledger=ledger.records,
    )
    return (
        {
            "specification": spec,
            "result": compact_result(result),
            "metrics": metrics,
            "query_ledger": compact_feedback(ledger.records),
            "evaluation": {"candidate_collisions": truth.labels.tolist()},
        },
        copy.deepcopy(spec),
        truth,
    )


def test_prediction_records_are_read_against_native_truth_and_approved_inputs(prediction):
    saved, spec, truth = prediction
    rows = prediction_record(saved, spec, truth, method="target_only", seed=11)
    assert [r["K"] for r in rows] == [0, 2, 4]
    assert all(r["pool_collisions"] == int(truth.labels.sum()) for r in rows)
    assert truth.profile["collisions"] >= 1


@pytest.mark.parametrize("change", ("approval", "feedback", "metric", "partition"))
def test_altered_prediction_receipts_or_metrics_are_rejected(prediction, change):
    saved, spec, truth = prediction
    if change == "approval":
        saved["specification"]["feedback_seed"] = 23
    elif change == "feedback":
        saved["query_ledger"][0]["feedback"]["safety"]["rss_margin"] += 0.2
    elif change == "metric":
        saved["metrics"][0]["brier"] += 0.1
    else:
        saved["result"]["prediction_snapshots"][1]["observed_indices"].reverse()
    with pytest.raises(ValueError):
        prediction_record(saved, spec, truth, method="target_only", seed=11)


def test_budget_reader_recomputes_discovery_and_costs_from_actual_native_receipts(bank):
    saved, spec, truth = make_budget(bank)
    row = budget_record(saved, spec, truth, method="random", seed=11)
    assert row["metrics"]["queries"] == 4
    assert row["metrics"]["unique_collisions"] >= 1
    saved["metrics"]["discovery_auc"] += 0.25
    with pytest.raises(ValueError, match="Independent discovery"):
        budget_record(saved, spec, truth, method="random", seed=11)


def test_budget_reader_checks_actual_initial_ids_not_only_the_number_of_discoveries(bank):
    saved, spec, truth = make_budget(bank, reverse_initial=True)
    with pytest.raises(ValueError, match="exact common initialization candidates"):
        budget_record(saved, spec, truth, method="random", seed=11)


@pytest.mark.parametrize("change", ("clock", "label"))
def test_inconsistent_native_trace_is_not_converted_to_a_safe_response(bank, change):
    altered = copy.copy(bank)
    altered.executions = copy.deepcopy(bank.executions)
    paid = next(iter(altered.executions.values()))
    if change == "clock":
        paid.trajectory[1]["time_s"] = 0.1
    else:
        paid.trajectory[0]["collision"] = 0
    with pytest.raises(ValueError):
        BankTruth(altered)
