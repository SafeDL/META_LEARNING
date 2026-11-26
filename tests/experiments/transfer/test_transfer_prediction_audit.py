"""Forecast reconstruction checks recorded state, feedback and historical inputs."""

import copy

import numpy as np
import pytest

from experiments import transfer_prediction_audit as audit
from experiments.k_shot import split_candidates
from experiments.query import QueryLedger
from experiments.rss import trajectory_margin
from experiments.storage import compact_feedback
from testing_methods.baselines.response_baselines.pooled_gp import select as pooled_select
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


@pytest.fixture(scope="module", params=("target_only", "pooled_history", "full"))
def saved_case(native_inputs, request):
    bank, _, models, sources, pooled, settings, _ = native_inputs
    method = request.param
    split = split_candidates(bank.candidates, 11, budget=4)
    ledger = QueryLedger(bank.candidates, 4, bank, joint_feedback=True)
    kwargs = {
        "query_order": split["query_order"],
        "budget": 4,
        "snapshot_counts": (0, 2, 4),
        **settings.prediction_kwargs(method),
    }
    selector = select
    if method == "pooled_history":
        # The selector calls this setting fit_settings; the runner uses the other name.
        kwargs["fit_settings"] = kwargs.pop("pooled_fit_settings")
        kwargs.update(pretrained=models[0], history=pooled, inference_backend="grouped")
        selector = pooled_select
    else:
        kwargs.update(
            initial_indices=split["query_order"][:1],
            pretrained=models[0] if method == "full" else None,
            sources=sources if method == "full" else None,
            feedback_mode="joint",
        )
        if method == "target_only":
            kwargs["raw_kernel_type"] = SharedPrivateKernel
    result = selector(bank.candidates, ledger, **kwargs)
    assert len(ledger.records) == 4
    assert all(record["status"] == "complete" for record in ledger.records)
    records = compact_feedback(ledger.records)
    margins = [
        trajectory_margin(bank.read_paid(record["index"]).trajectory)["minimum_margin"]
        for record in records
    ]
    return {
        "candidates": bank.candidates,
        "snapshots": result["prediction_snapshots"],
        "records": records,
        "margins": margins,
        "method": method,
        "models": models if method != "target_only" else (None, None),
        "pooled": pooled if method == "pooled_history" else None,
        "sources": sources if method == "full" else None,
        "settings": settings,
    }


def changed(case, key):
    return {**case, key: copy.deepcopy(case[key])}


def test_forecasts_reconstruct_without_querying_or_optimizing_again(
    saved_case, native_inputs, monkeypatch
):
    from testing_methods.history_transfer import inference

    def forbidden(*args, **kwargs):
        raise AssertionError("Forecast reconstruction cannot execute or fit")

    bank = native_inputs[0]
    monkeypatch.setattr(type(bank), "__call__", forbidden)
    monkeypatch.setattr(bank, "read_paid", forbidden)
    monkeypatch.setattr(inference, "fit_joint", forbidden)
    monkeypatch.setattr("torch.optim.Adam.step", forbidden)
    before = list(bank.touched)
    checked = audit.verify_forecasts(**saved_case)
    assert checked["maximum_probability_error"] <= 1e-10
    assert [s["K"] for s in checked["checkpoints"]] == [0, 2, 4]
    assert bank.touched == before


def test_changed_prediction_is_detected(saved_case):
    case = changed(saved_case, "snapshots")
    case["snapshots"][-1]["probabilities"][0] += 0.01
    with pytest.raises(ValueError, match="forecast differs"):
        audit.verify_forecasts(**case)


def test_changed_paid_collision_is_detected(saved_case):
    case = changed(saved_case, "records")
    feedback = case["records"][-1]["feedback"]
    feedback["collision"] = not feedback["collision"]
    with pytest.raises(ValueError, match="forecast differs"):
        audit.verify_forecasts(**case)


def test_later_paid_feedback_does_not_change_an_earlier_forecast(saved_case):
    case = changed(saved_case, "records")
    case["margins"] = list(saved_case["margins"])
    for record in case["records"][2:]:
        record["feedback"]["collision"] = not record["feedback"]["collision"]
    case["margins"][2:] = [1000.0, -1000.0]
    snapshot = next(row for row in saved_case["snapshots"] if row["paid_queries"] == 2)
    kwargs = {key: value for key, value in case.items() if key != "snapshots"}
    actual = audit.reconstruct(snapshot=snapshot, **kwargs)
    np.testing.assert_allclose(actual, snapshot["probabilities"], rtol=0, atol=1e-10)


def test_changed_feedback_order_is_detected(saved_case):
    case = changed(saved_case, "snapshots")
    case["snapshots"][-1]["observed_indices"].reverse()
    with pytest.raises(ValueError, match="another paid-feedback sequence"):
        audit.verify_forecasts(**case)


def test_missing_parameter_is_detected(saved_case):
    case = changed(saved_case, "snapshots")
    case["snapshots"][-1]["target_parameters"].pop("mean")
    with pytest.raises(ValueError, match="declared fitting configuration"):
        audit.verify_forecasts(**case)


def test_unmeasured_rss_is_rejected(saved_case):
    case = changed(saved_case, "margins")
    case["margins"][-1] = float("nan")
    with pytest.raises(ValueError, match="RSS must be finite"):
        audit.verify_forecasts(**case)


@pytest.mark.parametrize("saved_case", ["full"], indirect=True)
def test_changed_raw_branch_mass_is_detected_by_forecast_reconstruction(saved_case):
    case = changed(saved_case, "snapshots")
    weights = case["snapshots"][-1]["component_weights"]
    change = min(weights[0] / 2, 0.05)
    weights[0] -= change
    weights[-1] += change
    with pytest.raises(ValueError, match="forecast differs"):
        audit.verify_forecasts(**case)


@pytest.fixture(scope="module")
def recorded_mixture(native_inputs):
    bank, _, models, sources, _, settings, _ = native_inputs
    split = split_candidates(bank.candidates, 11, budget=4)
    ledger = QueryLedger(bank.candidates, 4, bank, joint_feedback=True)
    result = select(
        bank.candidates, ledger, query_order=split["query_order"], budget=4,
        snapshot_counts=(0, 2, 4), initial_indices=split["query_order"][:1],
        pretrained=models[0], sources=sources, feedback_mode="joint",
        online_settings=settings.online,
    )
    return result, compact_feedback(ledger.records), sources


def test_global_weights_reconstruct_only_from_prequery_forecasts_and_paid_labels(recorded_mixture):
    result, records, sources = recorded_mixture
    checked = audit.verify_weight_updates(result, records, sources)
    assert checked["queries"] == 4 and checked["maximum_weight_error"] <= 1e-10


@pytest.mark.parametrize("field", [
    "prequery_component_weights", "component_weights_after_query", "prequery_log_likelihood",
])
def test_changing_a_saved_weight_or_likelihood_breaks_the_update_chain(recorded_mixture, field):
    result, records, sources = recorded_mixture
    changed = copy.deepcopy(result)
    changed["queries"][1][field][-1] += 0.01
    with pytest.raises(ValueError, match="mixture weight state differs"):
        audit.verify_weight_updates(changed, records, sources)


def test_saved_mixture_weight_cannot_be_changed_together_with_its_forecast(recorded_mixture):
    result, records, sources = recorded_mixture
    changed = copy.deepcopy(result)
    changed["prediction_snapshots"][-1]["component_weights"][-1] += 0.01
    with pytest.raises(ValueError, match="mixture weight state differs"):
        audit.verify_weight_updates(changed, records, sources)
