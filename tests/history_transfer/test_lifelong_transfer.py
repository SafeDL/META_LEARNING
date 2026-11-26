import numpy as np
import pytest
import torch

from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from experiments.transfer_settings import native_settings
from testing_methods.history_transfer.inference import JointPosterior
from testing_methods.history_transfer.lifelong_archive import (
    WeightedJointPosterior, archive_weights, effective_size, source_weights,
)
from testing_methods.history_transfer.lifelong_reliability import (
    effective_weights, logistic_moments, mixture_moments, risk_beta, target_floor,
)
from testing_methods.history_transfer.lifelong_selection import select
from testing_methods.history_transfer.lifelong_training import (
    ANCHOR, anchor_penalty, gate_decision, round_robin, validation_folds,
)
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


def test_weighted_archives_keep_the_random_prefix_and_use_ess():
    weights, anchor = archive_weights(
        np.array([True] * 3 + [False] * 19 + [True, True]),
        np.array([0.5] * 20 + [0.95, 0.9, 0.1, 0.0]),
    )
    np.testing.assert_equal(weights[:20], np.ones(20))
    np.testing.assert_allclose(weights[20:], [(4 / 22) / 0.95, (4 / 22) / 0.9, 1, 1])
    assert anchor == pytest.approx(np.log(4 / 18))
    assert effective_size(weights) < len(weights)
    priors = source_weights((np.ones(500), weights))
    assert priors.sum() == pytest.approx(1)
    assert priors[0] / priors[1] == pytest.approx(500 / effective_size(weights))
    visits = np.bincount(list(round_robin((np.ones(500), weights), 300)))
    assert visits[0] > visits[1] > 0


def test_weighted_joint_posterior_matches_unit_weights_and_remains_positive():
    torch.set_num_threads(1)
    model = SharedPrivateKernel(1, deep=False).double()
    x = np.array([[0.1], [0.5], [0.8]])
    mean, covariance = model.point_mean(3), model.kernel(x)
    labels, margins = [False, True, True], [0.2, -0.3, -0.4]
    ordinary = JointPosterior(mean, covariance, labels, margins)
    weighted = WeightedJointPosterior(mean, covariance, labels, margins, [1, 1, 1])
    torch.testing.assert_close(weighted.nll, ordinary.nll, rtol=0, atol=1e-12)
    for actual, expected in zip(
        weighted.predict(mean, covariance, model.point_covariance(3)),
        ordinary.predict(mean, covariance, model.point_covariance(3)),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    weighted = WeightedJointPosterior(mean, covariance, labels, margins, [1, 0.15, 0.15])
    _, blocks = weighted.predict(mean, covariance, model.point_covariance(3))
    assert (torch.linalg.eigvalsh(blocks) > 0).all()
    weighted.nll.backward()
    assert all(parameter.grad.isfinite().all() for parameter in model.parameters())


def test_spatial_floor_and_epistemic_acquisition_are_finite():
    dtype = torch.float64
    logits = torch.tensor([0, -1, -900, -901], dtype=dtype)
    floor = target_floor(logits)
    assert floor[-2:].sum().item() == pytest.approx(0.05)
    assert floor[-2] / floor[-1] == pytest.approx(np.e)
    torch.testing.assert_close(floor.sum(), torch.ones((), dtype=dtype))
    locations = torch.tensor([[0.0], [1.0]], dtype=dtype)
    paid = torch.repeat_interleave(locations, 20, dim=0)
    residuals = [torch.tensor([-1, -2, -2, -2], dtype=dtype)] * 20
    residuals += [torch.tensor([-2, -1, -2, -2], dtype=dtype)] * 20
    spatial = effective_weights(torch.zeros(4, dtype=dtype), locations, paid, residuals, 0.1,
                                spatial=True, batch_size=1)
    assert spatial[0, 0] > spatial[1, 0] and spatial[1, 1] > spatial[0, 1]
    p, variance = logistic_moments(torch.tensor([-16, 0, 16], dtype=dtype),
                                   torch.tensor([0, 1, 10], dtype=dtype))
    assert ((0 <= p) & (p <= 1) & (variance >= 0)).all()
    mean, sigma = mixture_moments(torch.tensor([0.5, 0.5]), torch.tensor([0.2, 0.8]),
                                  torch.tensor([0.01, 0.03]))
    assert mean.item() == pytest.approx(0.5)
    assert sigma.item() == pytest.approx(np.sqrt(0.11))
    assert risk_beta(21, 200, 20, 0.1) > risk_beta(21, 200, 20, 0.9)
    assert risk_beta(200, 200, 20, 0.1) == 0


def test_anchor_gradient_and_gate_branches():
    parent, candidate = torch.nn.Linear(1, 1).double(), torch.nn.Linear(1, 1).double()
    candidate.load_state_dict(parent.state_dict())
    with torch.no_grad():
        candidate.weight.add_(2)
    anchor_penalty((candidate,), (parent,)).backward()
    torch.testing.assert_close(candidate.weight.grad, torch.full_like(candidate.weight, 2 * ANCHOR))
    assert parent.weight.grad is None
    assert gate_decision((candidate,), 1, 0.997)
    assert not gate_decision((candidate,), 1, 0.999)
    with torch.no_grad():
        candidate.weight.fill_(float("nan"))
    assert not gate_decision((candidate,), 1, 0.99)


def test_validation_splits_shared_coordinates_identically_and_keeps_archive_initialization():
    from types import SimpleNamespace

    scenes = sample("car_following", 50, "history")
    paid = sample("car_following", 50, "new")
    archives = [
        SimpleNamespace(scenarios=scenes, provenance={}),
        SimpleNamespace(scenarios=scenes[::-1], provenance={}),
        SimpleNamespace(scenarios=paid, provenance={"completed_task": True}),
    ]
    first, second, third = validation_folds(archives)
    assert {scenes[i].coordinate() for i in first[1]} == {
        scenes[49 - i].coordinate() for i in second[1]
    }
    assert third[0][:20].tolist() == list(range(20))
    assert len(third[1]) == 10


def test_risk_ucb_only_replays_the_same_probabilities_as_original_on_shared_queries():
    from testing_methods.history_transfer.selection import select as original

    scenes = sample("car_following", 7, "new")
    model = SharedPrivateKernel(features(scenes).shape[1]).double()
    sources = {
        "names": ["a"], "weights": np.ones(1),
        "means": np.zeros((1, len(scenes), 2)),
        "blocks": np.broadcast_to(np.eye(2), (1, len(scenes), 2, 2)).copy(),
        "specification": {},
    }

    class Oracle:
        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id, index == 1,
                SafetyMeasurements(1, 1, rss_margin=0.4 - 0.5 * index), 0,
                measurement_contract(),
            )

    common = dict(
        candidates=scenes, pretrained=model, sources=sources, budget=3,
        initial_indices=[0], query_order=[0, 1, 2], snapshot_counts=(0, 1, 2, 3),
        online_settings=native_settings().online,
    )
    old = original(oracle=Oracle(), **common)
    ucb = select(oracle=Oracle(), configuration="risk_ucb", **common)
    full = select(oracle=Oracle(), configuration="lifelong_full", **common)
    assert ucb["selected_indices"] == full["selected_indices"] == old["selected_indices"]
    for expected, actual in zip(old["prediction_snapshots"], ucb["prediction_snapshots"], strict=True):
        assert expected["physical_parameters"] == actual["physical_parameters"]
        np.testing.assert_allclose(expected["probabilities"], actual["probabilities"], rtol=0, atol=1e-12)
    for query in full["queries"]:
        assert sum(query["prequery_component_weights"][-2:]) >= 0.05 - 1e-12
        assert query["prequery_acquisition"] == pytest.approx(
            query["prequery_probability"] + query["risk_beta"] * query["prequery_epistemic_standard_deviation"]
        )
