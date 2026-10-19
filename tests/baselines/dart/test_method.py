"""Decision semantics, numerical conditioning and paid-feedback boundaries."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import stdtr
import torch

from methods.baselines.cg_brt import select as original_select
from methods.baselines.cg_brt.decision import set_inclusion
from methods.baselines.cg_brt.posterior import RiskPosterior
from methods.baselines.dart.encoder import ResponseSetEncoder
from methods.baselines.dart.fantasy import (
    conditional_parameters,
    fantasy_probabilities,
)
from methods.baselines.dart.objectives import (
    conditioned,
    event_probability,
    training_loss,
)
from methods.baselines.dart.planning import (
    acquisition_values,
    choose,
    replacement_losses,
    shortlist,
)
from methods.baselines.dart.selection import select

torch.set_num_threads(1)


def problem(size=18):
    coordinate = np.linspace(0, 1, size)
    x = np.column_stack(
        (
            coordinate,
            coordinate * 0.4,
            coordinate * 0.2,
            coordinate * 0.1,
            np.arange(size) % 2,
        )
    )
    covariance = 0.04 * np.exp(-(((coordinate[:, None] - coordinate) / 0.25) ** 2))
    covariance *= x[:, 4:5] == x[:, 4:5].T
    priors = np.column_stack((0.3 + 0.5 * coordinate, 0.8 - 0.5 * coordinate))
    return x, covariance, priors, np.array([0.5, 0.5]), priors.mean(1)


class Oracle:
    def __init__(self, x):
        self.risk = 0.2 + 0.7 * x[:, 0]
        self.queried = []

    def query(self, index):
        assert index not in self.queried
        self.queried.append(index)
        return SimpleNamespace(risk=float(self.risk[index]))


def test_disabled_enhancement_reproduces_original():
    x, covariance, priors, probability, fixed = problem()
    a = original_select(
        x,
        covariance,
        priors,
        probability,
        fixed,
        Oracle(x),
        budget=12,
        initial_queries=3,
    )
    b = select(
        x,
        covariance,
        priors,
        probability,
        fixed,
        Oracle(x),
        proposal_covariance=covariance,
        budget=12,
        initial_queries=3,
        enabled=False,
    )
    assert a["queries"] == b["queries"]
    assert a["evidence"] == [row["reference"] for row in b["evidence"]]


@pytest.mark.parametrize("risk", [0.63, -2.0, 3.0])
def test_pure_fantasy_matches_conditioning_and_keeps_state(risk):
    _, covariance, priors, probability, fixed = problem()
    model = RiskPosterior(covariance, priors, probability, fixed)
    model.observe(0, 0.4)
    before = copy.deepcopy(model)
    future = np.arange(2, len(fixed))
    mean, variance, alpha, beta, weights = conditional_parameters(
        model, 1, [risk], future
    )
    if 0 <= risk <= 1:
        other = copy.deepcopy(model)
        other.observe(1, risk)
        np.testing.assert_allclose(mean[0], other.source_mean[future], atol=1e-13)
        np.testing.assert_allclose(
            variance, np.diag(other.covariance_block(future)), atol=1e-13
        )
        np.testing.assert_allclose(beta[0], other.beta)
        np.testing.assert_allclose(weights[0], other.log_weight)
        assert alpha == other.alpha
        np.testing.assert_allclose(
            fantasy_probabilities(model, 1, [risk], future)[0],
            np.exp(other.log_probabilities()[0][0, future]),
            atol=1e-13,
        )
    else:
        with pytest.raises(ValueError):
            model.observe(1, risk)
        assert np.isfinite(mean).all()
    np.testing.assert_array_equal(model.source_mean, before.source_mean)
    np.testing.assert_array_equal(model.beta, before.beta)
    assert model.indices == before.indices


def test_torch_conditioning_and_quadrature_match_scipy():
    _, covariance, priors, probability, fixed = problem()
    truth = np.array([0.4, 0.63])
    model = RiskPosterior(covariance, priors, probability, fixed)
    for index, risk in enumerate(truth):
        model.observe(index, risk)
    tensors = [
        torch.as_tensor(a, dtype=torch.float64)
        for a in (covariance, priors, probability, truth)
    ]
    mean, block, alpha, beta, weight = conditioned(*tensors, 2)
    np.testing.assert_allclose(mean, model.source_mean[2:], atol=1e-13)
    np.testing.assert_allclose(
        block, model.covariance_block(np.arange(2, len(fixed))), atol=1e-13
    )
    predicted = event_probability(mean, block.diag(), alpha, beta, weight)
    np.testing.assert_allclose(
        predicted, np.exp(model.log_probabilities()[0][0, 2:]), atol=1e-4
    )
    for a in (2.5, 7.5, 27.5, 102.5):
        z = torch.linspace(-100, 100, 2001, dtype=torch.float64)
        p = event_probability(
            z[:, None] + 0.5,
            torch.ones_like(z),
            a,
            torch.tensor([a]),
            torch.tensor([0.0]),
        )
        np.testing.assert_allclose(p, stdtr(2 * a, z.numpy()), atol=1e-4)


def test_encoder_permutations_projectivity_and_positive_covariance():
    torch.manual_seed(11)
    x, covariance, _, _, _ = problem()
    contexts = {
        name: {"x": x[:10], "risk": 0.2 + 0.6 * x[:10, 0]}
        for name in ("idm_reactive", "idm_delayed")
    }
    model = ResponseSetEncoder().eval()
    with torch.no_grad():
        a = model.covariance(x, contexts, covariance).numpy()
        permuted = {
            name: {"x": c["x"][::-1].copy(), "risk": c["risk"][::-1].copy()}
            for name, c in reversed(list(contexts.items()))
        }
        b = model.covariance(x, permuted, covariance).numpy()
        subset = model.covariance(x[:5], contexts, covariance[:5, :5]).numpy()
    np.testing.assert_allclose(a, b, atol=1e-8)
    np.testing.assert_allclose(a[:5, :5], subset, atol=1e-8)
    assert np.linalg.eigvalsh(a).min() > -1e-9
    assert np.count_nonzero(a[x[:, 4, None] != x[None, :, 4]]) == 0


def test_decision_objective_has_finite_nonzero_gradients():
    x, covariance, priors, probability, _ = problem()
    contexts = {"idm_reactive": {"x": x, "risk": 0.2 + 0.7 * x[:, 0]}}
    model = ResponseSetEncoder()
    cov = model.covariance(x, contexts, covariance)
    args = [
        torch.tensor(a, dtype=torch.float64)
        for a in (priors, probability, 0.2 + 0.7 * x[:, 0])
    ]
    loss, parts = training_loss(cov, *args, 5)
    loss.backward()
    assert set(parts) == {"joint", "event", "rank", "update"}
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert abs(float(model.raw_amplitude.grad)) > 0


def test_static_inclusion_counterexample_and_last_query():
    labels = np.array([[1.0, 0.0, 0.0]] * 55 + [[0.0, 1.0, 1.0]] * 45)
    np.testing.assert_allclose(labels.mean(0), [0.55, 0.45, 0.45])
    np.testing.assert_allclose(set_inclusion(labels, 2), [0.55, 0.725, 0.725])
    _, cov, priors, prob, fixed = problem()
    model = RiskPosterior(cov, priors, prob, fixed)
    candidates = np.array([0, 3, 8])
    values, _, information = acquisition_values(
        model, candidates, np.ones(len(fixed), bool), 1, np.random.default_rng(1)
    )
    np.testing.assert_allclose(
        values, np.exp(model.log_probabilities()[0][0, candidates])
    )
    np.testing.assert_array_equal(information, 0)


def test_zero_loss_budget_retains_reference_and_queries_stay_paid():
    x, cov, priors, prob, fixed = problem()
    oracle = Oracle(x)
    result = select(
        x,
        cov,
        priors,
        prob,
        fixed,
        oracle,
        proposal_covariance=cov,
        budget=12,
        initial_queries=3,
        loss_fraction=0.0,
    )
    assert len(set(result["selected_indices"])) == 12
    assert result["selected_indices"] == oracle.queried
    assert all(d["selected_index"] == d["reference_index"] for d in result["decisions"])
    assert result["loss_budget_remaining"] == 0


def test_every_replacement_charges_its_loss_and_respects_ledger():
    x, cov, priors, prob, fixed = problem()
    result = select(
        x,
        cov,
        priors,
        prob,
        fixed,
        Oracle(x),
        proposal_covariance=cov,
        budget=12,
        initial_queries=3,
    )
    for d in result["decisions"]:
        if d["selected_index"] != d["reference_index"]:
            assert d["loss_charged"] == d["posterior_replacement_loss"]
            assert d["loss_charged"] <= d["loss_remaining_before"]
    assert sum(d["loss_charged"] for d in result["decisions"]) <= 1.2 + 1e-12


def test_batched_losses_match_original_joint_probabilities():
    _, cov, priors, prob, fixed = problem()
    model = RiskPosterior(cov, priors, prob, fixed)
    model.observe(0, 0.4)
    candidates = np.arange(1, len(fixed))
    actual = replacement_losses(model, candidates, 2)
    expected = [
        0.0 if c == 2 else model.discordance(int(c), 2)[0][1] for c in candidates
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_feasible_screen_preserves_full_candidate_virtual_draws():
    _, cov, priors, prob, fixed = problem()
    model = RiskPosterior(cov, priors, prob, fixed)
    remaining = np.ones(len(fixed), bool)
    probability = np.exp(model.log_probabilities()[0][0])
    candidates = shortlist(remaining, probability, probability, 2)
    dense, _, _ = acquisition_values(
        model, candidates, remaining, 6, np.random.default_rng(91)
    )
    _, _, record = choose(model, remaining, probability, 2, 6, 0.1, 91)
    for i, v in enumerate(record["candidate_values"]):
        if v is not None:
            assert abs(v - dense[i]) < 1e-12
