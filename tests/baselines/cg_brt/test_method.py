"""Numerical and feedback-boundary checks for the risk-only main method."""
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.stats import invgamma, multivariate_normal, t

from methods.baselines.cg_brt.config import NOISE, THRESHOLD
from methods.baselines.cg_brt.decision import set_inclusion
from methods.baselines.cg_brt.gate import RankingGate
from methods.baselines.cg_brt.posterior import RiskPosterior
from methods.baselines.cg_brt.selection import select
from methods.baselines.hgt.gp import RiskGP


def test_source_conditioning_and_gaussian_reference_match_independent_gps():
    covariance = np.array([[.1, .03, .01], [.03, .1, .04], [.01, .04, .1]])
    priors = np.array([[.3, .8], [.4, .7], [.6, .5]])
    probability = np.array([.4, .6])
    model = RiskPosterior(covariance, priors, probability, priors @ probability)
    sources = [RiskGP(covariance, priors[:, j]) for j in range(2)]
    reference = RiskGP(covariance, priors @ probability)
    for index, risk in ((0, .72), (2, .51)):
        model.observe(index, risk)
        reference.observe(index, risk)
        for j, source in enumerate(sources):
            source.observe(index, risk)
            np.testing.assert_allclose(model.source_mean[:, j], source.mean, atol=1e-14)
        np.testing.assert_allclose(model.mean, reference.mean, atol=1e-14)
        np.testing.assert_allclose(model.variance, reference.variance, atol=1e-14)
    assert len(model.weights) == 2
    assert model.weights[1] > probability[1]
    transfer, _ = model.log_probabilities()
    expected = sum(
        weight * t.sf(THRESHOLD, df=2 * model.alpha, loc=model.source_mean[:, j],
                      scale=np.sqrt((model.variance + NOISE + 1e-8) * model.beta[j] / model.alpha))
        for j, weight in enumerate(model.weights)
    )
    np.testing.assert_allclose(np.exp(transfer[0]), expected, atol=1e-14)


def test_joint_threshold_probability_uses_correlation_and_shared_scale():
    covariance = np.array([[.08, .04], [.04, .12]])
    model = RiskPosterior(covariance, np.array([[.65], [.35]]), [1.], [.65, .35])
    transfer, _ = model.discordance(0, 1)
    block = covariance + (NOISE + 1e-8) * np.eye(2)

    def density(scale):
        sd = np.sqrt(np.diag(block) * scale)
        z = np.array([.15, -.15]) / sd
        correlation = block[0, 1] / np.sqrt(block[0, 0] * block[1, 1])
        pair = multivariate_normal.cdf([z[0], -z[1]], cov=[[1., -correlation], [-correlation, 1.]])
        return pair * invgamma.pdf(scale, a=2.5, scale=1.5)

    expected = quad(density, 0, np.inf, epsabs=1e-8)[0]
    np.testing.assert_allclose(transfer[0], expected, atol=2e-4)
    marginal = np.exp(model.log_probabilities()[0][0])
    np.testing.assert_allclose(transfer[0] - transfer[1], marginal[0] - marginal[1], atol=2e-4)
    independent = marginal[0] * (1 - marginal[1])
    assert abs(transfer[0] - independent) > .01


def test_ranking_gate_requires_both_outcomes_and_rejects_wrong_order():
    informative = np.log([[.8, .2], [.2, .8]])
    uninformative = np.log(np.full((2, 2), .5))
    gate = RankingGate()
    gate.observe(0, True, informative, uninformative)
    assert gate.trust == 0
    gate.observe(1, False, informative, uninformative)
    assert gate.trust > 0
    reverse = RankingGate()
    reverse.observe(0, True, uninformative, informative)
    reverse.observe(1, False, uninformative, informative)
    assert reverse.trust == 0


def test_budget_set_inclusion_counts_threshold_labels_and_preserves_set_size():
    draws = np.array([[.6, .7, .1, .2], [.1, .6, .7, .8]])
    inclusion = set_inclusion(draws, 2)
    np.testing.assert_allclose(inclusion, [.5, 5 / 6, 1 / 3, 1 / 3])
    np.testing.assert_allclose(inclusion.sum(), 2)
    np.testing.assert_allclose(set_inclusion(np.full((3, 4), .5), 2), .5)


def test_gate_keeps_old_predictions_from_before_their_feedback():
    gate = RankingGate()
    original = np.log([[.8, .2], [.2, .8]])
    reference = np.log(np.full((2, 2), .5))
    gate.observe(0, True, original, reference)
    changed = original.copy()
    changed[:, 0] = np.log([.001, .999])
    record = gate.observe(1, False, changed, reference)
    assert record["ranking_log_score"][0] > record["ranking_log_score"][1]
    assert gate.trust > 0


def test_selector_does_not_access_collision_and_never_repeats_queries():
    coordinate = np.linspace(0, 1, 18)
    covariance = .02 * np.exp(-((coordinate[:, None] - coordinate) / .2) ** 2)
    priors = np.column_stack((.3 + .5 * coordinate, .8 - .5 * coordinate))

    class RiskOnlyOracle:
        def query(self, index):
            return SimpleNamespace(risk=float(.2 + .7 * coordinate[index]))

    result = select(
        coordinate[:, None], covariance, priors, [.5, .5], priors.mean(axis=1),
        RiskOnlyOracle(), budget=12, initial_queries=3,
    )
    assert len(set(result["selected_indices"])) == 12
    assert all("collision" not in row for row in result["queries"])
    assert all(row["high_risk"] == (row["risk"] > .5) for row in result["queries"])
