import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from methods.historical_expert_mixture.config import JITTER, NOISE
from methods.historical_expert_mixture.posterior import EvidenceMixture
from methods.history_guided_testing.gp import RiskGP


def test_incremental_evidence_equals_joint_gaussian_likelihood():
    covariance = np.array([[.3, .1, .05], [.1, .25, .02], [.05, .02, .2]])
    priors = np.array([[.1, .8], [.2, .7], [.4, .3]])
    probability = np.array([.4, .6])
    model = EvidenceMixture(covariance, priors, probability)
    indices, values = [2, 0], [.3, .9]
    for i, value in zip(indices, values):
        model.observe(i, value)
    measured = covariance[np.ix_(indices, indices)] + (NOISE + JITTER) * np.eye(2)
    evidence = np.array([multivariate_normal.logpdf(values, priors[indices, j], measured) for j in range(2)])
    expected = np.log(probability) + evidence
    expected -= logsumexp(expected)
    np.testing.assert_allclose(model.log_probability, expected, atol=1e-12)
    np.testing.assert_allclose(sum(r["mixture_log_density"] for r in model.evidence),
                               logsumexp(np.log(probability) + evidence), atol=1e-12)


def test_shared_conditioning_matches_separate_frozen_gps():
    rng = np.random.default_rng(31)
    a = rng.normal(size=(12, 4))
    covariance = a @ a.T / 20 + np.eye(12) * .01
    priors = rng.uniform(0, 1, (12, 3))
    mixture = EvidenceMixture(covariance, priors, np.ones(3) / 3)
    separate = [RiskGP(covariance, priors[:, j]) for j in range(3)]
    for index, risk in [(3, .4), (8, .9), (0, .2)]:
        mixture.observe(index, risk)
        for model in separate:
            model.observe(index, risk)
    np.testing.assert_allclose(mixture.mean, np.column_stack([m.mean for m in separate]), atol=1e-13)
    np.testing.assert_allclose(mixture.variance, separate[0].variance, atol=1e-13)


def test_feedback_can_reverse_risk_at_another_scene():
    prior = np.array([[.9, .1], [.1, .9]])
    model = EvidenceMixture(np.eye(2) * .001, prior, np.ones(2) / 2)
    before = model.predict()[0][1]
    model.observe(0, .9)
    assert model.predict()[0][1] < before - .3
    assert model.probability[0] > .99


def test_total_variance_includes_between_source_disagreement():
    model = EvidenceMixture(np.eye(2) * .2, np.array([[.1, .9], [.3, .7]]), [.25, .75])
    mean, variance = model.predict()
    np.testing.assert_allclose(mean, [.7, .6])
    np.testing.assert_allclose(variance, [.32, .23])


def test_one_component_acquisition_and_posterior_equal_reference():
    from methods.history_guided_testing.gp import clipped_risk_ei
    covariance = np.eye(3) * .25 + .03
    prior = np.array([.3, .8, .2])
    model = EvidenceMixture(covariance, prior[:, None], [1])
    reference = RiskGP(covariance, prior)
    for index, risk in [(1, .9), (0, .1)]:
        model.observe(index, risk)
        reference.observe(index, risk)
    np.testing.assert_allclose(model.acquisition(), clipped_risk_ei(reference.mean, reference.variance, 0))

