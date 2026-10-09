"""Calibrated prior moments and mixed observation conditioning."""
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from .censored_margin_response import CensoredMarginResponse, calibrated_margin_moments, neutral_margin_prior


def build(continuous_safe=True):
    reference = SimpleNamespace(session=SimpleNamespace(gp=SimpleNamespace(
        covariance=np.asarray([[1., .4, 0.], [.4, 1., 0.], [0., 0., 1.]]), noise=.1)))
    prior = {"mean": np.asarray([.3, .3, .3]), "report_variance": np.ones(3),
             "modes": np.zeros((3, 6))}
    return CensoredMarginResponse(reference, [0, 0, 1], prior,
                                  continuous_safe=continuous_safe, device="cpu")


def test_prior_matches_event_probability_and_positive_report_mean():
    p = np.asarray([.01, .2, .5, .8, .99])
    positive = np.asarray([.6, .4, .2, .15, .1])
    mean, variance = calibrated_margin_moments(p, positive)
    np.testing.assert_allclose(ndtr(-mean / np.sqrt(variance)), p, atol=1e-12)
    for m, v, probability, expected in zip(mean, variance, p, positive):
        value = quad(lambda y: y * norm.pdf(y, m, np.sqrt(v)), 0, np.inf, epsabs=1e-12)[0]
        np.testing.assert_allclose(value / (1 - probability), expected, rtol=1e-8)


def test_neutral_prior_removes_event_ranking_and_preserves_safe_proxy_scale():
    source = {"probability": np.asarray([.01, .5, .99]), "positive_mean": np.asarray([.6, .3, .1]),
              "modes": np.zeros((3, 6))}
    neutral = neutral_margin_prior(source)
    np.testing.assert_array_equal(neutral['probability'], [.5] * 3)
    np.testing.assert_array_equal(neutral['mean'], np.zeros(3))
    np.testing.assert_array_equal(neutral['positive_mean'], source['positive_mean'])
    np.testing.assert_array_equal(neutral['modes'], source['modes'])
    np.testing.assert_allclose(np.sqrt(neutral['report_variance']) * np.sqrt(2 / np.pi), source['positive_mean'])
    np.testing.assert_array_equal(source['probability'], [.01, .5, .99])


def test_failure_uses_inequality_moments_instead_of_a_fixed_numeric_target():
    model = build()
    old_mean = model.mean.numpy().copy()
    old_covariance = model.covariance.numpy().copy()
    noise = model.noise.numpy()[0]
    report_variance = old_covariance[0, 0] + noise
    evidence = norm.cdf(0, old_mean[0], np.sqrt(report_variance))
    posterior_mean = quad(lambda f: f * norm.pdf(f, old_mean[0], np.sqrt(old_covariance[0, 0]))
                          * norm.cdf(-f / np.sqrt(noise)), -np.inf, np.inf)[0] / evidence
    second = quad(lambda f: f ** 2 * norm.pdf(f, old_mean[0], np.sqrt(old_covariance[0, 0]))
                  * norm.cdf(-f / np.sqrt(noise)), -np.inf, np.inf)[0] / evidence
    model.observe(0, .51, True)
    np.testing.assert_allclose(model.mean.numpy()[0], posterior_mean, atol=1e-8)
    np.testing.assert_allclose(model.covariance.numpy()[0, 0], second - posterior_mean ** 2, atol=1e-8)
    assert model.records[0]['observed_proxy'] is None
    assert model.probabilities()[0] > ndtr(-.3)


def test_safe_numeric_proxy_matches_gaussian_conditioning():
    model = build()
    mean = model.mean.numpy().copy()
    covariance = model.covariance.numpy().copy()
    denominator = covariance[0, 0] + model.noise.numpy()[0]
    model.observe(0, .8, False)
    np.testing.assert_allclose(model.mean.numpy(), mean + covariance[:, 0] * (.2 - mean[0]) / denominator)
    np.testing.assert_allclose(model.covariance.numpy(), covariance - np.outer(covariance[:, 0], covariance[0]) / denominator)
    assert model.records[0]['likelihood'] == 'positive_value'


def test_binary_control_ignores_safe_severity_but_hybrid_distinguishes_it():
    binary_low, binary_high = build(False), build(False)
    mixed_low, mixed_high = build(), build()
    binary_low.observe(0, .1, False); binary_high.observe(0, .9, False)
    mixed_low.observe(0, .1, False); mixed_high.observe(0, .9, False)
    np.testing.assert_array_equal(binary_low.mean.numpy(), binary_high.mean.numpy())
    assert mixed_low.probabilities()[0] < mixed_high.probabilities()[0]
    assert mixed_low.records[0]['conditional_variance_factor'] == 1


def test_failure_risk_value_is_not_used_and_sequential_covariance_stays_positive():
    first, second = build(), build()
    first.observe(0, .51, True); second.observe(0, 1., True)
    np.testing.assert_array_equal(first.mean.numpy(), second.mean.numpy())
    for model in (first, second):
        model.observe(1, .4, False)
        model.observe(2, .8, True)
        assert np.linalg.eigvalsh(model.covariance.numpy()).min() >= -1e-10
        assert len(model.records) == len(model.indices) == 3
        assert np.isfinite(model.probabilities()).all()
