"""Joint sites preserve previous evidence and are insensitive to input order."""
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from .ep_margin_response import ExpectationPropagatedMargin


def build():
    reference = SimpleNamespace(session=SimpleNamespace(gp=SimpleNamespace(
        covariance=np.asarray([[1., .8, .2], [.8, 1., .2], [.2, .2, 1.]]), noise=.1)))
    prior = {"mean": np.full(3, .3), "report_variance": np.ones(3), "modes": np.zeros((3, 6))}
    return ExpectationPropagatedMargin(reference, [0] * 3, prior, continuous_safe=True, device="cpu")


def test_ep_one_censor_after_numeric_data_matches_exact_scalar_posterior_moments():
    model = build()
    model.observe(1, .2, False)
    mean, variance, noise = float(model.mean[0]), float(model.covariance[0, 0]), float(model.noise[0])
    probability = norm.cdf(0, mean, np.sqrt(variance + noise))
    def density(value):
        return norm.pdf(value, mean, np.sqrt(variance)) * norm.cdf(-value / np.sqrt(noise)) / probability
    expected_mean = quad(lambda x: x * density(x), -np.inf, np.inf)[0]
    expected_second = quad(lambda x: x * x * density(x), -np.inf, np.inf)[0]
    model.observe(0, 1., True)
    np.testing.assert_allclose(float(model.mean[0]), expected_mean, atol=2e-6)
    np.testing.assert_allclose(float(model.covariance[0, 0]), expected_second - expected_mean ** 2, atol=2e-6)


def test_joint_refitting_gives_same_posterior_for_the_same_data_in_different_orders():
    data = [(0, 1., True), (1, .2, False), (2, .8, True)]
    forward, reverse = build(), build()
    for index, risk, event in data:
        forward.observe(index, risk, event)
    for index, risk, event in reversed(data):
        reverse.observe(index, risk, event)
    np.testing.assert_allclose(forward.mean.numpy(), reverse.mean.numpy(), atol=2e-6)
    np.testing.assert_allclose(forward.covariance.numpy(), reverse.covariance.numpy(), atol=2e-6)
    assert np.linalg.eigvalsh(forward.covariance.numpy()).min() >= -1e-10
    assert max(r['relative_site_change'] for r in forward.ep_updates) < 1e-7
