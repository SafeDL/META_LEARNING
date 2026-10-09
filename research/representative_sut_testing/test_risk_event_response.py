"""Event correction preserves risk inference and permits nonmonotone outcomes."""
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr
from sklearn.isotonic import IsotonicRegression

from .risk_event_response import RiskEventResponse, integrated_event_probability


def build():
    reference = SimpleNamespace(session=SimpleNamespace(gp=SimpleNamespace(
        covariance=np.eye(3), mean=np.asarray([.5, .5, .5]), noise=.0025)))
    readout = {g: IsotonicRegression(out_of_bounds="clip").fit([0, 1], [.5, .5]) for g in (0, 1)}
    return RiskEventResponse(reference, [0, 0, 1], readout)


def test_uncorrected_event_marginal_matches_historical_readout():
    model = build()
    np.testing.assert_allclose(model.probabilities(), [.5] * 3, atol=1e-12)


def test_same_risk_can_receive_opposite_event_corrections():
    model = build()
    model.observe(0, .5, True)
    model.observe(1, .5, False)
    probability = model.probabilities()
    assert probability[0] > .5 > probability[1]
    np.testing.assert_allclose(probability[2], .5, atol=1e-12)
    np.testing.assert_allclose(model.alpha, -model.alpha[::-1], atol=1e-7)


def test_event_feedback_does_not_change_risk_posterior():
    positive, negative = build(), build()
    positive.observe(0, .7, True)
    negative.observe(0, .7, False)
    positive.probabilities()
    negative.probabilities()
    np.testing.assert_array_equal(positive.gp.mean, negative.gp.mean)
    np.testing.assert_array_equal(positive.gp.variance, negative.gp.variance)


def test_single_observation_laplace_mode_satisfies_stationarity():
    model = build()
    model.observe(0, .5, True)
    model.probabilities()
    mode = model.alpha[0]
    density = np.exp(-.5 * mode ** 2) / np.sqrt(2 * np.pi)
    np.testing.assert_allclose(mode, density / ndtr(mode), atol=1e-7)


def test_shared_event_calibration_transfers_inside_family_and_preserves_variance():
    model = build()
    np.testing.assert_array_equal(np.diag(model.kernel), np.ones(3))
    assert model.kernel[0, 1] == .5 and model.kernel[0, 2] == 0
    model.observe(0, .5, True)
    probability = model.probabilities()
    assert probability[0] > probability[1] > .5
    np.testing.assert_allclose(probability[2], .5, atol=1e-12)


def test_sharp_calibration_integration_matches_exact_gaussian_expectation():
    curve = IsotonicRegression(out_of_bounds="clip").fit([0, .804, .806, 1], [.01, .01, .99, .99])
    mean = np.asarray([.77, .81, .85])
    sd = np.asarray([.3, .05, .1])
    result = integrated_event_probability(mean, sd, np.zeros(3), np.ones(3), curve)
    x, p = curve.X_thresholds_, curve.y_thresholds_
    z = (x[None, :] - mean[:, None]) / sd[:, None]
    cdf = ndtr(z)
    density = np.exp(-.5 * z ** 2) / np.sqrt(2 * np.pi)
    mass = np.diff(cdf, axis=1)
    slope = np.diff(p) / np.diff(x)
    exact = (p[0] * cdf[:, 0] + p[-1] * ndtr((mean - x[-1]) / sd)
             + ((p[:-1] - slope * x[:-1]) * mass
                + slope * (mean[:, None] * mass + sd[:, None] * (density[:, :-1] - density[:, 1:]))).sum(axis=1))
    np.testing.assert_allclose(result, exact, atol=1e-7)
