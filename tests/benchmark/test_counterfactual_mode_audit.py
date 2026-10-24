import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from methods.common.classification import logistic_moments
from benchmark.discovery_conditional_adaptation import probabilities_numpy
from benchmark.discovery_counterfactual_mode_audit import legacy_prediction


def test_mode_check_reproduces_registered_ten_step_predictions():
    generator = np.random.default_rng(1313)
    a = generator.normal(size=(6, 6))
    covariance = a @ a.T/10+.1*np.eye(6)
    mean = generator.normal(size=6)/2
    actual, _, mode = legacy_prediction(mean, covariance, [0, 2, 5], [False, True, False])
    expected = probabilities_numpy(mean[None], covariance[None], [0, 2, 5], [False, True, False])[0]
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    assert mode['converged_within_audit_tolerance']


def test_finite_but_oscillating_mode_is_rejected_and_checked_against_analytic_solution():
    mean, covariance = np.array([-5.]), np.array([[100.]])
    old, stable, result = legacy_prediction(mean, covariance, [0], [True])
    optimum = brentq(lambda latent: (latent+5)/(100+1e-5)+expit(latent)-1, -5, 10)
    assert not result['converged_within_audit_tolerance']
    assert result['whitened_gradient_max'] > 1e-6
    assert result['stable_reference_gradient_max'] < 5e-9
    assert np.isfinite(old).all() and np.isfinite(stable).all()
    assert result['stable_probability_max_difference'] > 1e-3
    assert abs((optimum+5)/(100+1e-5)+expit(optimum)-1) < 1e-12
    alpha = (optimum+5)/(100+1e-5)
    weight = expit(optimum)*(1-expit(optimum))
    predicted_mean = -5+100*alpha
    variance = 100-100**2*weight/(1+(100+1e-5)*weight)
    expected = logistic_moments(predicted_mean, variance)[0]
    np.testing.assert_allclose(stable, [expected], atol=2e-13, rtol=2e-13)
