"""Scale marginalization, GLS degrees of freedom and predictive integration."""
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import t
from sklearn.isotonic import IsotonicRegression

from .risk_event_response import integrated_event_probability
from .risk_task_prior import TaskMeanRiskGP
from .student_risk_prior import StudentTaskRiskGP


def test_constant_family_has_correct_gls_effective_count_and_student_covariance():
    gp = StudentTaskRiskGP(np.eye(4), np.zeros(4), [0] * 4, noise=0.)
    for index, value in enumerate((.1, .3, .8)):
        gp.observe(index, value)
    np.testing.assert_allclose(gp.mean[3], .4, atol=1e-10)
    assert gp.predictive_df == 7
    # Three independent observations estimate one mean; residual sum of squares is .26.
    np.testing.assert_allclose(gp.variance[3], (1 + 1 / 3) * (3 + .26) / 5, atol=1e-8)


def test_scale_changes_with_feedback_values_but_keeps_gaussian_gls_mean():
    covariance = .01 * np.eye(4)
    quiet = StudentTaskRiskGP(covariance, np.full(4, .5), [0] * 4, noise=.001)
    surprising = StudentTaskRiskGP(covariance, np.full(4, .5), [0] * 4, noise=.001)
    gaussian = TaskMeanRiskGP(covariance, np.full(4, .5), [0] * 4, noise=.001)
    for index in range(3):
        quiet.observe(index, .5)
    for index, value in enumerate((.1, .9, .1)):
        surprising.observe(index, value)
        gaussian.observe(index, value)
    assert quiet.variance[3] < gaussian.variance[3] < surprising.variance[3]
    np.testing.assert_allclose(surprising.mean, gaussian.mean, atol=1e-12)
    assert surprising.scale_records[-1]['covariance_multiplier'] > 1


def test_student_readout_matches_independent_adaptive_density_integral():
    curve = IsotonicRegression(out_of_bounds='clip').fit([0, .6, .8, 1], [.01, .01, .99, .99])
    for df, mean, scale, residual_mean, residual_variance in ((5, .7, .15, -.5, .4), (25, .4, .3, .7, .8)):
        result = integrated_event_probability(np.array([mean]), np.array([scale]), np.array([residual_mean]),
                                              np.array([residual_variance]), curve, df)[0]
        def integrand(risk):
            mapped = curve.predict([risk])[0]
            chance = ndtr((np.sqrt(2) * ndtri(mapped) + residual_mean) / np.sqrt(1 + residual_variance))
            return chance * t.pdf((risk - mean) / scale, df) / scale
        knots = [-np.inf, *curve.X_thresholds_, np.inf]
        exact = sum(quad(integrand, a, b, epsabs=1e-10)[0] for a, b in zip(knots[:-1], knots[1:]))
        np.testing.assert_allclose(result, exact, atol=2e-7)
