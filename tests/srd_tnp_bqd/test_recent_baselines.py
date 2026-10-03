import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from methods.srd_tnp_bqd.baseline import (GaussianRiskGP, bas_scores,
                                         expected_bernoulli_variance, jackknife_variance)


def test_bas_matches_expectation_over_unknown_next_observation():
    for threshold in (-2., -.3, 0., .8):
        for fraction in (0., .2, .85, .999):
            def integrand(value):
                probability = ndtr((threshold - np.sqrt(fraction) * value) / np.sqrt(1 - fraction))
                return probability * (1 - probability) * norm.pdf(value)
            expected = quad(integrand, -10, 10, epsabs=1e-10)[0]
            np.testing.assert_allclose(expected_bernoulli_variance(threshold, fraction), expected, atol=1e-9)


def test_bas_prefers_correlated_boundary_over_irrelevant_high_variance():
    gp = GaussianRiskGP(np.arange(3.)[:, None])
    gp.mean = np.array([.5, .5, 10.])
    gp.covariance = np.array([[1., .9, 0.], [.9, 1., 0.], [0., 0., 4.]])
    score = bas_scores(gp, .5)
    assert score[0] > score[2] and score[1] > score[2]


def test_rf_jackknife_uses_pseudo_values_not_tree_dispersion():
    full = np.array([.5, .8])
    folds = np.array([[.4, .6], [.6, .9], [.5, .9]])
    pseudo = 3 * full - 2 * folds
    expected = np.var(pseudo, axis=0, ddof=1) / 3
    np.testing.assert_allclose(jackknife_variance(full, folds), expected)
