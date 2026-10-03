import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from methods.srd_tnp_bqd.qd import RiskArchive, expected_archive_improvement, posterior_risk_mean


@pytest.mark.parametrize("mu,s,a", [(0, 1, .5), (0, 10, .5), (9, 5, .9999), (-9, 4, .0001),
                                      (0, .000001, .5), (4, 20, .9), (-10, 20, .3)])
def test_eai_matches_quad(mu, s, a):
    boundary = (np.log(a / (1 - a)) - mu) / s
    reference = quad(lambda v: max(expit(mu + s * v) - a, 0) * np.exp(-v * v / 2) / np.sqrt(2 * np.pi),
                     max(boundary, -10), 10, epsabs=1e-10, limit=200)[0] if boundary < 10 else 0.
    result = expected_archive_improvement(mu, s * s, a)
    assert float(result) == pytest.approx(reference, abs=1e-6)


def test_zero_variance_saturated_and_vectorized():
    a = expected_archive_improvement([0, 100, -100], [0, 0, 0], [.5, 1, .5])
    np.testing.assert_array_equal(a, [0, 0, 0])
    assert np.isfinite(expected_archive_improvement([0, 4], [1, 100], [.5, .9])).all()


def test_archive_only_measured_quality_and_same_cell_threshold():
    archive = RiskArchive()
    archive.observe(1, .2)
    assert archive.metrics()["risk_occupied_cells"] == 0
    archive.observe(1, .7)
    archive.observe(1, .6)
    assert archive.elite_quality[1] == pytest.approx(.2)
    archive.observe(1, .9)
    assert archive.elite_quality[1] == pytest.approx(.4)
    scores = expected_archive_improvement([1, 1], [1, 1], .5 + archive.elite_quality[[1, 1]])
    assert scores[0] == scores[1]


def test_posterior_risk_mean_is_expectation():
    assert float(posterior_risk_mean(0, 1)) == pytest.approx(.5, abs=1e-8)
    assert float(posterior_risk_mean(2, 4)) < expit(2)
