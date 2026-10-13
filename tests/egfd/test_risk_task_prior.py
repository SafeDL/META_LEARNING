"""Task covariance transfers risk feedback without changing the historical mean."""
import numpy as np

from methods.egfd.risk_task_prior import TaskMeanRiskGP


def test_task_mean_feedback_shifts_distant_same_family_and_reports_uncertainty():
    gp = TaskMeanRiskGP(np.eye(3), np.asarray([.9, .9, .9]), [0, 0, 1], .0025)
    gp.observe(0, .6)
    np.testing.assert_allclose(gp.mean, [.6, .6, .9], atol=1e-12)
    assert gp.variance[1] > 1 and gp.variance[2] == 1
    np.testing.assert_allclose(gp.offsets['0'], -.3, atol=1e-12)


def test_multiple_task_mean_observations_have_finite_nonnegative_variances():
    covariance = np.asarray([[1., .4, 0.], [.4, 1., 0.], [0., 0., 1.]])
    gp = TaskMeanRiskGP(covariance, np.asarray([.9, .9, .8]), [0, 0, 1], .0025)
    gp.observe(0, .6)
    gp.observe(1, .65)
    gp.observe(2, .7)
    assert np.isfinite(gp.mean).all() and np.isfinite(gp.variance).all()
    assert gp.variance.min() >= 0 and set(gp.offsets)=={'0','1'}
