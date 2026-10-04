import numpy as np
import pytest
from scipy.integrate import quad
import torch

from methods.history_guided_testing.gp import RiskGP, clipped_risk_ei
from methods.history_guided_testing.kernel import MODES, RiskKernel, conditional_risk


@pytest.mark.parametrize("mode", MODES)
def test_incremental_matches_batch_and_kernel_is_psd(mode):
    rng = np.random.default_rng(17)
    x = rng.random((20, 5))
    x[:, 4] = np.arange(20) >= 10
    prior = rng.uniform(.1, .8, 20)
    kernel = RiskKernel(mode)
    covariance = kernel.covariance(x).detach().numpy()
    assert np.linalg.eigvalsh(covariance).min() > -1e-10
    np.testing.assert_array_equal(covariance[:10, 10:], 0)
    gp = RiskGP(covariance, prior)
    risks = np.array([1., 0., .99, .3, .5])
    for count in range(6):
        if count:
            gp.observe(count - 1, risks[count - 1])
        mean, batch = conditional_risk(kernel, x, prior, risks[:count], count)
        np.testing.assert_allclose(mean.detach().numpy(), gp.mean[count:], atol=1e-8)
        np.testing.assert_allclose(batch.diag().detach().numpy(), gp.variance[count:], atol=1e-8)
    with pytest.raises(ValueError):
        gp.observe(0, .5)


@pytest.mark.parametrize("mean,variance,threshold", [(-1., .2, 0.), (.3, .01, .5), (1.1, .1, .7), (.9, 0., .5)])
def test_clipped_normal_acquisition_matches_integration(mean, variance, threshold):
    if variance:
        points = [(threshold - mean) / np.sqrt(variance), (1 - mean) / np.sqrt(variance)]
        expected = quad(lambda t: max(np.clip(mean + np.sqrt(variance) * t, 0, 1) - threshold, 0) *
                        np.exp(-t * t / 2) / np.sqrt(2 * np.pi), -12, 12, points=points)[0]
    else:
        expected = max(np.clip(mean, 0, 1) - threshold, 0)
    np.testing.assert_allclose(clipped_risk_ei(mean, variance, threshold), expected, atol=1e-8)
