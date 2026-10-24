import numpy as np
import pytest
import torch

from benchmark.discovery_scaml_audit import posterior_numpy
from methods.common.classification import logistic_moments
from methods.gcrt.conditional_adaptation import component_probabilities as legacy
from methods.gcrt.converged_response import component_probabilities


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_component_posteriors_match_independent_converged_inference_and_preserve_inputs(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    torch.set_num_threads(1)
    generator = np.random.default_rng(1321)
    a = generator.normal(size=(2, 7, 7))
    covariance = a @ a.transpose(0, 2, 1)/8+.1*np.eye(7)
    mean = generator.normal(size=(2, 7))
    m = torch.tensor(mean, device=device)
    c = torch.tensor(covariance, device=device)
    paid, labels = [1, 3, 6], [False, True, False]
    before_m, before_c = m.clone(), c.clone()
    probability, info = component_probabilities(m, c, paid, labels)
    expected = []
    for mu, sigma in zip(mean, covariance):
        posterior_mean, posterior_covariance = posterior_numpy(mu, sigma, paid, labels)
        expected.append(logistic_moments(posterior_mean, np.diag(posterior_covariance))[0])
    np.testing.assert_allclose(probability.detach().cpu(), expected, atol=2e-10, rtol=2e-10)
    assert all(row['gradient_max'] <= 1e-10 for row in info)
    torch.testing.assert_close(m, before_m, atol=0, rtol=0)
    torch.testing.assert_close(c, before_c, atol=0, rtol=0)
    permuted, _ = component_probabilities(m[[1, 0]], c[[1, 0]], paid, labels)
    torch.testing.assert_close(probability[[1, 0]], permuted)


def test_convergence_correction_changes_the_known_finite_oscillating_case():
    mean = torch.tensor([[-5.]], dtype=torch.float64)
    covariance = torch.tensor([[[100.]]], dtype=torch.float64)
    old = legacy(mean, covariance, [0], [True])
    corrected, info = component_probabilities(mean, covariance, [0], [True])
    pm, pc = posterior_numpy(mean[0].numpy(), covariance[0].numpy(), [0], [True])
    expected = logistic_moments(pm, np.diag(pc))[0]
    np.testing.assert_allclose(corrected[0], expected, atol=2e-12, rtol=2e-12)
    assert abs(float(old[0, 0]-corrected[0, 0])) > 1e-3
    assert info[0]['gradient_max'] < 1e-10


def test_no_feedback_returns_the_prior_without_a_mode_solve():
    mean = torch.tensor([[-2., 1.]], dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64)[None]
    p, info = component_probabilities(mean, covariance, [], [])
    expected = logistic_moments(mean.numpy(), covariance.diagonal(dim1=-2, dim2=-1).numpy())[0]
    np.testing.assert_allclose(p, expected, atol=2e-14, rtol=2e-14)
    assert info == [{'iterations': 0, 'gradient_max': 0., 'backtracks': 0}]


def test_low_variance_converged_legacy_case_is_numerically_unchanged():
    mean = torch.tensor([[-1., .2, .4]], dtype=torch.float64)
    covariance = torch.eye(3, dtype=torch.float64)[None]*.2
    corrected, _ = component_probabilities(mean, covariance, [0, 2], [False, True])
    torch.testing.assert_close(corrected, legacy(mean, covariance, [0, 2], [False, True]), atol=2e-12, rtol=2e-12)
