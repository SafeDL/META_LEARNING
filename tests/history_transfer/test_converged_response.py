import numpy as np
import pytest
import torch

from experiments.inference_reference import posterior_numpy
from testing_methods.common.classification import logistic_moments
from testing_methods.history_transfer.converged_response import component_probabilities


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_component_posteriors_match_independent_converged_inference_and_preserve_inputs(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.set_num_threads(1)
    generator = np.random.default_rng(1321)
    a = generator.normal(size=(2, 7, 7))
    covariance = a @ a.transpose(0, 2, 1) / 8 + 0.1 * np.eye(7)
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
    assert all(row["gradient_max"] <= 1e-10 for row in info)
    torch.testing.assert_close(m, before_m, atol=0, rtol=0)
    torch.testing.assert_close(c, before_c, atol=0, rtol=0)
    permuted, _ = component_probabilities(m[[1, 0]], c[[1, 0]], paid, labels)
    torch.testing.assert_close(probability[[1, 0]], permuted)


def test_converged_solver_handles_a_high_variance_positive_observation():
    mean = torch.tensor([[-5.0]], dtype=torch.float64)
    covariance = torch.tensor([[[100.0]]], dtype=torch.float64)
    corrected, info = component_probabilities(mean, covariance, [0], [True])
    pm, pc = posterior_numpy(mean[0].numpy(), covariance[0].numpy(), [0], [True])
    expected = logistic_moments(pm, np.diag(pc))[0]
    np.testing.assert_allclose(corrected[0], expected, atol=2e-12, rtol=2e-12)
    assert info[0]["gradient_max"] < 1e-10


def test_no_feedback_returns_the_prior_without_a_mode_solve():
    mean = torch.tensor([[-2.0, 1.0]], dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64)[None]
    p, info = component_probabilities(mean, covariance, [], [])
    expected = logistic_moments(mean.numpy(), covariance.diagonal(dim1=-2, dim2=-1).numpy())[0]
    np.testing.assert_allclose(p, expected, atol=2e-14, rtol=2e-14)
    assert info == [{"iterations": 0, "gradient_max": 0.0, "backtracks": 0}]


def test_low_variance_posterior_matches_independent_solver():
    mean = torch.tensor([[-1.0, 0.2, 0.4]], dtype=torch.float64)
    covariance = torch.eye(3, dtype=torch.float64)[None] * 0.2
    corrected, _ = component_probabilities(mean, covariance, [0, 2], [False, True])
    pm, pc = posterior_numpy(mean[0].numpy(), covariance[0].numpy(), [0, 2], [False, True])
    expected = logistic_moments(pm, np.diag(pc))[0]
    np.testing.assert_allclose(corrected[0], expected, atol=2e-12, rtol=2e-12)
