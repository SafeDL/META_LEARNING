"""Validate the model-error likelihood and its differentiable parameters."""
import math

import torch

from .calibration import risk_error_nll


def test_risk_likelihood_equals_independent_multivariate_gaussian_density():
    x = torch.tensor([[0., 0.], [0.1, 0.2], [0.4, 0.3]], dtype=torch.float64)
    distance = torch.cdist(x, x)[None, :]
    residual = torch.tensor([[0.05, -0.02, 0.1]], dtype=torch.float64)
    log_parameters = torch.tensor([0.02, 0.005, 0.3],
                                  dtype=torch.float64).log()
    scaled = math.sqrt(5) * distance[0] / 0.3
    covariance = 0.02 * (1 + scaled + scaled.square() / 3) * (-scaled).exp()
    covariance += 0.005 * torch.eye(3, dtype=torch.float64)
    reference = -torch.distributions.MultivariateNormal(
        torch.zeros(3, dtype=torch.float64), covariance).log_prob(
            residual[0]) / 3
    torch.testing.assert_close(risk_error_nll(distance, residual,
                                              log_parameters),
                               reference,
                               atol=1e-12,
                               rtol=0)
    assert torch.autograd.gradcheck(
        lambda value: risk_error_nll(distance, residual, value),
        (log_parameters.requires_grad_(), ))
