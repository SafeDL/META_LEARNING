"""Bernoulli-logistic Gaussian moment updates with Gauss-Hermite quadrature."""

import numpy as np
import torch
from scipy.special import expit

NODES, WEIGHTS = np.polynomial.hermite.hermgauss(32)
WEIGHTS = WEIGHTS / np.sqrt(np.pi)


def logistic_moments(mean, variance):
    z = np.asarray(mean)[..., None] + np.sqrt(2 * np.maximum(variance, 0))[..., None] * NODES
    p = expit(z)
    derivative = p * (1 - p)
    return (p @ WEIGHTS, derivative @ WEIGHTS, (derivative * (1 - 2 * p)) @ WEIGHTS)


def torch_moments(mean, variance):
    nodes = torch.as_tensor(NODES, dtype=mean.dtype, device=mean.device)
    weights = torch.as_tensor(WEIGHTS, dtype=mean.dtype, device=mean.device)
    z = mean[..., None] + torch.sqrt(2 * variance.clamp_min(1e-12))[..., None] * nodes
    p = z.sigmoid()
    first = p * (1 - p)
    return (p @ weights, first @ weights, (first * (1 - 2 * p)) @ weights)
