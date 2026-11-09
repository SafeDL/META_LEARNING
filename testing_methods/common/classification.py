"""Bernoulli-logistic Gaussian moment updates with Gauss-Hermite quadrature."""

import numpy as np
import torch

NODES, WEIGHTS = np.polynomial.hermite.hermgauss(32)
WEIGHTS = WEIGHTS / np.sqrt(np.pi)


def torch_moments(mean, variance):
    nodes = torch.as_tensor(NODES, dtype=mean.dtype, device=mean.device)
    weights = torch.as_tensor(WEIGHTS, dtype=mean.dtype, device=mean.device)
    z = mean[..., None] + torch.sqrt(2 * variance.clamp_min(1e-12))[..., None] * nodes
    p = z.sigmoid()
    first = p * (1 - p)
    return (p @ weights, first @ weights, (first * (1 - 2 * p)) @ weights)
