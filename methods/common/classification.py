"""Bernoulli-logistic Gaussian moment updates with Gauss-Hermite quadrature."""

import numpy as np
from scipy.special import expit
import torch


NODES, WEIGHTS = np.polynomial.hermite.hermgauss(32)
WEIGHTS = WEIGHTS / np.sqrt(np.pi)


def logistic_moments(mean, variance):
    z = np.asarray(mean)[..., None] + np.sqrt(2 * np.maximum(variance, 0))[..., None] * NODES
    p = expit(z)
    derivative = p * (1 - p)
    return (p @ WEIGHTS, derivative @ WEIGHTS, (derivative * (1 - 2 * p)) @ WEIGHTS)


def site_derivatives(mean, variance, label):
    if type(label) is not bool:
        raise TypeError("A Bernoulli update requires the actual Boolean collision label")
    p, first, second = logistic_moments(mean, variance)
    sign = 1 if label else -1
    likelihood = np.maximum(p if label else 1 - p, 1e-15)
    gradient = sign * first / likelihood
    curvature = sign * second / likelihood - gradient**2
    return np.log(likelihood), gradient, curvature


def torch_moments(mean, variance):
    nodes = torch.as_tensor(NODES, dtype=mean.dtype, device=mean.device)
    weights = torch.as_tensor(WEIGHTS, dtype=mean.dtype, device=mean.device)
    z = mean[..., None] + torch.sqrt(2 * variance.clamp_min(1e-12))[..., None] * nodes
    p = z.sigmoid()
    first = p * (1 - p)
    return (p @ weights, first @ weights, (first * (1 - 2 * p)) @ weights)


def torch_condition(mean, covariance, index, label):
    p, first, second = torch_moments(mean[:, index], covariance[:, index, index])
    sign = 2 * label - 1
    likelihood = (p if label else 1 - p).clamp_min(1e-15)
    gradient = sign * first / likelihood
    curvature = sign * second / likelihood - gradient.square()
    cross = covariance[:, :, index]
    return (mean + cross * gradient[:, None],
            covariance + curvature[:, None, None] * cross[:, :, None] * cross[:, None, :],
            likelihood.log())
