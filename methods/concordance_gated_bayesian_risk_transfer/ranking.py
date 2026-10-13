"""Deterministic correlated Gaussian threshold probabilities."""
import math

import numpy as np
from scipy.special import ndtr

from .config import CDF_NODES


NODES, WEIGHTS = np.polynomial.legendre.leggauss(CDF_NODES)


def bivariate_normal_cdf(first, second, correlation):
    first, second, correlation = np.broadcast_arrays(first, second, correlation)
    angle = np.arcsin(np.clip(correlation, -1, 1))
    theta = .5 * angle[..., None] * (NODES + 1)
    exponent = -.5 * (
        (first[..., None] - second[..., None] * np.sin(theta)) ** 2
        / np.cos(theta) ** 2 + second[..., None] ** 2
    )
    correction = angle * (np.exp(exponent) @ WEIGHTS) / (4 * math.pi)
    return np.clip(ndtr(first) * ndtr(second) + correction, 0, 1)
