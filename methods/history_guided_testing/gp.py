"""Incremental Gaussian conditioning and exact clipped-normal acquisition."""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr


def positive_normal_part(mean, variance, threshold):
    mean, variance, threshold = np.broadcast_arrays(mean, variance, threshold)
    sd = np.sqrt(np.maximum(variance, 0))
    standardized = (mean - threshold) / np.maximum(sd, 1e-15)
    return (mean - threshold) * ndtr(standardized) + sd * np.exp(-.5 * standardized ** 2) / np.sqrt(2 * np.pi)


def clipped_risk_ei(mean, variance, threshold):
    return np.maximum(positive_normal_part(mean, variance, threshold) -
                      positive_normal_part(mean, variance, 1.), 0)


class RiskGP:
    def __init__(self, covariance, prior, noise=.0025):
        self.covariance = np.asarray(covariance)
        self.mean = np.asarray(prior, dtype=float).copy()
        self.variance = np.diag(self.covariance).copy()
        self.noise = noise
        self.indices = []
        self.factor = np.empty((0, 0))
        self.solved_cross = np.empty((len(prior), 0))

    def observe(self, index, risk):
        if index in self.indices or not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("invalid or duplicate risk observation")
        count = len(self.indices)
        if count:
            column = solve_triangular(self.factor, self.covariance[self.indices, index], lower=True)
            cross = self.covariance[:, index] - self.solved_cross @ column
            denominator = self.covariance[index, index] + self.noise + 1e-8 - column @ column
        else:
            column = np.empty(0)
            cross = self.covariance[:, index].copy()
            denominator = self.covariance[index, index] + self.noise + 1e-8
        if denominator <= 0:
            raise FloatingPointError("invalid GP conditional variance")
        factor = np.zeros((count + 1, count + 1))
        factor[:count, :count] = self.factor
        factor[count, :count] = column
        factor[count, count] = np.sqrt(denominator)
        self.factor = factor
        self.mean += cross * (risk - self.mean[index]) / denominator
        normalized = cross / np.sqrt(denominator)
        self.variance -= normalized ** 2
        if self.variance.min() < -1e-9:
            raise FloatingPointError("negative GP variance")
        self.variance = np.maximum(self.variance, 0)
        self.solved_cross = np.column_stack([self.solved_cross, normalized])
        self.indices.append(int(index))
