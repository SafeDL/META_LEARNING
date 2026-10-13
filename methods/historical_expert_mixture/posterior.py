"""Shared Cholesky updates with pre-observation Gaussian model evidence."""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from methods.history_guided_testing.gp import clipped_risk_ei

from .config import JITTER, NOISE


class EvidenceMixture:
    def __init__(self, covariance, priors, probability):
        self.covariance = np.asarray(covariance)
        self.mean = np.asarray(priors, dtype=float).copy()
        self.variance = np.diag(covariance).copy()
        self.log_probability = np.log(np.asarray(probability, dtype=float))
        self.log_probability -= logsumexp(self.log_probability)
        self.factor = np.empty((0, 0))
        self.solved_cross = np.empty((len(priors), 0))
        self.indices = []
        self.evidence = []

    @property
    def probability(self):
        return np.exp(self.log_probability)

    def predict(self):
        mean = self.mean @ self.probability
        between = ((self.mean - mean[:, None]) ** 2) @ self.probability
        return mean, self.variance + between

    def acquisition(self):
        return clipped_risk_ei(self.mean, self.variance[:, None], 0) @ self.probability

    def observe(self, index, risk):
        if index in self.indices or not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid or duplicate target feedback")
        count = len(self.indices)
        if count:
            column = solve_triangular(self.factor, self.covariance[self.indices, index], lower=True)
            cross = self.covariance[:, index] - self.solved_cross @ column
            denominator = self.covariance[index, index] + NOISE + JITTER - column @ column
        else:
            column = np.empty(0)
            cross = self.covariance[:, index].copy()
            denominator = self.covariance[index, index] + NOISE + JITTER
        if denominator <= 0:
            raise FloatingPointError("Non-positive GP predictive variance")
        residual = risk - self.mean[index]
        log_density = -.5 * (residual ** 2 / denominator + np.log(2 * np.pi * denominator))
        before = self.probability.copy()
        mixture_log_density = float(logsumexp(self.log_probability + log_density))
        self.log_probability += log_density
        self.log_probability -= logsumexp(self.log_probability)
        self.evidence.append({"index": int(index), "probability_before": before.tolist(),
                              "component_mean_before": self.mean[index].tolist(),
                              "predictive_variance": float(denominator),
                              "log_density": log_density.tolist(), "mixture_log_density": mixture_log_density,
                              "probability_after": self.probability.tolist()})
        factor = np.zeros((count + 1, count + 1))
        factor[:count, :count] = self.factor
        factor[count, :count], factor[count, count] = column, np.sqrt(denominator)
        self.factor = factor
        self.mean += cross[:, None] * residual[None, :] / denominator
        normalized = cross / np.sqrt(denominator)
        self.variance -= normalized ** 2
        if self.variance.min() < -1e-9:
            raise FloatingPointError("Negative GP conditional variance")
        self.variance = np.maximum(self.variance, 0)
        self.solved_cross = np.column_stack([self.solved_cross, normalized])
        self.indices.append(int(index))

