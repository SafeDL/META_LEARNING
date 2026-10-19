"""Exact inverse-gamma GP scale integration for each response hypothesis."""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp, stdtr

from methods.baselines.hem.config import JITTER, NOISE


PRIOR_DF = 5


def student_log_density(residual, scale_squared, df):
    return (gammaln((df + 1) / 2) - gammaln(df / 2)
            - .5 * np.log(df * np.pi * scale_squared)
            - .5 * (df + 1) * np.log1p(residual ** 2 / (df * scale_squared)))


def positive_student_part(mean, scale, threshold, df):
    deviation = mean - threshold
    z = deviation / np.maximum(scale, 1e-15)
    density = np.exp(student_log_density(z, np.ones_like(z), df))
    return deviation * stdtr(df, z) + scale * (df + z ** 2) * density / (df - 1)


class ScaleMixture:
    def __init__(self, covariance, priors, probability):
        self.covariance = np.asarray(covariance)
        self.mean = np.asarray(priors, dtype=float).copy()
        self.variance = np.diag(covariance).copy()
        self.log_probability = np.log(np.asarray(probability, dtype=float))
        self.log_probability -= logsumexp(self.log_probability)
        self.alpha = PRIOR_DF / 2
        self.beta = np.full(priors.shape[1], (PRIOR_DF - 2) / 2)
        self.factor = np.empty((0, 0))
        self.solved_cross = np.empty((len(priors), 0))
        self.indices, self.evidence = [], []

    @property
    def probability(self):
        return np.exp(self.log_probability)

    @property
    def scale_mean(self):
        return self.beta / (self.alpha - 1)

    def predict(self):
        mean = self.mean @ self.probability
        within = self.variance[:, None] * self.scale_mean
        between = (self.mean - mean[:, None]) ** 2
        return mean, (within + between) @ self.probability

    def acquisition(self):
        scale = np.sqrt(self.variance[:, None] * self.beta / self.alpha)
        positive = positive_student_part(self.mean, scale, 0, 2 * self.alpha)
        excess = positive_student_part(self.mean, scale, 1, 2 * self.alpha)
        return np.clip(positive - excess, 0, 1) @ self.probability

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
            raise FloatingPointError("Non-positive base conditional variance")
        residual = risk - self.mean[index]
        predictive_scale = self.beta / self.alpha * denominator
        log_density = student_log_density(residual, predictive_scale, 2 * self.alpha)
        before = self.probability.copy()
        scale_before = self.scale_mean.copy()
        mixture_log_density = float(logsumexp(self.log_probability + log_density))
        self.log_probability += log_density
        self.log_probability -= logsumexp(self.log_probability)
        self.beta += .5 * residual ** 2 / denominator
        self.alpha += .5
        self.evidence.append({"index": int(index), "probability_before": before.tolist(),
                              "probability_after": self.probability.tolist(),
                              "component_mean_before": self.mean[index].tolist(),
                              "predictive_scale_squared": predictive_scale.tolist(),
                              "predictive_df": 2 * self.alpha - 1,
                              "log_density": log_density.tolist(),
                              "mixture_log_density": mixture_log_density,
                              "scale_mean_before": scale_before.tolist(),
                              "scale_mean_after": self.scale_mean.tolist()})
        factor = np.zeros((count + 1, count + 1))
        factor[:count, :count] = self.factor
        factor[count, :count], factor[count, count] = column, np.sqrt(denominator)
        self.factor = factor
        self.mean += cross[:, None] * residual[None] / denominator
        normalized = cross / np.sqrt(denominator)
        self.variance -= normalized ** 2
        if self.variance.min() < -1e-9:
            raise FloatingPointError("Negative conditional covariance")
        self.variance = np.maximum(self.variance, 0)
        self.solved_cross = np.column_stack([self.solved_cross, normalized])
        self.indices.append(int(index))
