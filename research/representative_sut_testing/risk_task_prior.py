"""Risk-only task variation from the same frozen historical source split."""
import math

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from methods.history_guided_testing.history import historical_risk


class TaskMeanRiskGP:
    """Universal-kriging mean correction for functional families already observed."""
    def __init__(self, covariance, prior, family, noise):
        self.covariance = np.asarray(covariance)
        self.prior = np.asarray(prior).copy()
        self.family = np.asarray(family, dtype=int)
        self.noise = noise
        self.mean = self.prior.copy()
        self.variance = np.diag(self.covariance).copy()
        self.indices, self.values = [], []
        self.offsets = {}

    def observe(self, index, risk):
        if index in self.indices or not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid or duplicate task-mean risk feedback")
        self.indices.append(int(index))
        self.values.append(float(risk))
        indices = np.asarray(self.indices)
        matrix = self.covariance[np.ix_(indices, indices)] + (self.noise + 1e-8) * np.eye(len(indices))
        factor = np.linalg.cholesky(matrix)
        cross = self.covariance[:, indices]
        residual = np.asarray(self.values) - self.prior[indices]
        groups = np.unique(self.family[indices])
        observed_design = (self.family[indices, None] == groups[None, :]).astype(float)
        target_design = (self.family[:, None] == groups[None, :]).astype(float)
        solved_design = cho_solve((factor, True), observed_design)
        precision = observed_design.T @ solved_design
        trend_factor = np.linalg.cholesky(precision)
        trend = cho_solve((trend_factor, True), observed_design.T @ cho_solve((factor, True), residual))
        alpha = cho_solve((factor, True), residual - observed_design @ trend)
        self.residual_quadratic = float((residual - observed_design @ trend) @ alpha)
        self.observed_mean_parameters = len(groups)
        self.mean = self.prior + target_design @ trend + cross @ alpha
        solved_cross = solve_triangular(factor, cross.T, lower=True)
        prediction_design = target_design - cross @ solved_design
        trend_uncertainty = solve_triangular(trend_factor, prediction_design.T, lower=True)
        self.variance = (np.diag(self.covariance) - (solved_cross ** 2).sum(axis=0)
                         + (trend_uncertainty ** 2).sum(axis=0))
        if self.variance.min() < -1e-8:
            raise FloatingPointError("Invalid task-mean risk posterior variance")
        self.variance = np.maximum(self.variance, 0)
        self.offsets = {str(int(group)): float(value) for group, value in zip(groups, trend)}


def risk_templates(history, x):
    return np.column_stack([historical_risk({name: source}, x) for name, source in history.items()])


def add_risk_task_variation(response, templates):
    modes = (templates - templates.mean(axis=1, keepdims=True)) / math.sqrt(max(templates.shape[1] - 1, 1))
    original = response.gp
    response.gp = TaskMeanRiskGP(original.covariance + modes @ modes.T, original.mean,
                                 response.family, noise=original.noise)
    return response
