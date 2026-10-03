"""Float64 Cholesky conditioning; real target feedback never updates history."""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve_triangular

from .data import PosteriorOutput, risk_logit


def stable_cholesky(matrix, jitter=1e-8, maximum=1e-3):
    """Differentiable full-covariance Cholesky with declared jitter bound."""
    identity = torch.eye(len(matrix), dtype=matrix.dtype, device=matrix.device)
    current = jitter
    while current <= maximum * (1 + 1e-12):
        factor, info = torch.linalg.cholesky_ex(matrix + current * identity)
        if not info.any():
            return factor, current
        current *= 10
    raise FloatingPointError("Cholesky failed at maximum jitter")


def conditional_distribution(kernel, x, h, m, z_support, support_count, noise=1e-3, jitter=1e-8):
    """Training-only joint Gaussian conditional; support/query are disjoint."""
    k = kernel.matrix(x, h, x, h)
    query = slice(support_count, None)
    mean = m[query].double().reshape(-1, 1)
    covariance = k[query, query]
    if support_count:
        ks = k[:support_count, :support_count] + noise * torch.eye(support_count, device=k.device, dtype=k.dtype)
        factor, _ = stable_cholesky(ks, jitter)
        e = z_support.double().reshape(-1, 1) - m[:support_count].double().reshape(-1, 1)
        cross = k[query, :support_count]
        mean = mean + cross @ torch.cholesky_solve(e, factor)
        solve = torch.linalg.solve_triangular(factor, cross.T, upper=False)
        covariance = covariance - solve.T @ solve
    covariance = .5 * (covariance + covariance.T)
    covariance = covariance + noise * torch.eye(len(mean), device=k.device, dtype=k.dtype)
    return mean, covariance


def gaussian_nll(mean, covariance, target, jitter=1e-8):
    factor, used = stable_cholesky(covariance, jitter)
    residual = target.double().reshape(-1, 1) - mean
    solved = torch.linalg.solve_triangular(factor, residual, upper=False)
    nll = (solved.square().sum() + 2 * factor.diag().log().sum() + len(mean) * np.log(2 * np.pi)) / (2 * len(mean))
    return nll, used


class DiscrepancyGP:
    def __init__(self, kernel, noise_variance=1e-3, jitter=1e-8, jitter_max=1e-3, epsilon=1e-4):
        if noise_variance <= 0 or not 0 < jitter <= jitter_max:
            raise ValueError("invalid fixed noise/jitter")
        self.kernel = kernel.freeze()
        self.noise, self.jitter, self.jitter_max, self.epsilon = noise_variance, jitter, jitter_max, epsilon
        self.indices, self.x, self.h, self.e = [], [], [], []
        self.factor = None
        self.used_jitter = jitter

    def observe(self, index, x, h, m, risk):
        if index in self.indices:
            raise ValueError("residual already observed")
        residual = float(risk_logit(risk, self.epsilon) - float(np.asarray(m).item()))
        self.indices.append(int(index))
        self.x.append(np.asarray(x, dtype=np.float64).reshape(-1))
        self.h.append(np.asarray(h, dtype=np.float64).reshape(-1))
        self.e.append(residual)
        with torch.no_grad():
            covariance = self.kernel.matrix(np.array(self.x), np.array(self.h), np.array(self.x), np.array(self.h))
            matrix = covariance + self.noise * torch.eye(len(self.x), dtype=covariance.dtype, device=covariance.device)
            factor, self.used_jitter = stable_cholesky(matrix, self.jitter, self.jitter_max)
            self.factor = factor.cpu().numpy()

    def predict(self, x, h, m):
        m = np.asarray(m, dtype=np.float64).reshape(-1, 1)
        with torch.no_grad():
            variance = self.kernel.diagonal(x, h).cpu().numpy()
            if not self.indices:
                return PosteriorOutput(m.copy(), variance[:, None])
            cross = self.kernel.matrix(x, h, np.array(self.x), np.array(self.h)).cpu().numpy()
        alpha = solve_triangular(self.factor.T, solve_triangular(self.factor, np.array(self.e), lower=True), lower=False)
        mean = m + (cross @ alpha)[:, None]
        solved = solve_triangular(self.factor, cross.T, lower=True)
        variance = variance - np.sum(solved * solved, axis=0)
        if np.min(variance) < -1e-8:
            raise FloatingPointError("negative latent variance beyond tolerance")
        return PosteriorOutput(mean, np.maximum(variance, 0)[:, None])


class PoolDiscrepancyGP:
    """Same fixed-kernel Cholesky conditioning cached over a known candidate pool.

    Appends one Cholesky column and its triangularly solved cross-covariance.
    This is an exact arithmetic reorganization, not a sparse approximation.
    """
    def __init__(self, kernel, x, h, m, noise=1e-3, jitter=1e-8, epsilon=1e-4,
                 mean_calibration=None):
        kernel.freeze()
        self.m = np.asarray(m, float).reshape(-1, 1)
        with torch.no_grad():
            self.covariance = kernel.matrix(x, h, x, h).cpu().numpy()
        self.coefficient_mean = None
        if mean_calibration is not None:
            # Integrate uncertain offset/scale jointly with the residual GP.
            self.basis = np.column_stack((np.ones(len(x)), self.m[:, 0]))
            self.coefficient_mean = np.array([0., 1.])
            self.coefficient_covariance = np.diag([
                mean_calibration["offset_variance"], mean_calibration["scale_variance"]])
            self.coefficient_cross = self.coefficient_covariance @ self.basis.T
            self.covariance += self.basis @ self.coefficient_cross
        self.mean = self.m.copy()
        self.variance = np.diag(self.covariance).copy()
        self.noise, self.jitter, self.epsilon = noise, jitter, epsilon
        self.indices = []
        self.factor = np.zeros((0, 0))
        self.solved_cross = np.empty((len(x), 0))

    def predict(self):
        return PosteriorOutput(self.mean.copy(), self.variance[:, None].copy())

    def observe(self, index, risk):
        if index in self.indices:
            raise ValueError("duplicate residual")
        z = float(risk_logit(risk, self.epsilon))
        old = len(self.indices)
        if old:
            column = solve_triangular(self.factor, self.covariance[self.indices, index], lower=True)
            conditional_cross = self.covariance[:, index] - self.solved_cross @ column
            diagonal = self.covariance[index, index] + self.noise + self.jitter - column @ column
        else:
            column = np.empty(0)
            conditional_cross = self.covariance[:, index].copy()
            diagonal = self.covariance[index, index] + self.noise + self.jitter
        if diagonal <= 0 or not np.isfinite(diagonal):
            raise FloatingPointError("incremental Cholesky failed; explicit rebuild required")
        root = np.sqrt(diagonal)
        newfactor = np.zeros((old + 1, old + 1))
        newfactor[:old, :old] = self.factor
        newfactor[old, :old] = column
        newfactor[old, old] = root
        self.factor = newfactor
        innovation = z - self.mean[index, 0]
        if self.coefficient_mean is not None:
            coefficient_cross = self.coefficient_cross[:, index].copy()
            self.coefficient_mean += coefficient_cross * innovation / diagonal
            self.coefficient_covariance -= np.outer(coefficient_cross, coefficient_cross) / diagonal
            self.coefficient_cross -= np.outer(coefficient_cross, conditional_cross) / diagonal
        self.mean += (conditional_cross * innovation / diagonal)[:, None]
        normalized = conditional_cross / root
        self.variance -= normalized * normalized
        if self.variance.min() < -1e-8:
            raise FloatingPointError("negative pool latent variance")
        self.variance = np.maximum(self.variance, 0)
        self.solved_cross = np.column_stack([self.solved_cross, normalized])
        self.indices.append(int(index))
