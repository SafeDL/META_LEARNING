"""Measured risk elites and converged logistic-Gaussian archive improvement."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import expit


@lru_cache(maxsize=4)
def nodes_weights(count):
    return np.polynomial.legendre.leggauss(count)


def _integral(mu, s, threshold, count, tail=10):
    nodes, weights = nodes_weights(count)
    boundary = (np.log(threshold) - np.log1p(-threshold) - mu) / s
    lo, hi = np.maximum(boundary, -tail), float(tail)
    width = np.maximum(hi - lo, 0)
    v = lo[:, None] + (nodes[None, :] + 1) * width[:, None] / 2
    integrand = np.maximum(expit(mu[:, None] + s[:, None] * v) - threshold[:, None], 0)
    integrand *= np.exp(-v * v / 2) / np.sqrt(2 * np.pi)
    return (integrand @ weights) * width / 2


def expected_archive_improvement(mean, latent_var, threshold, tolerance=1e-6, nodes=64, max_nodes=256):
    mu, variance, a = np.broadcast_arrays(np.asarray(mean, float), np.asarray(latent_var, float), np.asarray(threshold, float))
    shape = mu.shape
    mu, variance, a = mu.ravel(), variance.ravel(), a.ravel()
    if not np.isfinite(mu).all() or not np.isfinite(variance).all() or not np.isfinite(a).all() or np.min(variance) < -1e-8:
        raise ValueError("invalid posterior or archive threshold")
    if np.any(a < 0):
        raise ValueError("negative quality threshold")
    answer = np.zeros_like(mu)
    deterministic = (variance <= 1e-16) & (a < 1)
    answer[deterministic] = np.maximum(expit(mu[deterministic]) - a[deterministic], 0)
    active = (variance > 1e-16) & (a < 1)
    if np.any(active):
        ix = np.flatnonzero(active)
        local_a = np.maximum(a[ix], np.finfo(float).tiny)
        s = np.sqrt(variance[ix])
        previous = _integral(mu[ix], s, local_a, nodes)
        remaining = np.ones(len(ix), bool)
        count = nodes * 2
        while count <= max_nodes and remaining.any():
            current = _integral(mu[ix[remaining]], s[remaining], local_a[remaining], count)
            converged = np.abs(current - previous[remaining]) <= tolerance
            old_remaining = np.flatnonzero(remaining)
            previous[old_remaining] = current
            remaining[old_remaining[converged]] = False
            count *= 2
        for j in np.flatnonzero(remaining):
            b = (np.log(local_a[j]) - np.log1p(-local_a[j]) - mu[ix[j]]) / s[j]
            lo = max(float(b), -10)
            if lo >= 10:
                previous[j] = 0
                continue
            value, error = quad(lambda v: max(expit(mu[ix[j]] + s[j] * v) - local_a[j], 0) * np.exp(-v * v / 2) / np.sqrt(2 * np.pi),
                                lo, 10, epsabs=tolerance / 10, epsrel=1e-9, limit=200)
            if error > tolerance:
                raise FloatingPointError("EAI integration failed convergence")
            previous[j] = value
        answer[ix] = previous
    if np.any(answer < -1e-12) or not np.isfinite(answer).all():
        raise FloatingPointError("invalid EAI")
    return np.maximum(answer, 0).reshape(shape)


def posterior_risk_mean(mean, latent_var):
    """E[sigmoid(Z)], separately from the sigmoid(mean) baseline point."""
    return expected_archive_improvement(mean, latent_var, 0.)


class RiskArchive:
    def __init__(self, bins=4, risk_threshold=.5, cell_count=None):
        if not 0 < risk_threshold < 1:
            raise ValueError("risk threshold must be in (0,1)")
        self.threshold = risk_threshold
        self.elite_quality = np.zeros(bins ** 4 if cell_count is None else cell_count)

    def observe(self, cell, risk):
        if not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("invalid measured risk")
        quality = max(risk - self.threshold, 0)
        self.elite_quality[cell] = max(self.elite_quality[cell], quality)

    def metrics(self):
        occupied = self.elite_quality > 0
        quality = self.elite_quality[occupied]
        return {"risk_occupied_cells": int(occupied.sum()),
                "risk_coverage_fraction_all_geometric_cells": float(occupied.mean()),
                "risk_qd_score": float(quality.sum()),
                "mean_elite_quality": float(quality.mean()) if len(quality) else 0.,
                "max_elite_quality": float(quality.max()) if len(quality) else 0.}
