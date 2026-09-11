"""Pure DIVA diagnostic and failure-mining acquisition functions."""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

from .posterior import LatentVulnerabilityPosterior


def score_level_set_weight(
    mean: np.ndarray, variance: np.ndarray, threshold: float
) -> np.ndarray:
    std = np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64), 1e-12))
    z = (np.asarray(mean, dtype=np.float64) - threshold) / std
    probability = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
    return 4.0 * probability * (1.0 - probability)


def diagnostic_scores(
    posterior: LatentVulnerabilityPosterior,
    basis: np.ndarray,
    mean: np.ndarray,
    observation_noise_var: np.ndarray,
    evaluability: np.ndarray,
    *,
    level_set_threshold: float,
    eta: float = 0.20,
    evaluability_power: float = 2.0,
) -> np.ndarray:
    matrix = np.asarray(basis, dtype=np.float64)
    noise = np.asarray(observation_noise_var, dtype=np.float64)
    predicted_variance = np.einsum("ij,jk,ik->i", matrix, posterior.covariance, matrix) + noise
    predicted_mean = np.asarray(mean, dtype=np.float64) + matrix @ posterior.mean
    information = np.asarray([
        posterior.information_gain(row, value) for row, value in zip(matrix, noise)
    ])
    boundary = score_level_set_weight(
        predicted_mean, predicted_variance, level_set_threshold
    )
    return information * (eta + (1.0 - eta) * boundary) * np.power(evaluability, evaluability_power)


def mining_scores(
    mean: np.ndarray,
    variance: np.ndarray,
    evaluability: np.ndarray,
    novelty: np.ndarray,
    *,
    beta: float = 0.50,
    evaluability_power: float = 2.0,
    novelty_floor: float = 0.25,
) -> np.ndarray:
    risk_ucb = np.clip(np.asarray(mean) + beta * np.sqrt(np.maximum(variance, 0.0)), 0.0, 1.0)
    novelty_weight = novelty_floor + (1.0 - novelty_floor) * np.asarray(novelty)
    return np.power(np.asarray(evaluability), evaluability_power) * risk_ucb * novelty_weight


def novelty_weight(features: np.ndarray, archive: np.ndarray, bandwidth: float = 0.50) -> np.ndarray:
    points = np.asarray(features, dtype=np.float64)
    if len(archive) == 0:
        return np.ones(len(points), dtype=np.float64)
    distances2 = np.square(points[:, None, :] - np.asarray(archive)[None, :, :]).sum(axis=-1)
    minimum = distances2.min(axis=1)
    return 1.0 - np.exp(-minimum / (2.0 * bandwidth ** 2))
