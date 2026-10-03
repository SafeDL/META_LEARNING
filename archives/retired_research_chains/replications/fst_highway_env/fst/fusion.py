"""NumPy-side FST fusion and baseline estimators."""

from __future__ import annotations

import numpy as np


def validate_weights(weights: np.ndarray, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < -1e-12):
        raise ValueError("weights must be a finite non-negative vector")
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError(f"weights sum to {w.sum()}, not 1")
    return w


def estimate(response: np.ndarray, selected_indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Estimate each model's reference event rate with a frozen selected set."""
    values = np.asarray(response, dtype=float)
    indices = np.asarray(selected_indices, dtype=int)
    w = validate_weights(weights, len(indices))
    if values.ndim == 1:
        values = values[None, :]
    return values[:, indices] @ w


def inverse_distance_attention(
    features: np.ndarray,
    selected_indices: np.ndarray,
    epsilon: float = 1e-6,
    temperature: float = 1.0,
) -> np.ndarray:
    """Handcrafted FST-C-inspired control with the same query normalization."""
    x = np.asarray(features, dtype=float)
    selected = np.asarray(selected_indices, dtype=int)
    distances = np.linalg.norm(x[selected, None, :] - x[None, :, :], axis=2)
    logits = 1.0 / (distances + epsilon) / temperature
    logits -= logits.max(axis=0, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=0, keepdims=True)


def weights_from_attention(attention: np.ndarray, probability: np.ndarray) -> np.ndarray:
    s = np.asarray(attention, dtype=float)
    p = np.asarray(probability, dtype=float)
    if s.ndim != 2 or s.shape[1] != len(p):
        raise ValueError("attention must have shape [n,L]")
    if not np.allclose(s.sum(axis=0), 1.0, atol=1e-7):
        raise ValueError("attention columns must sum to one")
    return validate_weights(s @ p, s.shape[0])

