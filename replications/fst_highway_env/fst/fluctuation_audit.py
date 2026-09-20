"""Algebraic audit of the signed fluctuation estimator in paper Eq. (24)."""

from __future__ import annotations

import numpy as np


def signed_fluctuation(
    response: np.ndarray,
    selected_indices: np.ndarray,
    attention: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return paper Eq. (24), representative weights, and identity residual."""
    f = np.asarray(response, dtype=float)
    selected = np.asarray(selected_indices, dtype=int)
    s = np.asarray(attention, dtype=float)
    p = np.asarray(probability, dtype=float)
    w = s @ p
    numerator = ((f[None, :] - f[selected, None]) * s * p[None, :]).sum(axis=1)
    fluctuation = np.divide(numerator, w, out=np.zeros_like(numerator), where=w > 0)
    residual = float(w @ fluctuation - (p @ f - w @ f[selected]))
    return fluctuation, w, residual
