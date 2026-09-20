"""Simplex-constrained response-mixture fitting used by AdaTE A0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class QPDiagnostics:
    """Auditable result of one nonnegative, sum-to-one least-squares fit."""

    alpha: np.ndarray
    status: str
    residual_norm: float
    iterations: int
    simplex_violation: float
    fallback: bool = False


def uniform_alpha(n_models: int) -> np.ndarray:
    if n_models < 1:
        raise ValueError("n_models must be positive")
    return np.full(n_models, 1.0 / n_models, dtype=float)


def simplex_least_squares(design: np.ndarray,
                          target: np.ndarray,
                          previous: np.ndarray | None = None,
                          ridge: float = 0.0) -> QPDiagnostics:
    """Fit alpha on the simplex without inspecting unrevealed target responses."""
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    if design.ndim != 2 or target.shape != (design.shape[0], ):
        raise ValueError("design must be [observations, models] and target [observations]")
    n_models = design.shape[1]
    if n_models < 1 or ridge < 0:
        raise ValueError("invalid model count or ridge")
    fallback = uniform_alpha(n_models) if previous is None else np.asarray(previous, dtype=float)
    if fallback.shape != (n_models, ) or np.any(
            fallback < -1e-8) or not np.isclose(fallback.sum(), 1.0):
        raise ValueError("previous alpha must lie on the simplex")
    if not len(target):
        return QPDiagnostics(fallback.copy(), "no_observations", 0.0, 0, 0.0)
    centre = uniform_alpha(n_models)

    def objective(alpha: np.ndarray) -> float:
        residual = design @ alpha - target
        return float(0.5 * residual @ residual + 0.5 * ridge * np.sum((alpha - centre)**2))

    result = minimize(objective,
                      fallback,
                      method="SLSQP",
                      bounds=[(0.0, 1.0)] * n_models,
                      constraints={
                          "type": "eq",
                          "fun": lambda alpha: float(alpha.sum() - 1.0)
                      },
                      options={
                          "ftol": 1e-11,
                          "maxiter": 500
                      })
    candidate = np.asarray(result.x, dtype=float) if result.success else fallback.copy()
    violation = max(float(abs(candidate.sum() - 1.0)), float(np.max(np.maximum(-candidate, 0.0))))
    valid = bool(result.success and np.all(np.isfinite(candidate)) and violation <= 1e-7)
    alpha = candidate if valid else fallback.copy()
    residual = float(np.linalg.norm(design @ alpha - target))
    return QPDiagnostics(alpha,
                         str(result.message) if valid else "fallback_after_solver_failure",
                         residual, int(getattr(result, "nit",
                                               0)), violation if valid else 0.0, not valid)
