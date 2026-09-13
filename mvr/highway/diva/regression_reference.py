"""Source-only population statistics for regression-vulnerability testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionReference:
    """Historical population mean and commonness of failures per anchor."""

    mean_vulnerability: np.ndarray
    failure_prevalence: np.ndarray
    smoothed_failure_prevalence: np.ndarray


def build_regression_reference(
    source_vulnerability: np.ndarray,
    source_collisions: np.ndarray,
    source_near_misses: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> RegressionReference:
    """Build a beta-smoothed reference from LOSO source SUTs only."""
    vulnerability = np.asarray(source_vulnerability, dtype=float)
    collisions = np.asarray(source_collisions, dtype=bool)
    near_misses = np.asarray(source_near_misses, dtype=bool)
    if vulnerability.ndim != 2 or vulnerability.shape[0] == 0:
        raise ValueError("source_vulnerability must have nonempty SUT and anchor axes")
    if (
        collisions.shape != vulnerability.shape
        or near_misses.shape != vulnerability.shape
    ):
        raise ValueError("source outcome arrays must match source_vulnerability")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Beta smoothing parameters must be positive")
    failures = collisions | near_misses
    counts = failures.sum(axis=0, dtype=float)
    source_count = vulnerability.shape[0]
    return RegressionReference(
        mean_vulnerability=vulnerability.mean(axis=0),
        failure_prevalence=counts / source_count,
        smoothed_failure_prevalence=(counts + alpha) / (source_count + alpha + beta),
    )
