"""Population-only vulnerability and failure-prevalence reference data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PopulationReference:
    """Reference statistics fitted exclusively on LOSO source SUTs."""

    mean_vulnerability: np.ndarray
    failure_prevalence: np.ndarray
    smoothed_failure_prevalence: np.ndarray


def build_population_reference(
    source_vulnerability: np.ndarray,
    source_collisions: np.ndarray,
    source_near_misses: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> PopulationReference:
    """Build a Beta-smoothed failure reference from source rows only."""
    vulnerability = np.asarray(source_vulnerability, dtype=float)
    collisions = np.asarray(source_collisions, dtype=bool)
    near_misses = np.asarray(source_near_misses, dtype=bool)
    if vulnerability.ndim != 2 or vulnerability.shape[0] == 0:
        raise ValueError("source_vulnerability must contain source SUT rows")
    if (
        collisions.shape != vulnerability.shape
        or near_misses.shape != vulnerability.shape
    ):
        raise ValueError("source outcome arrays must match source_vulnerability")
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta smoothing parameters must be positive")

    failures = collisions | near_misses
    n_source = failures.shape[0]
    failure_count = failures.sum(axis=0)
    prevalence = failure_count / n_source
    smoothed_prevalence = (failure_count + alpha) / (n_source + alpha + beta)
    return PopulationReference(
        mean_vulnerability=vulnerability.mean(axis=0),
        failure_prevalence=prevalence,
        smoothed_failure_prevalence=smoothed_prevalence,
    )
