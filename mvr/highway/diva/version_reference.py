"""True version-regression labels and eligible previous-safe candidate sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VersionReference:
    """Information visible before querying a new controller version."""

    previous_safe: np.ndarray
    previous_vulnerability: np.ndarray
    historical_critical: np.ndarray


@dataclass(frozen=True)
class RegressionLabels:
    """Offline truth labels; never hand this object to a selector."""

    regression: np.ndarray
    new_in_archive: np.ndarray
    reintroduced: np.ndarray


def critical_events(collisions: np.ndarray, near_misses: np.ndarray) -> np.ndarray:
    """Return the protocol's binary collision-or-near-miss event."""
    return np.asarray(collisions, dtype=bool) | np.asarray(near_misses, dtype=bool)


def build_version_reference(
    historical_collisions: np.ndarray,
    historical_near_misses: np.ndarray,
    previous_vulnerability: np.ndarray,
    previous_collisions: np.ndarray,
    previous_near_misses: np.ndarray,
    previous_valid: np.ndarray,
) -> VersionReference:
    """Build only historical and direct-predecessor information."""
    previous_critical = critical_events(previous_collisions, previous_near_misses)
    historical = critical_events(historical_collisions, historical_near_misses)
    if historical.ndim != 2 or historical.shape[1:] != previous_critical.shape:
        raise ValueError("history and predecessor must share anchor dimensions")
    if np.asarray(previous_valid).shape != previous_critical.shape:
        raise ValueError("previous_valid must contain one value per anchor")
    return VersionReference(
        previous_safe=np.asarray(previous_valid, dtype=bool) & ~previous_critical,
        previous_vulnerability=np.asarray(previous_vulnerability, dtype=float),
        historical_critical=np.any(historical, axis=0),
    )


def classify_regressions(
    reference: VersionReference,
    target_collisions: np.ndarray,
    target_near_misses: np.ndarray,
    target_valid: np.ndarray,
) -> RegressionLabels:
    """Classify safe-to-critical transitions using valid offline labels."""
    target_critical = critical_events(target_collisions, target_near_misses)
    valid = np.asarray(target_valid, dtype=bool)
    if target_critical.shape != reference.previous_safe.shape or valid.shape != target_critical.shape:
        raise ValueError("target labels must match the reference anchor count")
    regression = reference.previous_safe & valid & target_critical
    new_in_archive = regression & ~reference.historical_critical
    reintroduced = regression & reference.historical_critical
    return RegressionLabels(regression, new_in_archive, reintroduced)
