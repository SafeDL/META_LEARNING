"""Explicit failure-oracle helpers shared by replications."""

from __future__ import annotations

import numpy as np


def collision_failure(collisions: np.ndarray) -> np.ndarray:
    """Main DETOUR-Scenario-H failure oracle: an actual collision."""
    return np.asarray(collisions, dtype=bool)


def critical_event(collisions: np.ndarray, near_misses: np.ndarray) -> np.ndarray:
    """Offline reporting event, kept distinct from the collision failure oracle."""
    return np.asarray(collisions, dtype=bool) | np.asarray(near_misses, dtype=bool)
