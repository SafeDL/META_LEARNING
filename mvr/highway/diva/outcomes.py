"""Shared formal safety-outcome semantics for Highway vulnerability methods."""

from __future__ import annotations

import numpy as np


def formal_rewards(collisions: np.ndarray, near_misses: np.ndarray) -> np.ndarray:
    """Map outcomes to collision=1, near-miss=0.5, otherwise=0 rewards."""
    return np.where(collisions, 1.0, np.where(near_misses, 0.5, 0.0))
