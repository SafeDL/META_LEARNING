"""Common baseline names and deterministic action samplers for fair budgets."""
from __future__ import annotations

import numpy as np


def low_discrepancy_samples(seed: int, dimension: int, count: int) -> np.ndarray:
    """Low-discrepancy fallback that avoids another runtime dependency."""
    rng = np.random.default_rng(seed)
    offsets = rng.random(dimension)
    sequence = (np.arange(count)[:, None] * np.sqrt(np.arange(2, dimension + 2)) + offsets) % 1.0
    return sequence * 2.0 - 1.0
