"""Common baseline names and deterministic action samplers for fair budgets."""
from __future__ import annotations

from enum import Enum
import numpy as np


class Baseline(str, Enum):
    RANDOM = "random_sobol"
    CEM = "cem"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    INNER_RANDOM_CONFIG = "inner_sac_random_configuration"
    UNIVERSAL_NO_Z = "universal_no_z"
    OUTER_NO_Z = "outer_no_z"
    Z_OUTER_ONLY = "z_outer_only"
    Z_INNER_ONLY = "z_inner_only"
    LEGACY_PEARL = "legacy_pearl"
    FULL = "hierarchical_map_aware_meta_rl"


def sobol_like(seed: int, dimension: int, count: int) -> np.ndarray:
    """Low-discrepancy fallback that avoids another runtime dependency."""
    rng = np.random.default_rng(seed)
    offsets = rng.random(dimension)
    sequence = (np.arange(count)[:, None] * np.sqrt(np.arange(2, dimension + 2)) + offsets) % 1.0
    return sequence * 2.0 - 1.0
