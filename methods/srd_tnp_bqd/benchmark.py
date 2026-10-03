"""Shared query budget, seeds and parameter-cell definitions."""
import numpy as np

from .common import DEFAULT_CONFIG, config_at


CONFIG = config_at(DEFAULT_CONFIG)
BUDGET = CONFIG["benchmark"]["budget"]
CHECKPOINTS = tuple(CONFIG["benchmark"]["checkpoints"])
GRID_BINS = CONFIG["benchmark"]["grid_bins"]
SEEDS = tuple(CONFIG["benchmark"]["seeds"])


def parameter_cells(x, bins=GRID_BINS):
    x = np.asarray(x)
    digits = np.minimum((x * bins).astype(int), bins - 1)
    return np.ravel_multi_index(digits.T, (bins,) * x.shape[1])
