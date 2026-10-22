"""Label-independent common initial queries for matched testing tasks."""

import numpy as np


def initial_indices(size, count, seed):
    return np.random.default_rng(seed).permutation(size)[:count].tolist()
