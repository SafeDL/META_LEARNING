"""Label-independent common initial queries for matched testing tasks."""

import numpy as np


def initial_indices(size, count, seed, *, pool_size=None):
    pool_size = size if pool_size is None else pool_size
    if not 0 <= count <= pool_size <= size:
        raise ValueError("Initialization must fit within the shared candidate prefix")
    return np.random.default_rng(seed).permutation(pool_size)[:count].tolist()
