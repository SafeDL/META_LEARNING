"""Label-independent common initial queries for matched testing tasks."""

import numpy as np


def initial_indices(size, count, seed):
    return np.random.default_rng(seed).permutation(size)[:count].tolist()


def validate_initial(indices, size, count):
    if indices is not None and (
        len(indices) != count
        or len(set(indices)) != count
        or any(int(i) != i or not 0 <= i < size for i in indices)
    ):
        raise ValueError("Initialization must contain count unique pool indices")
