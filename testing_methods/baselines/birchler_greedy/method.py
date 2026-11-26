"""Birchler et al. (2022), §3.4, with the common prefix and unit query costs."""

import numpy as np
from scipy.spatial.distance import cdist

from testing_methods.baselines.detour.method import parameter_features


def prioritize(candidates, initial_indices):
    x = np.stack([parameter_features(scene) for scene in candidates])
    scale = x.std(axis=0)
    x = (x - x.mean(axis=0)) / np.where(scale > 0, scale, 1)
    distance = cdist(x, x)
    remaining = np.ones(len(x), dtype=bool)
    remaining[initial_indices] = False
    total = distance[initial_indices].sum(axis=0)
    rng = np.random.default_rng(0)
    order = []
    for _ in range(int(remaining.sum())):
        best = np.max(total[remaining])
        # Every candidate has the same cost and the same averaging denominator.
        ties = np.flatnonzero(remaining & np.isclose(total, best, rtol=1e-13, atol=1e-13))
        index = int(rng.choice(ties)) if len(ties) > 1 else int(ties[0])
        order.append(index)
        remaining[index] = False
        total += distance[index]
    return order
