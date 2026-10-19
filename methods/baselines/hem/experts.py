"""Separate, scene-isolated kNN sources and source-only reliability estimates."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from methods.baselines.hgt.config import group_of

from .config import NEIGHBORS, NEIGHBOR_LENGTH


def neighbor_prediction(source, x, *, exclude_exact=False):
    means = np.empty(len(x))
    for family in np.unique(x[:, 4]):
        rows = np.flatnonzero(x[:, 4] == family)
        columns = np.flatnonzero(source["x"][:, 4] == family)
        distance = cdist(x[rows, :4], source["x"][columns, :4])
        if exclude_exact:
            distance[distance == 0] = np.inf
        indices = np.argsort(distance, axis=1, kind="stable")[:, :NEIGHBORS]
        local = np.take_along_axis(distance, indices, axis=1)
        weights = np.exp(-.5 * (local / NEIGHBOR_LENGTH) ** 2)
        weights /= weights.sum(axis=1, keepdims=True)
        means[rows] = (weights * source["risk"][columns[indices]]).sum(axis=1)
    return means


@dataclass
class ExpertPredictions:
    names: tuple
    groups: tuple
    mean: np.ndarray

    @property
    def group_weights(self):
        groups = np.asarray(self.groups)
        return np.asarray([1 / (len(set(groups)) * np.count_nonzero(groups == g)) for g in groups])

    def fixed_mean(self):
        grouped = [np.mean(self.mean[:, np.asarray(self.groups) == group].T, axis=0)
                   for group in dict.fromkeys(self.groups)]
        return np.mean(grouped, axis=0)



class HistoricalExperts:
    def __init__(self, history):
        self.history = history

    def predict(self, x, *, excluded_groups=(), exclude_exact=False):
        names = tuple(name for name in self.history if group_of(name) not in excluded_groups)
        means = []
        for name in names:
            mean = neighbor_prediction(self.history[name], x, exclude_exact=exclude_exact)
            means.append(mean)
        return ExpertPredictions(names, tuple(group_of(n) for n in names),
                                 np.asarray(means).T)

