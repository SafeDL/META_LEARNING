"""Archive-only means and input-support covariance, frozen before selection."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist

from methods.baselines.hem.experts import HistoricalExperts
from methods.baselines.hgt.config import group_of

from .config import JITTER, NOISE


def physical_kernel(x, y=None):
    y = x if y is None else y
    distance = np.sqrt(5.0) * cdist(x[:, :4], y[:, :4]) / 0.2
    return (
        0.25
        * (1 + distance + distance**2 / 3)
        * np.exp(-distance)
        * (x[:, 4, None] == y[None, :, 4])
    )


def archive_covariance(x, source_x, physical=None):
    physical = physical_kernel(x) if physical is None else physical
    result = physical.copy()
    for family in (0, 1):
        rows = np.flatnonzero(x[:, 4] == family)
        context = source_x[source_x[:, 4] == family]
        if not len(context) or not len(rows):
            continue
        factor = cholesky(
            physical_kernel(context) + (NOISE + JITTER) * np.eye(len(context)),
            lower=True,
        )
        solved = solve_triangular(factor, physical_kernel(context, x[rows]), lower=True)
        result[np.ix_(rows, rows)] -= solved.T @ solved
    return 0.5 * (result + result.T)


def visible_history(history, condition, seed):
    output = {}
    for name in sorted(history):
        if condition.startswith("without_") and group_of(name) == condition[8:]:
            continue
        source = history[name]
        keep = np.arange(len(source["x"]))
        if condition == "sparse":
            rng = np.random.default_rng(442001 + seed + sum(map(ord, name)))
            keep = np.concatenate(
                [
                    rng.choice(rows, max(8, round(len(rows) / 4)), replace=False)
                    for family in np.unique(source["x"][:, 4])
                    for rows in [np.flatnonzero(source["x"][:, 4] == family)]
                ]
            )
        elif condition in ("cutin_gap", "brake_gap"):
            family = int(condition == "brake_gap")
            keep = keep[~((source["x"][:, 4] == family) & (source["x"][:, 0] < 0.5))]
        output[name] = {"x": source["x"][keep], "risk": source["risk"][keep]}
    return output


@dataclass
class TransferPrior:
    x: np.ndarray
    means: np.ndarray
    weights: np.ndarray
    names: tuple
    enhanced_covariance: np.ndarray
    physical_covariance: np.ndarray
    archive_covariances: tuple


def build_prior(x, history, enhanced_covariance):
    history = {name: history[name] for name in sorted(history)}
    prediction = HistoricalExperts(history).predict(x)
    physical = physical_kernel(x)
    cache = {}
    covariances = []
    for source in history.values():
        # Sources with the same archive locations share one factorization.
        key = source["x"].tobytes()
        if key not in cache:
            cache[key] = archive_covariance(x, source["x"], physical)
        covariances.append(cache[key])
    return TransferPrior(
        x,
        prediction.mean,
        prediction.group_weights,
        prediction.names,
        enhanced_covariance,
        physical,
        tuple(covariances),
    )
