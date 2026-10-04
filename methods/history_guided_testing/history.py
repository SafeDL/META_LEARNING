"""Measured historical risk, with one spatial training/validation split."""
import numpy as np
from scipy.spatial.distance import cdist

from .config import GROUPS, PROFILES, ROOT, group_of


def load_history():
    bank = np.load(ROOT / "history/responses.npz")
    return {profile.name: {"x": bank["x"], "risk": bank[profile.name]}
            for profile in PROFILES}


def split_indices(x):
    train, validation = [], []
    for family in (0, 1):
        indices = np.flatnonzero(x[:, 4] == family)
        order = np.random.default_rng(75017 + family).permutation(indices)
        train.extend(order[:819])
        validation.extend(order[819:])
    return np.asarray(train), np.asarray(validation)


def subset(history, indices, excluded_group=None):
    return {name: {"x": source["x"][indices], "risk": source["risk"][indices]}
            for name, source in history.items() if group_of(name) != excluded_group}


def historical_risk(history, x):
    groups = []
    for names in GROUPS.values():
        estimates = []
        for name in names:
            if name not in history:
                continue
            source = history[name]
            distances = cdist(x, source["x"])
            nearest = np.argsort(distances, axis=1, kind="stable")[:, :8]
            weights = np.exp(-.5 * (np.take_along_axis(distances, nearest, axis=1) / .15) ** 2)
            weights /= weights.sum(axis=1, keepdims=True)
            estimates.append((weights * source["risk"][nearest]).sum(axis=1))
        if estimates:
            groups.append(np.mean(estimates, axis=0))
    return np.mean(groups, axis=0)
