"""Exact paired sign-flip tests and profile-level co-primary inference."""
import numpy as np


def signed_sums(values):
    result = np.zeros(1)
    for value in values:
        result = np.concatenate((result - value, result + value))
    return result


def exact_sign_flip(differences):
    probabilities = []
    tolerance = len(differences) * 1e-12
    for column in differences.T:
        observed = abs(column.sum())
        values = column[column != 0]
        if observed <= tolerance:
            probabilities.append(1.)
            continue
        split = len(values) // 2
        left = signed_sums(values[:split])
        right = np.sort(signed_sums(values[split:]))
        positive = len(right) - np.searchsorted(
            right, observed - tolerance - left, side="left")
        negative = np.searchsorted(right,
                                   -observed + tolerance - left,
                                   side="right")
        extreme = positive.sum(dtype=np.int64) + negative.sum(dtype=np.int64)
        probabilities.append(float(extreme / 2**len(values)))
    return np.asarray(probabilities)


def holm_adjust(probabilities):
    order = np.argsort(probabilities)
    adjusted = np.empty_like(probabilities)
    adjusted[order] = np.minimum(
        1.,
        np.maximum.accumulate(probabilities[order] *
                              np.arange(len(probabilities), 0, -1)))
    return adjusted


def bootstrap_indices(profiles):
    rng = np.random.default_rng(20261019)
    controllers = np.array([p["controller"] for p in profiles])
    return np.concatenate([
        rng.choice(np.flatnonzero(controllers == controller),
                   size=(20000, int((controllers == controller).sum())),
                   replace=True) for controller in ("IDM", "FVDM")
    ],
                          axis=1)
