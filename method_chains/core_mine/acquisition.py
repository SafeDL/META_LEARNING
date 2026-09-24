"""Risk, severity, balanced, and marginal-coverage selection rules."""

from __future__ import annotations

import numpy as np

from .config import B_REF, COVERAGE_LENGTH


def rbf_by_mode(features: np.ndarray, modes: np.ndarray, left: np.ndarray, right: np.ndarray, length: float = COVERAGE_LENGTH) -> np.ndarray:
    result = np.zeros((len(left), len(right)), dtype=float)
    labels = np.asarray(modes, dtype=str)
    for mode in np.unique(labels):
        li, ri = np.flatnonzero(labels[left] == mode), np.flatnonzero(labels[right] == mode)
        if len(li) and len(ri):
            diff = features[left[li], None, :] - features[right[ri]][None, :, :]
            result[np.ix_(li, ri)] = np.exp(-np.sum(diff * diff, axis=2) / (2.0 * length * length))
    return result


def marginal_scores(features: np.ndarray, modes: np.ndarray, selected: list[int], severity: list[float], p_event: np.ndarray, p_collision: np.ndarray, lambda_: float, evaluation_indices: np.ndarray | None = None) -> np.ndarray:
    candidates = np.arange(len(modes), dtype=int)
    references = candidates if evaluation_indices is None else np.asarray(evaluation_indices, dtype=int)
    unique, counts = np.unique(np.asarray(modes)[references], return_counts=True)
    weights = np.zeros(len(references), dtype=float)
    for mode, count in zip(unique, counts, strict=True):
        weights[np.asarray(modes)[references] == mode] = 1.0 / (len(unique) * count)
    if selected:
        selected_array = np.asarray(selected, dtype=int)
        prior = np.max(rbf_by_mode(features, modes, references, selected_array) * np.asarray(severity)[None, :], axis=1)
    else:
        prior = np.zeros(len(references), dtype=float)
    kernel = rbf_by_mode(features, modes, references, candidates)
    delta_one = np.sum(weights[:, None] * (np.maximum(prior[:, None], kernel) - prior[:, None]), axis=0)
    delta_half = np.sum(weights[:, None] * (np.maximum(prior[:, None], 0.5 * kernel) - prior[:, None]), axis=0)
    return p_collision * delta_one + (p_event - p_collision) * delta_half + lambda_ / B_REF * (p_event + p_collision) / 2.0


def verified_novelty_scores(features: np.ndarray, modes: np.ndarray, selected: list[int],
                            severity: list[float], p_event: np.ndarray, penalty: float) -> np.ndarray:
    """Prefer unexplored risk only after a verified nearby event exists."""
    if not 0.0 <= penalty <= 1.0:
        raise ValueError("novelty penalty must be in [0,1]")
    if not selected:
        return p_event.copy()
    positive = np.flatnonzero(np.asarray(severity) > 0.0)
    if not len(positive):
        return p_event.copy()
    candidate_indices = np.arange(len(modes), dtype=int)
    positives = np.asarray(selected, dtype=int)[positive]
    kernel = rbf_by_mode(features, modes, candidate_indices, positives)
    redundancy = np.max(kernel * np.asarray(severity, dtype=float)[positive][None, :], axis=1)
    return p_event * (1.0 - penalty * redundancy)


def choose(scores: np.ndarray, selected: list[int], modes: np.ndarray, support_budget: int,
           balanced: bool = False, allowed_indices: np.ndarray | None = None) -> int:
    available = np.ones(len(scores), dtype=bool)
    available[np.asarray(selected, dtype=int)] = False if selected else True
    if allowed_indices is not None:
        available &= np.isin(np.arange(len(scores)), np.asarray(allowed_indices, dtype=int))
    if balanced:
        labels = np.asarray(modes, dtype=str)
        counts = {mode: int(np.sum(labels[np.asarray(selected, dtype=int)] == mode)) if selected else 0 for mode in np.unique(labels)}
        least = min(counts.values())
        available &= np.isin(labels, [mode for mode, count in counts.items() if count == least])
    elif len(selected) < support_budget:
        labels = np.asarray(modes, dtype=str)
        covered = set(labels[np.asarray(selected, dtype=int)]) if selected else set()
        missing = [mode for mode in np.unique(labels) if mode not in covered]
        if missing and support_budget - len(selected) <= len(missing):
            available &= np.isin(labels, missing)
    candidates = np.flatnonzero(available)
    if not len(candidates):
        raise RuntimeError("no available candidate")
    return int(candidates[np.lexsort((candidates, -scores[candidates]))[0]])
