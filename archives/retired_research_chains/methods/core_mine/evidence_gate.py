"""Developmental event-rate evidence gate for two historical selectors."""

from __future__ import annotations

import numpy as np
from scipy.special import betaln
from scipy.stats import rankdata

from methods.core_mine.simple_residual_ablation import corrected_scores


SUPPORT_PREFIX = 10


def log_mode_heterogeneity_bayes_factor(
    modes: np.ndarray, selected: list[int], events: list[bool]
) -> float:
    """Exact H1 independent-mode / H0 shared-rate log BF, Beta(1,1) priors."""
    if len(selected) != len(events):
        raise ValueError("selected/event ledgers differ")
    if not selected:
        return 0.0
    labels = np.asarray(modes, dtype=str)
    indices = np.asarray(selected, dtype=int)
    observed = np.asarray(events, dtype=int)
    if np.any(indices < 0) or np.any(indices >= len(labels)):
        raise IndexError("selected candidate is outside the scenario pool")
    if np.any((observed != 0) & (observed != 1)):
        raise ValueError("events must be binary")
    log_h1 = 0.0
    for mode in np.unique(labels):
        subset = observed[labels[indices] == mode]
        successes = int(subset.sum())
        log_h1 += float(betaln(1 + successes,
                               1 + len(subset) - successes))
    successes = int(observed.sum())
    log_h0 = float(betaln(1 + successes,
                           1 + len(observed) - successes))
    return log_h1 - log_h0


def static_mode_quantile_scores(
    task, eligible_indices: np.ndarray | None = None
) -> np.ndarray:
    source = task.source_y.mean(axis=0)
    eligible = (np.arange(task.count, dtype=int) if eligible_indices is None
                else np.asarray(eligible_indices, dtype=int))
    scores = np.full(task.count, -np.inf, dtype=float)
    for mode in np.unique(task.modes[eligible]):
        indices = eligible[task.modes[eligible] == mode]
        ranks = rankdata(source[indices], method="average")
        scores[indices] = (ranks - 1) / max(len(indices) - 1, 1)
    return scores


def gated_scores(task, selected: list[int], responses: list[float],
                 events: list[bool], eligible_indices: np.ndarray | None = None
                 ) -> tuple[np.ndarray, str, float]:
    """Use only target outcomes revealed by the current charged campaign."""
    if len(selected) != len(responses) or len(selected) != len(events):
        raise ValueError("target-feedback ledgers differ")
    log_bf = log_mode_heterogeneity_bayes_factor(
        task.modes, selected, events)
    if len(selected) < SUPPORT_PREFIX or log_bf <= 0.0:
        return static_mode_quantile_scores(task, eligible_indices), "static", log_bf
    return (corrected_scores(task, selected, responses,
                             "HistoryMargin-ModeShift"),
            "adaptive", log_bf)
