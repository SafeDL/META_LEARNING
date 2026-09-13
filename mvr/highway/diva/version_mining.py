"""Reveal-only chronological replay for fixed-budget version regression testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.posterior import LatentPosterior, adapt_posterior
from mvr.highway.diva.regression_probability import critical_probability
from mvr.highway.diva.version_reference import RegressionLabels, VersionReference


@dataclass(frozen=True)
class RevealedOutcome:
    """The single target record exposed by one legal replay query."""

    anchor_index: int
    vulnerability: float
    collision: bool
    near_miss: bool
    valid: bool


class TargetReplay:
    """Stateful oracle that exposes each target anchor at most once."""

    def __init__(
        self,
        vulnerability: np.ndarray,
        collisions: np.ndarray,
        near_misses: np.ndarray,
        valid: np.ndarray,
        budget: int,
    ) -> None:
        self._vulnerability = np.asarray(vulnerability, dtype=float)
        self._collisions = np.asarray(collisions, dtype=bool)
        self._near_misses = np.asarray(near_misses, dtype=bool)
        self._valid = np.asarray(valid, dtype=bool)
        arrays = (self._collisions, self._near_misses, self._valid)
        if any(array.shape != self._vulnerability.shape for array in arrays):
            raise ValueError("target arrays must share an anchor shape")
        self._budget = budget
        self._queried: list[int] = []

    @property
    def queried_indices(self) -> tuple[int, ...]:
        return tuple(self._queried)

    def reveal(self, anchor_index: int) -> RevealedOutcome:
        """Expose one target outcome only after the caller chose its index."""
        if len(self._queried) >= self._budget:
            raise ValueError("target replay budget exhausted")
        if anchor_index in self._queried:
            raise ValueError("a target anchor may be revealed only once")
        if not 0 <= anchor_index < len(self._vulnerability):
            raise IndexError("target anchor index out of range")
        self._queried.append(anchor_index)
        return RevealedOutcome(
            anchor_index=anchor_index,
            vulnerability=float(self._vulnerability[anchor_index]),
            collision=bool(self._collisions[anchor_index]),
            near_miss=bool(self._near_misses[anchor_index]),
            valid=bool(self._valid[anchor_index]),
        )


def masked_probability_scores(
    prior: LowRankPrior,
    posterior: LatentPosterior,
    reference: VersionReference,
    selected: tuple[int, ...],
    threshold: float,
    observation_noise: float,
) -> np.ndarray:
    """Score prior-safe unqueried anchors; excluded anchors are negative infinity."""
    probabilities = critical_probability(
        prior, posterior, threshold, observation_noise
    )
    allowed = reference.previous_safe.copy()
    allowed[list(selected)] = False
    scores = np.full(probabilities.shape, -np.inf)
    scores[allowed] = probabilities[allowed]
    return scores


def deterministic_argmax(scores: np.ndarray) -> int:
    """Choose the smallest anchor index among equal finite maxima."""
    values = np.asarray(scores, dtype=float)
    candidates = np.flatnonzero(np.isfinite(values))
    if not len(candidates):
        raise ValueError("no eligible anchors remain")
    return int(candidates[np.argmax(values[candidates])])


def sequential_indices(
    prior: LowRankPrior,
    reference: VersionReference,
    replay: TargetReplay,
    budget: int,
    threshold: float,
    observation_noise: float,
    update: bool,
) -> tuple[list[int], list[RevealedOutcome], list[np.ndarray]]:
    """Select, reveal, and optionally update without pre-reading outcomes."""
    posterior = adapt_posterior(prior, np.array([], dtype=int), np.array([]))
    selected: list[int] = []
    outcomes: list[RevealedOutcome] = []
    score_trace: list[np.ndarray] = []
    for _ in range(budget):
        scores = masked_probability_scores(
            prior, posterior, reference, tuple(selected), threshold, observation_noise
        )
        chosen = deterministic_argmax(scores)
        outcome = replay.reveal(chosen)
        if not outcome.valid:
            raise ValueError("an eligible target outcome must be valid")
        selected.append(chosen)
        outcomes.append(outcome)
        score_trace.append(scores)
        if update:
            posterior = adapt_posterior(
                prior,
                np.asarray(selected, dtype=int),
                np.asarray([item.vulnerability for item in outcomes], dtype=float),
                observation_noise,
            )
    return selected, outcomes, score_trace


def trace_counts(indices: list[int], labels: RegressionLabels) -> dict[str, object]:
    """Score a completed trace using offline labels after selection is complete."""
    selected = np.asarray(indices, dtype=int)
    regression = labels.regression[selected]
    new = labels.new_in_archive[selected]
    reintroduced = labels.reintroduced[selected]
    return {
        "regression_count": int(regression.sum()),
        "new_in_archive_count": int(new.sum()),
        "reintroduced_count": int(reintroduced.sum()),
        "regression_curve": np.cumsum(regression, dtype=int),
    }
