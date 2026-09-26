"""Sequential failure search using only revealed target responses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FunctionPosterior:
    indices: np.ndarray
    responses: np.ndarray
    events: np.ndarray
    log_weights: np.ndarray

    @property
    def weights(self) -> np.ndarray:
        weights = np.exp(self.log_weights - self.log_weights.max())
        return weights / weights.sum()


class PosteriorSearch:
    """A selector without access to an oracle or complete target vector.

    Each function has a discrete source-response hypothesis. A Gaussian
    discrepancy model updates its probability after a target execution.
    Queries greedily maximize posterior event probability and use predicted
    severity and scenario index as deterministic tie breakers.
    """

    def __init__(
        self,
        source_responses: np.ndarray,
        source_events: np.ndarray,
        modes: np.ndarray,
        noise_scale: float,
    ) -> None:
        self.noise_scale = noise_scale
        self.selected: list[int] = []
        self.candidate_count = source_responses.shape[1]
        labels = np.asarray(modes)
        self.functions: list[FunctionPosterior] = []
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            responses = np.asarray(source_responses[:, indices], dtype=float)
            events = np.asarray(source_events[:, indices], dtype=float)
            signatures = np.concatenate((responses, events), axis=1)
            _, unique = np.unique(signatures, axis=0, return_index=True)
            responses, events = responses[unique], events[unique]
            self.functions.append(FunctionPosterior(
                indices, responses, events, np.zeros(len(responses)),
            ))

    def observe(self, index: int, response: float) -> None:
        if index in self.selected:
            raise ValueError("a target execution cannot be counted twice")
        self.selected.append(index)
        for function in self.functions:
            local = np.flatnonzero(function.indices == index)
            if len(local):
                error = (function.responses[:, local[0]] - response) / self.noise_scale
                function.log_weights -= 0.5 * error**2
                break

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        probability = np.empty(self.candidate_count)
        severity = np.empty(self.candidate_count)
        for function in self.functions:
            weights = function.weights
            probability[function.indices] = weights @ function.events
            severity[function.indices] = weights @ function.responses
        return probability, severity

    def propose(
        self, allowed_indices: np.ndarray | None = None,
    ) -> tuple[int, dict[str, float]]:
        probability, severity = self.predict()
        available = np.ones(self.candidate_count, dtype=bool)
        available[self.selected] = False
        if not available.any():
            raise ValueError("the candidate pool is exhausted")
        allowed = available.copy()
        if allowed_indices is not None:
            allowed &= np.isin(np.arange(self.candidate_count), allowed_indices)
        candidates = np.flatnonzero(allowed)
        chosen = int(candidates[np.lexsort((
            candidates,
            -severity[candidates],
            -probability[candidates],
        ))[0]])
        return chosen, {
            "event_probability": float(probability[chosen]),
            "predicted_severity": float(severity[chosen]),
        }


ADAPTIVE_METHOD = "Posterior Search (adaptive support)"
BUDGETS = (10, 20, 30, 50)
NOISE_SCALES = (0.05, 0.15, 0.30)
SUPPORT_BUDGET = 10
TOTAL_BUDGET = 50


def run_adaptive_campaign(
    source_responses: np.ndarray,
    source_events: np.ndarray,
    modes: np.ndarray,
    target_responses: object,
    noise_scale: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Run the confirmed target-hidden search with all queries on budget."""
    selector = PosteriorSearch(
        source_responses,
        source_events,
        modes,
        noise_scale,
    )
    audit = []
    while len(selector.selected) < TOTAL_BUDGET:
        allowed = None
        if len(selector.selected) < SUPPORT_BUDGET:
            covered = np.asarray(modes)[selector.selected]
            missing = np.setdiff1d(np.unique(modes), covered)
            remaining_support = SUPPORT_BUDGET - len(selector.selected)
            if remaining_support <= len(missing):
                allowed = np.flatnonzero(np.isin(modes, missing))
        chosen, scores = selector.propose(allowed)
        audit.append({
            "step": len(selector.selected) + 1,
            "index": chosen,
            **scores,
        })
        selector.observe(chosen, float(target_responses[chosen]))
    return np.asarray(selector.selected), audit
