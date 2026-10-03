"""Target-hidden fixed-pool selectors for the AdaTE response-mixture stage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mixture import QPDiagnostics, simplex_least_squares, uniform_alpha


@dataclass
class MixtureSelector:
    """Reveal-only selector: target values enter only via ``observe``."""
    source_responses: np.ndarray
    budget: int
    variant: str = "sequential"
    static_k: int = 0
    ridge: float = 0.0
    alpha: np.ndarray = field(init=False)
    selected: list[int] = field(default_factory=list, init=False)
    revealed: list[float] = field(default_factory=list, init=False)
    diagnostics: list[QPDiagnostics] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.source_responses = np.asarray(self.source_responses, dtype=float)
        if self.source_responses.ndim != 2:
            raise ValueError("source_responses must have shape [models, candidates]")
        if not 0 < self.budget <= self.source_responses.shape[1]:
            raise ValueError("budget must be within candidate count")
        if self.variant not in {"uniform", "static", "sequential"}:
            raise ValueError("unknown selector variant")
        if self.variant == "static" and not 0 < self.static_k <= self.budget:
            raise ValueError("static_k must be within budget")
        self.alpha = uniform_alpha(self.source_responses.shape[0])

    def propose(self) -> int | None:
        if len(self.selected) >= self.budget:
            return None
        scores = self.alpha @ self.source_responses
        if self.selected:
            scores[np.asarray(self.selected, dtype=int)] = -np.inf
        return int(np.argmax(scores))

    def observe(self, index: int, target_response: float) -> QPDiagnostics | None:
        if index in self.selected or not 0 <= index < self.source_responses.shape[1]:
            raise ValueError("invalid or repeated target reveal")
        self.selected.append(index)
        self.revealed.append(float(target_response))
        should_fit = self.variant == "sequential" or (self.variant == "static"
                                                      and len(self.selected) == self.static_k)
        if not should_fit:
            return None
        diagnostic = simplex_least_squares(self.source_responses[:, self.selected].T,
                                           np.asarray(self.revealed), self.alpha, self.ridge)
        self.alpha = diagnostic.alpha
        self.diagnostics.append(diagnostic)
        return diagnostic
