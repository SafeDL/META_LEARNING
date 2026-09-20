"""Counterfactual utility teacher over the frozen, source-only anchor bank."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..diva.factorization import fit_low_rank_vulnerability
from ..diva.source_bank import SourceBank
from .state import TeacherState, TeacherWorld

CONTINUATION_POLICIES = (
    "mine_now",
    "diagnose1_then_mine",
    "diagnose2_then_mine",
)


@dataclass(frozen=True)
class TeacherActionValue:
    candidate_index: int
    q_formal: float
    q_vulnerability: float
    best_continuation: str
    immediate_formal_score: float
    immediate_vulnerability_response: float


@dataclass(frozen=True)
class TeacherEvaluation:
    """One fixed-budget replay result, with no simulator interaction."""

    policy: str
    source_index: int
    formal_score: float
    cumulative_vulnerability: float
    formal_curve: tuple[float, ...]
    vulnerability_curve: tuple[float, ...]
    selected_indexes: tuple[int, ...]


class CounterfactualTeacher:
    """Analytic posterior plus a three-policy continuation portfolio."""
    def __init__(self, world: TeacherWorld) -> None:
        self.world = world

    @classmethod
    def from_source_bank(
        cls,
        bank: SourceBank,
        training_source_indexes: Iterable[int],
        *,
        rank: int = 2,
        device: str = "cpu",
        gp_fit_steps: int = 100,
        proxy_event_threshold: float = 0.75,
    ) -> "CounterfactualTeacher":
        indexes = tuple(int(index) for index in training_source_indexes)
        if len(indexes) < 2 or len(set(indexes)) != len(indexes):
            raise ValueError("teacher prior requires at least two distinct training sources")
        if any(index < 0 or index >= len(bank.source_refs) for index in indexes):
            raise IndexError("teacher training source is outside the source bank")
        if not 0 <= rank < len(indexes):
            raise ValueError("teacher rank must be smaller than its training-source count")
        factorization = fit_low_rank_vulnerability(
            bank.responses[np.asarray(indexes)],
            bank.eligible[np.asarray(indexes)],
            tuple(bank.source_refs[index] for index in indexes),
            bank.design_ids,
            rank,
        )
        if not factorization.common_eligible_mask.all():
            raise ValueError("Stage 1 teacher requires common eligible source anchors")
        # The teacher is evaluated only on the 64 observed common anchors.  At
        # those anchors the fitted low-rank prior is exact, so invoking the GP
        # interpolator would add numerical approximation and hours of needless
        # work without changing the teacher's information set.  The GP remains
        # the prior-task generator for off-anchor synthetic pretraining.
        mean = factorization.mean
        bases = factorization.basis
        noise = np.full(len(bank.designs),
                        0.02 + factorization.residual_variance,
                        dtype=np.float64)
        world = TeacherWorld(
            pool=bank.designs,
            base_mean=np.asarray(mean, dtype=np.float64),
            bases=np.asarray(bases, dtype=np.float64),
            noise=np.asarray(noise, dtype=np.float64),
            formal_scores=np.asarray(bank.formal_scores, dtype=np.float64),
            responses=np.asarray(bank.responses, dtype=np.float64),
            evaluability=np.ones(len(bank.designs), dtype=np.float64),
            proxy_event_threshold=float(proxy_event_threshold),
        )
        return cls(world)

    def initial_state(self) -> TeacherState:
        return TeacherState(self.world)

    def _observe_oracle(self, state: TeacherState, source_index: int,
                        candidate_index: int) -> None:
        state.observe(
            candidate_index,
            float(self.world.responses[source_index, candidate_index]),
            float(self.world.formal_scores[source_index, candidate_index]),
        )

    def _continuation(self, state: TeacherState, source_index: int, remaining_steps: int,
                      policy: str) -> tuple[float, float]:
        if remaining_steps < 0:
            raise ValueError("remaining teacher budget cannot be negative")
        diagnostic_steps = {
            "mine_now": 0,
            "diagnose1_then_mine": 1,
            "diagnose2_then_mine": 2,
        }.get(policy)
        if diagnostic_steps is None:
            raise ValueError("unknown teacher continuation policy")
        formal = 0.0
        vulnerability = 0.0
        for step in range(remaining_steps):
            selection = "diagnostic" if step < diagnostic_steps else "mining"
            candidate = state.select_index(selection)
            formal_score = float(self.world.formal_scores[source_index, candidate])
            response = float(self.world.responses[source_index, candidate])
            self._observe_oracle(state, source_index, candidate)
            formal += formal_score
            vulnerability += response
        return formal, vulnerability

    def action_value(self, state: TeacherState, source_index: int, candidate_index: int,
                     remaining_budget: int) -> TeacherActionValue:
        """Evaluate one action without mutating ``state``.

        The b=1 path intentionally bypasses continuation replay, preserving the
        exact teacher identity required by the Stage 1 contract.
        """
        if remaining_budget < 1:
            raise ValueError("remaining teacher budget must be at least one")
        if candidate_index in state.selected_indexes:
            raise ValueError("counterfactual candidate was already selected")
        immediate_formal = float(self.world.formal_scores[source_index, candidate_index])
        immediate_vulnerability = float(self.world.responses[source_index, candidate_index])
        if remaining_budget == 1:
            return TeacherActionValue(
                candidate_index,
                immediate_formal,
                immediate_vulnerability,
                "terminal",
                immediate_formal,
                immediate_vulnerability,
            )
        returns: list[tuple[str, float, float]] = []
        for policy in CONTINUATION_POLICIES:
            branch = state.clone()
            self._observe_oracle(branch, source_index, candidate_index)
            future_formal, future_vulnerability = self._continuation(branch, source_index,
                                                                     remaining_budget - 1, policy)
            returns.append((policy, immediate_formal + future_formal,
                            immediate_vulnerability + future_vulnerability))
        # Formal reward is the primary teacher objective.  Dense vulnerability
        # remains a separately maximised auxiliary return, as required by the
        # Stage 1 formal/dense accounting contract.
        formal_best = max(returns, key=lambda item: (item[1], item[2], item[0]))
        vulnerability_best = max(returns, key=lambda item: (item[2], item[1], item[0]))
        return TeacherActionValue(
            candidate_index=candidate_index,
            q_formal=float(formal_best[1]),
            q_vulnerability=float(vulnerability_best[2]),
            best_continuation=formal_best[0],
            immediate_formal_score=immediate_formal,
            immediate_vulnerability_response=immediate_vulnerability,
        )

    def action_values(self, state: TeacherState, source_index: int,
                      remaining_budget: int) -> list[TeacherActionValue]:
        return [
            self.action_value(state, source_index, int(index), remaining_budget)
            for index in np.flatnonzero(state.available_mask())
        ]

    def run(self, source_index: int, total_budget: int, policy: str) -> TeacherEvaluation:
        """Replay a baseline or the portfolio policy against one source row."""
        if total_budget < 1 or total_budget > len(self.world.pool):
            raise ValueError("teacher budget must lie within the anchor pool")
        state = self.initial_state()
        formal_curve: list[float] = []
        vulnerability_curve: list[float] = []
        formal_total = 0.0
        vulnerability_total = 0.0
        for step in range(total_budget):
            remaining = total_budget - step
            if policy == "teacher":
                values = self.action_values(state, source_index, remaining)
                best = max(values,
                           key=lambda value:
                           (value.q_formal, value.q_vulnerability, -value.candidate_index))
                candidate = best.candidate_index
            elif policy == "fixed_k4":
                candidate = state.select_index("diagnostic" if step < 4 else "mining")
            elif policy == "mine_now":
                candidate = state.select_index("mining")
            elif policy == "frozen_source_mean" or policy == "highest_shared_risk":
                candidate = state.select_index("shared_risk")
            elif policy == "random":
                available = np.flatnonzero(state.available_mask())
                candidate = int(available[0])
            else:
                raise ValueError("unknown replay policy")
            formal = float(self.world.formal_scores[source_index, candidate])
            vulnerability = float(self.world.responses[source_index, candidate])
            self._observe_oracle(state, source_index, candidate)
            formal_total += formal
            vulnerability_total += vulnerability
            formal_curve.append(formal_total)
            vulnerability_curve.append(vulnerability_total)
        return TeacherEvaluation(
            policy=policy,
            source_index=source_index,
            formal_score=formal_total,
            cumulative_vulnerability=vulnerability_total,
            formal_curve=tuple(formal_curve),
            vulnerability_curve=tuple(vulnerability_curve),
            selected_indexes=tuple(state.history_indexes),
        )
