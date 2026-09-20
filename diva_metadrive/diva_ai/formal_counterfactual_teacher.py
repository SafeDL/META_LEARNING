"""Time-sensitive counterfactual teacher aligned with formal failures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..diva.acquisition import novelty_weight
from ..diva.source_bank import SourceBank
from .counterfactual_teacher import CounterfactualTeacher
from .formal_calibrator import FORMAL_CLASS_SCORES, FormalEventCalibrator
from .state import TeacherState

CONTINUATION_POLICIES = (
    "vulnerability_mine_now",
    "diagnose1_then_vulnerability_mine",
    "diagnose2_then_vulnerability_mine",
    "formal_mine_now",
    "diagnose1_then_formal_mine",
    "diagnose2_then_formal_mine",
)

BASELINE_POLICIES = (
    "fixed_k4",
    "mine_now",
    "fixed_k4_formal",
    "formal_mine_now",
    "highest_shared_risk",
)


@dataclass(frozen=True)
class FormalTeacherActionValue:
    candidate_index: int
    q_auc_formal: float
    q_auc_vulnerability: float
    q_terminal_formal: float
    q_terminal_vulnerability: float
    normalized_q_auc_formal: float
    normalized_q_auc_vulnerability: float
    best_formal_continuation: str
    best_vulnerability_continuation: str
    immediate_formal_score: float
    immediate_vulnerability_response: float
    formal_probabilities: tuple[float, float, float]
    immediate_expected_formal_utility: float


@dataclass(frozen=True)
class FormalTeacherEvaluation:
    policy: str
    source_index: int
    total_budget: int
    formal_curve: tuple[float, ...]
    vulnerability_curve: tuple[float, ...]
    selected_indexes: tuple[int, ...]

    def score_at(self, budget: int) -> float:
        if budget < 1 or budget > self.total_budget:
            raise ValueError("score budget is outside the replay horizon")
        return float(self.formal_curve[budget - 1])

    def auc_at(self, budget: int) -> float:
        if budget < 1 or budget > self.total_budget:
            raise ValueError("AUC budget is outside the replay horizon")
        return float(sum(self.formal_curve[:budget]))

    @property
    def first_critical_step(self) -> int | None:
        return next((index + 1 for index, score in enumerate(self.formal_curve) if score > 0.0),
                    None)


class FormalCounterfactualTeacher:
    """AUC teacher with vulnerability- and formal-aligned continuations."""
    def __init__(self, base: CounterfactualTeacher, calibrator: FormalEventCalibrator) -> None:
        self.base = base
        self.world = base.world
        self.calibrator = calibrator

    @classmethod
    def from_source_bank(
        cls,
        bank: SourceBank,
        training_source_indexes: Iterable[int],
        *,
        rank: int = 2,
        proxy_event_threshold: float = 0.75,
        calibration_context_sizes: tuple[int, ...] = (0, 1, 2, 4),
        calibration_l2: float = 1e-3,
        calibration_max_iterations: int = 500,
    ) -> "FormalCounterfactualTeacher":
        indexes = tuple(int(index) for index in training_source_indexes)
        base = CounterfactualTeacher.from_source_bank(
            bank,
            indexes,
            rank=rank,
            proxy_event_threshold=proxy_event_threshold,
        )
        calibrator = FormalEventCalibrator.fit(
            base.world,
            indexes,
            context_sizes=calibration_context_sizes,
            l2=calibration_l2,
            max_iterations=calibration_max_iterations,
        )
        return cls(base, calibrator)

    def initial_state(self) -> TeacherState:
        return self.base.initial_state()

    def _observe_oracle(self, state: TeacherState, source_index: int,
                        candidate_index: int) -> None:
        self.base._observe_oracle(state, source_index, candidate_index)

    def formal_mining_values(self, state: TeacherState) -> np.ndarray:
        archive = (self.world.features[np.asarray(state.failure_indexes, dtype=int)]
                   if state.failure_indexes else np.empty(
                       (0, self.world.features.shape[1]), dtype=np.float64))
        return (self.calibrator.expected_utility(state) * np.square(self.world.evaluability) *
                (0.25 + 0.75 * novelty_weight(self.world.features, archive)))

    def _select_index(self, state: TeacherState, policy: str) -> int:
        if policy == "diagnostic":
            return state.select_index("diagnostic")
        if policy == "vulnerability_mining":
            return state.select_index("mining")
        if policy == "shared_risk":
            return state.select_index("shared_risk")
        if policy != "formal_mining":
            raise ValueError("unknown formal-teacher selection policy")
        values = np.where(state.available_mask(), self.formal_mining_values(state), -np.inf)
        maximum = np.max(values)
        ties = np.flatnonzero(np.isclose(values, maximum))
        return int(min(ties, key=lambda index: self.world.pool[int(index)].design_id))

    @staticmethod
    def _continuation_schedule(policy: str) -> tuple[int, str]:
        schedules = {
            "vulnerability_mine_now": (0, "vulnerability_mining"),
            "diagnose1_then_vulnerability_mine": (1, "vulnerability_mining"),
            "diagnose2_then_vulnerability_mine": (2, "vulnerability_mining"),
            "formal_mine_now": (0, "formal_mining"),
            "diagnose1_then_formal_mine": (1, "formal_mining"),
            "diagnose2_then_formal_mine": (2, "formal_mining"),
        }
        if policy not in schedules:
            raise ValueError("unknown formal-teacher continuation policy")
        return schedules[policy]

    def _continuation(self, state: TeacherState, source_index: int, remaining_steps: int,
                      policy: str) -> tuple[float, float, float, float]:
        if remaining_steps < 0:
            raise ValueError("remaining teacher budget cannot be negative")
        diagnostic_steps, mining_policy = self._continuation_schedule(policy)
        formal_terminal = vulnerability_terminal = 0.0
        formal_auc = vulnerability_auc = 0.0
        for step in range(remaining_steps):
            selection = "diagnostic" if step < diagnostic_steps else mining_policy
            candidate = self._select_index(state, selection)
            formal = float(self.world.formal_scores[source_index, candidate])
            vulnerability = float(self.world.responses[source_index, candidate])
            self._observe_oracle(state, source_index, candidate)
            weight = remaining_steps - step
            formal_terminal += formal
            vulnerability_terminal += vulnerability
            formal_auc += weight * formal
            vulnerability_auc += weight * vulnerability
        return formal_terminal, vulnerability_terminal, formal_auc, vulnerability_auc

    def action_value(self, state: TeacherState, source_index: int, candidate_index: int,
                     remaining_budget: int) -> FormalTeacherActionValue:
        if remaining_budget < 1:
            raise ValueError("remaining teacher budget must be at least one")
        if candidate_index in state.selected_indexes:
            raise ValueError("counterfactual candidate was already selected")
        formal = float(self.world.formal_scores[source_index, candidate_index])
        vulnerability = float(self.world.responses[source_index, candidate_index])
        probabilities = self.calibrator.predict_proba(state)[candidate_index]
        expected = float(probabilities @ FORMAL_CLASS_SCORES)
        if remaining_budget == 1:
            return FormalTeacherActionValue(
                candidate_index,
                formal,
                vulnerability,
                formal,
                vulnerability,
                formal,
                vulnerability,
                "terminal",
                "terminal",
                formal,
                vulnerability,
                tuple(float(value) for value in probabilities),
                expected,
            )

        returns: list[tuple[str, float, float, float, float]] = []
        for policy in CONTINUATION_POLICIES:
            branch = state.clone()
            self._observe_oracle(branch, source_index, candidate_index)
            terminal_f, terminal_v, auc_f, auc_v = self._continuation(
                branch, source_index, remaining_budget - 1, policy)
            returns.append((
                policy,
                formal + terminal_f,
                vulnerability + terminal_v,
                remaining_budget * formal + auc_f,
                remaining_budget * vulnerability + auc_v,
            ))
        formal_best = max(returns, key=lambda row: (row[3], row[1], row[4], row[0]))
        vulnerability_best = max(returns, key=lambda row: (row[4], row[2], row[3], row[0]))
        normalizer = remaining_budget * (remaining_budget + 1) / 2.0
        return FormalTeacherActionValue(
            candidate_index=candidate_index,
            q_auc_formal=float(formal_best[3]),
            q_auc_vulnerability=float(vulnerability_best[4]),
            q_terminal_formal=float(max(row[1] for row in returns)),
            q_terminal_vulnerability=float(max(row[2] for row in returns)),
            normalized_q_auc_formal=float(formal_best[3] / normalizer),
            normalized_q_auc_vulnerability=float(vulnerability_best[4] / normalizer),
            best_formal_continuation=formal_best[0],
            best_vulnerability_continuation=vulnerability_best[0],
            immediate_formal_score=formal,
            immediate_vulnerability_response=vulnerability,
            formal_probabilities=tuple(float(value) for value in probabilities),
            immediate_expected_formal_utility=expected,
        )

    def action_values(self, state: TeacherState, source_index: int,
                      remaining_budget: int) -> list[FormalTeacherActionValue]:
        return [
            self.action_value(state, source_index, int(index), remaining_budget)
            for index in np.flatnonzero(state.available_mask())
        ]

    def run(self, source_index: int, total_budget: int, policy: str) -> FormalTeacherEvaluation:
        if total_budget < 1 or total_budget > len(self.world.pool):
            raise ValueError("teacher budget must lie within the anchor pool")
        state = self.initial_state()
        formal_curve: list[float] = []
        vulnerability_curve: list[float] = []
        formal_total = vulnerability_total = 0.0
        for step in range(total_budget):
            remaining = total_budget - step
            if policy == "teacher":
                values = self.action_values(state, source_index, remaining)
                candidate = max(
                    values,
                    key=lambda value: (
                        value.q_auc_formal,
                        value.q_auc_vulnerability,
                        -value.candidate_index,
                    ),
                ).candidate_index
            elif policy in {"fixed_k4", "fixed_k4_formal"}:
                if step < 4:
                    candidate = self._select_index(state, "diagnostic")
                else:
                    mining = "formal_mining" if policy.endswith(
                        "formal") else "vulnerability_mining"
                    candidate = self._select_index(state, mining)
            elif policy == "mine_now":
                candidate = self._select_index(state, "vulnerability_mining")
            elif policy == "formal_mine_now":
                candidate = self._select_index(state, "formal_mining")
            elif policy == "highest_shared_risk":
                candidate = self._select_index(state, "shared_risk")
            else:
                raise ValueError("unknown formal-teacher replay policy")
            formal = float(self.world.formal_scores[source_index, candidate])
            vulnerability = float(self.world.responses[source_index, candidate])
            self._observe_oracle(state, source_index, candidate)
            formal_total += formal
            vulnerability_total += vulnerability
            formal_curve.append(formal_total)
            vulnerability_curve.append(vulnerability_total)
        return FormalTeacherEvaluation(
            policy=policy,
            source_index=source_index,
            total_budget=total_budget,
            formal_curve=tuple(formal_curve),
            vulnerability_curve=tuple(vulnerability_curve),
            selected_indexes=tuple(state.history_indexes),
        )
