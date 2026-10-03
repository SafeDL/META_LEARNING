"""Sparse expected-policy TD learning, explicitly not control Q-learning."""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

StateKey = tuple[int, ...]


@dataclass
class SparseExpectedTD:
    actions: int
    natural_policy: np.ndarray
    learning_rate: float = 0.25
    gamma: float = 1.0
    values: dict[StateKey, np.ndarray] = field(default_factory=dict)
    visits: dict[StateKey, np.ndarray] = field(
        default_factory=lambda: defaultdict(lambda: np.zeros(0, dtype=int)))

    def __post_init__(self) -> None:
        self.natural_policy = np.asarray(self.natural_policy, dtype=float)
        if self.natural_policy.shape != (self.actions, ) or np.any(
                self.natural_policy < 0) or not np.isclose(self.natural_policy.sum(), 1.0):
            raise ValueError("natural_policy must be a probability vector")
        if not 0 < self.learning_rate <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("invalid TD hyperparameters")

    def action_values(self, state: StateKey) -> np.ndarray:
        if state not in self.values:
            self.values[state] = np.zeros(self.actions, dtype=float)
            self.visits[state] = np.zeros(self.actions, dtype=int)
        return self.values[state]

    def update(self, state: StateKey, action: int, reward: float, next_state: StateKey | None,
               terminal: bool, critical: bool) -> float:
        current = self.action_values(state)
        self.visits[state][action] += 1
        if not critical:
            return 0.0
        bootstrap = 0.0 if terminal or next_state is None else float(
            self.natural_policy @ self.action_values(next_state))
        error = float(reward + self.gamma * bootstrap - current[action])
        current[action] += self.learning_rate * error
        return error

    def snapshot(self) -> dict[StateKey, np.ndarray]:
        return {key: value.copy() for key, value in self.values.items()}
