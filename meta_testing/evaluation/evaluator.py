from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..failure.metrics import FixedBudgetMetrics
from ..failure.signature import FailureSignature
from .budget_protocol import BudgetProtocol


@dataclass
class FixedBudgetEvaluator:
    protocol: BudgetProtocol

    def evaluate(self, execute_episode: Callable[[int], FailureSignature]) -> dict[str, float | int | None]:
        self.protocol.validate()
        metrics = FixedBudgetMetrics(self.protocol.total_episodes)
        for episode in range(self.protocol.total_episodes):
            metrics.add(execute_episode(episode))
        result = metrics.summary()
        result["support_shots"] = list(self.protocol.support_shots)
        result["all_in_budget"] = True
        return result
