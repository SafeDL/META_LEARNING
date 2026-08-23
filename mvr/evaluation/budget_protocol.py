from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetProtocol:
    total_episodes: int = 20
    support_shots: tuple[int, ...] = (0, 1, 2, 4)

    def validate(self) -> None:
        if self.total_episodes < 1 or not self.support_shots or any(shot < 0 or shot > self.total_episodes for shot in self.support_shots):
            raise ValueError("invalid all-in fixed-budget protocol")
