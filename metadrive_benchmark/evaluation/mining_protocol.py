"""Strict accounting contracts for Mining target-SUT evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MiningBudgetLedger:
    """Record every attempted physical call before it can be executed."""

    total_budget: int = 20
    entries: list[dict[str, Any]] = field(default_factory=list)

    def reserve(self, *, phase: str, design_id: str, episode_seed: int) -> int:
        if phase not in {"support", "mining", "query"}:
            raise ValueError("unknown Mining evaluation phase")
        if len(self.entries) >= self.total_budget:
            raise RuntimeError("Mining target episode budget is exhausted")
        entry = {
            "budget_index": len(self.entries) + 1,
            "phase": phase,
            "design_id": design_id,
            "episode_seed": int(episode_seed),
            "completed": False,
        }
        self.entries.append(entry)
        return len(self.entries) - 1

    def complete(self, index: int, *, status: str, score: float) -> None:
        entry = self.entries[index]
        if entry["completed"]:
            raise ValueError("a budget entry can only be completed once")
        entry.update({"completed": True, "status": status, "score": float(score)})

    @property
    def consumed(self) -> int:
        return len(self.entries)

    @property
    def cumulative_score(self) -> float:
        return float(sum(float(entry.get("score", 0.0)) for entry in self.entries))

    def report(self) -> dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "consumed": self.consumed,
            "cumulative_valid_critical_score": self.cumulative_score,
            "entries": self.entries,
        }
