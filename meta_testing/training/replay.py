"""Separate inner-transition and outer-episode replay prevents evidence leakage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import random


@dataclass(frozen=True)
class InnerTransition:
    episode_id: str
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool


@dataclass(frozen=True)
class OuterTransition:
    episode_id: str
    history_before: Any
    action: Any
    reward: float
    history_after: Any


@dataclass
class InnerReplay:
    capacity: int = 100_000
    rows: list[InnerTransition] = field(default_factory=list)

    def add(self, row: InnerTransition) -> None:
        self.rows.append(row)
        del self.rows[:-self.capacity]

    def sample(self, count: int, *, excluded_episode_ids: set[str] | None = None, rng: random.Random | None = None) -> list[InnerTransition]:
        eligible = [row for row in self.rows if row.episode_id not in (excluded_episode_ids or set())]
        if len(eligible) < count:
            raise ValueError("not enough leakage-free inner transitions")
        return (rng or random).sample(eligible, count)


@dataclass
class OuterReplay:
    capacity: int = 20_000
    rows: list[OuterTransition] = field(default_factory=list)

    def add(self, row: OuterTransition) -> None:
        self.rows.append(row)
        del self.rows[:-self.capacity]

    def sample(self, count: int, *, rng: random.Random | None = None) -> list[OuterTransition]:
        if len(self.rows) < count:
            raise ValueError("not enough outer transitions")
        return (rng or random).sample(self.rows, count)
