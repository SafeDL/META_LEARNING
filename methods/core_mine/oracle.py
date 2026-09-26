"""Selective cache oracle: complete labels never enter a selector directly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import CachedTask


@dataclass(frozen=True)
class RevealedOutcome:
    index: int
    y: float
    event: bool
    collision: bool
    ttc: float

    @property
    def severity(self) -> float:
        return 1.0 if self.collision else 0.5 if self.event else 0.0


class CacheOracle:
    def __init__(self, task: CachedTask) -> None:
        self._task = task
        self.revealed: list[int] = []

    def reveal(self, index: int) -> RevealedOutcome:
        if index in self.revealed:
            raise ValueError("a scenario cannot be revealed or charged twice")
        if not 0 <= index < self._task.count:
            raise IndexError(index)
        self.revealed.append(index)
        return RevealedOutcome(index, float(self._task.target_y[index]), bool(self._task.target_event[index]),
                               bool(self._task.target_collision[index]), float(self._task.target_ttc[index]))
