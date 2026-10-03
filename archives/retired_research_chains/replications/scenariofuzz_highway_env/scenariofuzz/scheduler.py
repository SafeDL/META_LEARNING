"""Frequency-aware local seed scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corpus import LocalScenarioSeed


@dataclass
class FrequencyScheduler:
    seeds: tuple[LocalScenarioSeed, ...]
    rng: np.random.Generator
    visits: dict[str, int] = field(default_factory=dict)

    def choose(self) -> LocalScenarioSeed:
        weights = np.asarray([1.0 / (1.0 + self.visits.get(seed.seed_id, 0)) for seed in self.seeds])
        index = int(self.rng.choice(len(self.seeds), p=weights / weights.sum()))
        chosen = self.seeds[index]
        self.visits[chosen.seed_id] = self.visits.get(chosen.seed_id, 0) + 1
        return chosen

