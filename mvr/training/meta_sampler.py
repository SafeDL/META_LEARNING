"""Sample transferable tasks across independent SUT and geometry axes."""
from __future__ import annotations

import random
from typing import Sequence

from ..scenario.task_spec import ScenarioMiningTaskSpec


class MetaTaskSampler:
    def __init__(self, tasks: Sequence[ScenarioMiningTaskSpec]) -> None:
        if not tasks:
            raise ValueError("meta-task sampler requires at least one task")
        self.tasks = tuple(tasks)

    def sample(self) -> ScenarioMiningTaskSpec:
        return random.choice(self.tasks)

    def shuffled_epoch(self) -> list[ScenarioMiningTaskSpec]:
        """Return one shuffled visit of every configured training task."""
        tasks = list(self.tasks)
        random.shuffle(tasks)
        return tasks

    def epochs(self, count: int) -> list[ScenarioMiningTaskSpec]:
        """Return ``count`` complete epochs in a freshly shuffled order."""
        if count < 1:
            raise ValueError("epoch count must be positive")
        return [task for _ in range(count) for task in self.shuffled_epoch()]
