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
