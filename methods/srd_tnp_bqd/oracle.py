"""Budgeted one-index disclosure; full banks stay with trusted oracle owners."""
from __future__ import annotations

import numpy as np

from .data import Observation


class ContinuousOracle:
    def __init__(self, scenario_ids, provider, budget):
        self._scenario_ids = tuple(scenario_ids)
        self._provider = provider
        self.budget = int(budget)
        self.queried = set()

    def query(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self._scenario_ids):
            raise ValueError("invalid query index")
        if int(index) in self.queried:
            raise ValueError("duplicate query")
        if len(self.queried) >= self.budget:
            raise ValueError("visible query budget exhausted")
        label, risk, valid = self._provider(int(index))
        self.queried.add(int(index))
        if valid and (risk is None or not np.isfinite(risk) or not 0 <= risk <= 1):
            raise ValueError("oracle reported invalid risk as valid")
        return Observation(int(index), self._scenario_ids[index], label,
                           float(risk) if valid else None, bool(valid), len(self.queried))
