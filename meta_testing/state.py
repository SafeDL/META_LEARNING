"""Canonical fixed-width state input for the Inner policy."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class CanonicalStateExtractor:
    """Convert a raw simulator observation to a finite, fixed-width vector."""

    def __init__(self, state_dim: int) -> None:
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        self.state_dim = int(state_dim)

    @staticmethod
    def _flatten(observation: Any) -> np.ndarray:
        if isinstance(observation, Mapping):
            parts = [CanonicalStateExtractor._flatten(observation[key]) for key in sorted(observation)]
            return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    def __call__(self, observation: Any) -> np.ndarray:
        values = self._flatten(observation)
        if values.size < self.state_dim:
            raise ValueError(f"raw simulator observation has {values.size} values; expected at least {self.state_dim}")
        return np.clip(np.nan_to_num(values[:self.state_dim], nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)
