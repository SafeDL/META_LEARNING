"""Immutable numeric module contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SourceContext:
    source_id: str
    x: np.ndarray
    z: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class Observation:
    index: int
    scenario_id: str
    collision: Optional[int]
    risk: Optional[float]
    valid_risk: bool
    query_number: int


@dataclass(frozen=True)
class HistoryOutput:
    m: np.ndarray
    h: np.ndarray


@dataclass(frozen=True)
class PosteriorOutput:
    mean: np.ndarray
    latent_var: np.ndarray


def risk_logit(risk, epsilon=1e-4):
    r = np.asarray(risk, dtype=np.float64)
    if not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError("invalid measured risk")
    r = np.clip(r, epsilon, 1 - epsilon)
    return np.log(r) - np.log1p(-r)
