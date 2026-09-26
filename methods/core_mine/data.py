"""Response-bank adaptation with mode-aware physical feature masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from highway_sim_env.data.response_bank import ResponseBank

from .config import SOURCE_COUNT


@dataclass(frozen=True)
class CachedTask:
    seed: int
    target_name: str
    coverage: str
    heterogeneity: str
    anchors: np.ndarray
    modes: np.ndarray
    controls: np.ndarray
    features: np.ndarray
    active_dimensions: tuple[tuple[int, ...], ...]
    source_y: np.ndarray
    source_event: np.ndarray
    source_collision: np.ndarray
    target_y: np.ndarray
    target_event: np.ndarray
    target_collision: np.ndarray
    target_ttc: np.ndarray
    source_safe_target_failure: np.ndarray

    @property
    def count(self) -> int:
        return len(self.modes)


def response_value(ttc: np.ndarray, event: np.ndarray, collision: np.ndarray) -> np.ndarray:
    """The pre-registered shared continuous response, with missing TTC mapped to zero."""
    ttc = np.asarray(ttc, dtype=float)
    base = np.where(np.isfinite(ttc), np.exp(-np.maximum(ttc, 0.0) / 3.0), 0.0)
    return 0.25 * base + 0.5 * np.asarray(event, dtype=float) + 0.5 * np.asarray(collision, dtype=float)


def _target_metadata(name: str) -> tuple[str, str]:
    bits = name.split("-")
    if len(bits) < 4 or bits[0] != "Target":
        raise ValueError(f"unexpected frozen target name: {name}")
    return bits[1], bits[2]


def _features(anchors: np.ndarray, modes: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Normalize only schedule dimensions that physically affect each mode."""
    raw = np.column_stack((anchors, controls))
    output = np.zeros_like(raw, dtype=float)
    dimensions: list[tuple[int, ...]] = []
    for mode in np.unique(modes):
        index = np.flatnonzero(modes == mode)
        # slow_lead_following has no controlled schedule; all retained modes otherwise do.
        active = (0, 1) if str(mode) == "slow_lead_following" else (0, 1, 2, 3)
        values = raw[index][:, active]
        low, high = values.min(axis=0), values.max(axis=0)
        span = np.where(high > low, high - low, 1.0)
        output[np.ix_(index, active)] = (values - low) / span
    for mode in modes:
        dimensions.append((0, 1) if str(mode) == "slow_lead_following" else (0, 1, 2, 3))
    return output, tuple(dimensions)


def load_tasks(path: Path, seed: int) -> tuple[CachedTask, ...]:
    bank = ResponseBank.load(path)
    if bank.modes is None or bank.scenario_controls is None:
        raise ValueError(f"bank {path} lacks required mode/control fields")
    keep = np.asarray(bank.modes, dtype=str) != "passing_cutin"
    modes = np.asarray(bank.modes, dtype=str)[keep]
    anchors = np.asarray(bank.anchors, dtype=float)[keep]
    controls = np.asarray(bank.scenario_controls, dtype=float)[keep]
    features, dimensions = _features(anchors, modes, controls)
    source_event = (bank.collisions[:SOURCE_COUNT] | bank.near_misses[:SOURCE_COUNT])[:, keep]
    source_collision = bank.collisions[:SOURCE_COUNT, keep]
    source_y = response_value(bank.min_ttc[:SOURCE_COUNT, keep], source_event, source_collision)
    tasks = []
    for target_index, name in enumerate(bank.sut_names[SOURCE_COUNT:], start=SOURCE_COUNT):
        event = (bank.collisions[target_index] | bank.near_misses[target_index])[keep]
        collision = bank.collisions[target_index, keep]
        coverage, heterogeneity = _target_metadata(name)
        tasks.append(CachedTask(
            seed, name, coverage, heterogeneity, anchors, modes, controls, features, dimensions,
            source_y, source_event, source_collision,
            response_value(bank.min_ttc[target_index, keep], event, collision), event, collision,
            bank.min_ttc[target_index, keep], (~source_event.any(axis=0)) & event,
        ))
    return tuple(tasks)
