"""Pre-registered cache metrics independent of the selection kernel."""

from __future__ import annotations

import numpy as np

from .acquisition import rbf_by_mode
from .config import COVERAGE_LENGTH
from .data import CachedTask


def severity(task: CachedTask, indices: np.ndarray) -> np.ndarray:
    return np.where(task.target_collision[indices], 1.0, np.where(task.target_event[indices], 0.5, 0.0))


def coverage_utility(task: CachedTask, indices: np.ndarray) -> float:
    if not len(indices): return 0.0
    all_indices = np.arange(task.count)
    values = severity(task, indices)
    kernel = rbf_by_mode(task.features, task.modes, all_indices, indices, COVERAGE_LENGTH)
    unique, counts = np.unique(task.modes, return_counts=True)
    weights = np.zeros(task.count)
    for mode, count in zip(unique, counts, strict=True): weights[task.modes == mode] = 1.0 / (len(unique) * count)
    return float(np.sum(weights * np.max(kernel * values[None, :], axis=1)))


def cvs(task: CachedTask, indices: np.ndarray, bins: int = 4) -> float:
    best: dict[tuple[str, int, int], float] = {}
    for index, value in zip(indices, severity(task, indices), strict=True):
        mode = str(task.modes[index]); xy = task.features[index, :2]
        cell = tuple(np.clip(np.floor(xy * bins).astype(int), 0, bins).tolist())
        key = (mode, *cell); best[key] = max(best.get(key, 0.0), float(value))
    return float(sum(best.values()))


def record_metrics(task: CachedTask, queried: list[int], budget: int) -> dict[str, object]:
    indices = np.asarray(queried[:budget], dtype=int); values = severity(task, indices)
    modes = {str(mode): int(np.sum(task.modes[indices] == mode)) for mode in np.unique(task.modes)}
    return {
        "seed": task.seed, "target": task.target_name, "coverage": task.coverage, "heterogeneity": task.heterogeneity,
        "budget": budget, "CollisionCount": int(np.sum(task.target_collision[indices])),
        "CriticalCount": int(np.sum(task.target_event[indices])), "SeveritySum": float(values.sum()),
        "CVS": cvs(task, indices), "CollisionCells": int(np.sum(values == 1.0)),
        "F": coverage_utility(task, indices), "NewHistoricalFailureCount": int(np.sum(task.source_safe_target_failure[indices])),
        "function_queries": ";".join(f"{mode}:{count}" for mode, count in modes.items()),
        "queried_indices": ";".join(map(str, indices.tolist())),
    }
