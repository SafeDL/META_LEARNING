"""Mode-level intercept plus local GP correction of historical safety response."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from method_chains.core_mine.config import COLLISION_THRESHOLD, EVENT_THRESHOLD
from method_chains.core_mine.data import CachedTask
from method_chains.core_mine.posterior import _matern52


class HierarchicalResidualModel:
    """Analytic fixed-kernel posterior, updated only by revealed target calls."""

    def __init__(self, task: CachedTask, *, global_amplitude: float = 0.35,
                 local_amplitude: float = 0.15, length: float = 0.30,
                 observation_noise: float = 0.05) -> None:
        if min(global_amplitude, local_amplitude, length,
               observation_noise) <= 0:
            raise ValueError("kernel/noise parameters must be positive")
        self.task = task
        self.global_amplitude = global_amplitude
        self.local_amplitude = local_amplitude
        self.length = length
        self.observation_noise = observation_noise
        self.source_mean = task.source_y.mean(axis=0)
        self.selected: list[int] = []
        self.outcomes: list[float] = []

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        local = _matern52(self.task.features[left], self.task.features[right],
                          self.task.modes[left], self.task.modes[right],
                          self.length, self.local_amplitude)
        same_mode = self.task.modes[left, None] == self.task.modes[None, right]
        return local + self.global_amplitude**2 * same_mode

    def predict(self) -> dict[str, np.ndarray]:
        mean = self.source_mean.copy()
        variance = np.full(self.task.count,
                           self.global_amplitude**2 + self.local_amplitude**2)
        for mode in np.unique(self.task.modes):
            candidates = np.flatnonzero(self.task.modes == mode)
            seen_positions = [position for position, index
                              in enumerate(self.selected)
                              if self.task.modes[index] == mode]
            if not seen_positions:
                continue
            seen = np.asarray([self.selected[position]
                               for position in seen_positions], dtype=int)
            observed = np.asarray([self.outcomes[position]
                                   for position in seen_positions], dtype=float)
            gram = self._kernel(seen, seen)
            gram.flat[::len(seen) + 1] += self.observation_noise**2 + 1e-9
            chol = np.linalg.cholesky(gram)
            cross = self._kernel(candidates, seen)
            residual = observed - self.source_mean[seen]
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
            mean[candidates] += cross @ alpha
            solved = np.linalg.solve(chol, cross.T)
            variance[candidates] -= np.sum(solved * solved, axis=0)
        variance = np.maximum(variance, 1e-10)
        scale = np.sqrt(variance + self.observation_noise**2)
        p_event = ndtr((mean - EVENT_THRESHOLD) / scale)
        p_collision = np.minimum(ndtr((mean - COLLISION_THRESHOLD) / scale),
                                 p_event)
        return {"mean": mean, "variance": variance,
                "p_event": p_event, "p_collision": p_collision}

    def observe(self, index: int, response: float) -> None:
        if index in self.selected:
            raise ValueError("each target result must be charged once")
        if not 0 <= index < self.task.count or not np.isfinite(response):
            raise ValueError("invalid charged target response")
        self.selected.append(index)
        self.outcomes.append(float(response))
