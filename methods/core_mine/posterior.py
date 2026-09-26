"""Small, analytic compositional-residual posterior used by CoRe-Mine."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp, ndtr

from .config import COLLISION_THRESHOLD, EVENT_THRESHOLD, CoreMineConfig
from .data import CachedTask
from .oracle import RevealedOutcome


def _matern52(left: np.ndarray, right: np.ndarray, modes_left: np.ndarray, modes_right: np.ndarray, length: float, amplitude: float) -> np.ndarray:
    output = np.zeros((len(left), len(right)), dtype=float)
    for mode in np.unique(modes_left):
        li, ri = np.flatnonzero(modes_left == mode), np.flatnonzero(modes_right == mode)
        if len(li) and len(ri):
            radius = np.sqrt(np.sum((left[li, None] - right[ri][None, :]) ** 2, axis=2)) / length
            output[np.ix_(li, ri)] = amplitude**2 * (1.0 + math.sqrt(5.0) * radius + 5.0 * radius**2 / 3.0) * np.exp(-math.sqrt(5.0) * radius)
    return output


class PosteriorModel:
    """Source hypotheses plus optional local GP residual, exposing no target vector."""

    def __init__(self, task: CachedTask, config: CoreMineConfig, branch: str, residual: bool) -> None:
        self.task, self.config, self.branch, self.residual = task, config, branch, residual
        self.selected: list[int] = []
        self.outcomes: list[float] = []
        self._gp_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._modes = tuple(str(item) for item in np.unique(task.modes))
        self._source_bases: dict[str, np.ndarray] = {}
        self._log_weights: dict[str, np.ndarray] = {}
        for mode in self._modes:
            mask = task.modes == mode
            if branch == "composition":
                base = task.source_y.copy()
                # De-duplicate identical complete source hypotheses in this function.
                _, ids = np.unique(base[:, mask], axis=0, return_index=True)
                base = base[np.sort(ids)]
            elif branch == "global":
                base = task.source_y.copy()
            elif branch == "mean":
                base = task.source_y.mean(axis=0, keepdims=True)
            elif branch == "target":
                base = np.full((1, task.count), 0.50)
            else:
                raise ValueError(branch)
            if config.include_null and branch in {"composition", "global"}:
                base = np.vstack((base, np.full((1, task.count), 0.50)))
                prior = np.full(len(base), 0.90 / (len(base) - 1)); prior[-1] = 0.10
            else:
                prior = np.full(len(base), 1.0 / len(base))
            self._source_bases[mode] = base
            key = "__global__" if branch == "global" else mode
            if key not in self._log_weights:
                self._log_weights[key] = np.log(prior)

    def _key(self, mode: str) -> str:
        return "__global__" if self.branch == "global" else mode

    def _gp(self, index: int, base: np.ndarray) -> tuple[np.ndarray, float]:
        mode = str(self.task.modes[index])
        if mode not in self._gp_cache:
            seen = np.asarray([item for item in self.selected if self.task.modes[item] == mode], dtype=int)
            if len(seen) and self.residual:
                k_ii = _matern52(self.task.features[seen], self.task.features[seen], self.task.modes[seen], self.task.modes[seen], self.config.residual_length, self.config.residual_amplitude)
                k_ii.flat[::len(seen) + 1] += self.config.observation_noise**2 + 1e-9
                self._gp_cache[mode] = (seen, np.linalg.cholesky(k_ii), np.asarray([self.outcomes[self.selected.index(item)] for item in seen]))
            else:
                self._gp_cache[mode] = (seen, np.empty((0, 0)), np.empty(0))
        seen, chol, observed = self._gp_cache[mode]
        if not self.residual or not len(seen):
            return base[:, index].copy(), self.config.residual_amplitude**2 if self.residual else 0.0
        k_xi = _matern52(self.task.features[[index]], self.task.features[seen], self.task.modes[[index]], self.task.modes[seen], self.config.residual_length, self.config.residual_amplitude)[0]
        residuals = observed[None, :] - base[:, seen]
        solved = np.linalg.solve(chol.T, np.linalg.solve(chol, residuals.T)).T
        mean = base[:, index] + solved @ k_xi
        variance = max(self.config.residual_amplitude**2 - float(k_xi @ np.linalg.solve(chol.T, np.linalg.solve(chol, k_xi))), 1e-10)
        return mean, variance

    def predict(self) -> dict[str, np.ndarray]:
        n = self.task.count
        event, collision, mean, variance = (np.zeros(n) for _ in range(4))
        for mode in self._modes:
            indices = np.flatnonzero(self.task.modes == mode)
            base = self._source_bases[mode]
            # Populate/reuse the common Cholesky factor for this function.
            self._gp(int(indices[0]), base)
            seen, chol, observed = self._gp_cache[mode]
            if self.residual and len(seen):
                k_xi = _matern52(self.task.features[indices], self.task.features[seen], self.task.modes[indices], self.task.modes[seen], self.config.residual_length, self.config.residual_amplitude)
                residuals = observed[None, :] - base[:, seen]
                alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, residuals.T)).T
                mus = base[:, indices] + alpha @ k_xi.T
                solved = np.linalg.solve(chol.T, np.linalg.solve(chol, k_xi.T))
                var = np.maximum(self.config.residual_amplitude**2 - np.sum(k_xi.T * solved, axis=0), 1e-10)
            else:
                mus = base[:, indices]
                var = np.full(len(indices), self.config.residual_amplitude**2 if self.residual else 0.0)
            logw = self._log_weights[self._key(mode)]; weights = np.exp(logw - logsumexp(logw))
            scale = np.sqrt(var + self.config.observation_noise**2)
            event[indices] = weights @ ndtr((mus - EVENT_THRESHOLD) / scale[None, :])
            collision[indices] = weights @ ndtr((mus - COLLISION_THRESHOLD) / scale[None, :])
            mean[indices] = weights @ mus
            variance[indices] = var + weights @ (mus - mean[indices][None, :]) ** 2
        collision = np.minimum(collision, event)
        return {"p_event": event, "p_collision": collision, "mean": mean, "variance": np.maximum(variance, 1e-10)}

    def observe(self, index: int, outcome: RevealedOutcome) -> None:
        if index in self.selected:
            raise ValueError("each outcome may update the posterior once")
        mode = str(self.task.modes[index]); base = self._source_bases[mode]
        mus, var = self._gp(index, base)
        key = self._key(mode)
        # This is the predictive evidence before adding this observation.
        total_var = var + self.config.observation_noise**2
        self._log_weights[key] -= 0.5 * ((outcome.y - mus) ** 2 / total_var + math.log(2.0 * math.pi * total_var))
        self._log_weights[key] -= logsumexp(self._log_weights[key])
        self.selected.append(index); self.outcomes.append(outcome.y); self._gp_cache.clear()

    def propose(self, allowed_indices: np.ndarray | None = None) -> int:
        prediction = self.predict()
        available = np.ones(self.task.count, dtype=bool); available[np.asarray(self.selected, dtype=int)] = False if self.selected else True
        if allowed_indices is not None: available &= np.isin(np.arange(self.task.count), allowed_indices)
        candidates = np.flatnonzero(available)
        return int(candidates[np.lexsort((candidates, -prediction["p_event"][candidates]))[0]])
