"""Local failure expansion on a frozen S01 Sobol pool."""

from __future__ import annotations

import numpy as np


class DiverseFailureExpansion:
    def __init__(self, coordinates: np.ndarray, *, radius_factor: float = 1.5,
                 sigma_factor: float = 2.0, novelty_weight: float = 0.25,
                 pass_penalty: float = 0.25):
        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise ValueError("DFE requires normalized S01 four-dimensional coordinates")
        self.coordinates = np.asarray(coordinates, dtype=float)
        distances = self._pairwise(self.coordinates)
        np.fill_diagonal(distances, np.inf)
        self.h = float(np.median(np.min(distances, axis=1)))
        self.radius = radius_factor * self.h
        self.sigma = sigma_factor * self.h
        self.novelty_weight = novelty_weight
        self.pass_penalty = pass_penalty
        self.failures: list[int] = []
        self.local_probes: list[tuple[int, int, int]] = []  # origin, candidate, valid label

    @staticmethod
    def _pairwise(points: np.ndarray) -> np.ndarray:
        delta = points[:, None] - points[None, :]
        return np.sqrt(np.mean(delta * delta, axis=-1))

    def components(self) -> list[list[int]]:
        unseen = set(self.failures)
        components = []
        while unseen:
            root = min(unseen)
            unseen.remove(root)
            stack, group = [root], []
            while stack:
                i = stack.pop()
                group.append(i)
                near = [j for j in sorted(unseen) if self._distance(i, j) <= self.radius]
                unseen.difference_update(near)
                stack.extend(near)
            components.append(sorted(group))
        return components

    def representatives(self, component: list[int]) -> list[int]:
        selected = [min(component)]
        while len(selected) < min(4, len(component)):
            remaining = [i for i in component if i not in selected]
            selected.append(max(remaining, key=lambda i: min(self._distance(i, j) for j in selected)))
        return selected

    def _distance(self, a: int, b: int) -> float:
        delta = self.coordinates[a] - self.coordinates[b]
        return float(np.sqrt(np.mean(delta * delta)))

    def _direction(self, origin: int, candidate: int) -> np.ndarray:
        delta = self.coordinates[candidate] - self.coordinates[origin]
        norm = np.linalg.norm(delta)
        return delta / norm if norm else np.zeros_like(delta)

    def local_score(self, candidate: int, origin: int, probability: float) -> float:
        distance = self._distance(candidate, origin)
        v = self._direction(origin, candidate)
        prior_directions = [self._direction(seed, point) for seed, point, _ in self.local_probes
                            if seed == origin and point != origin]
        novelty = (min((1 - float(v @ old)) / 2 for old in prior_directions)
                   if prior_directions else 1.0)
        barrier = 1.0
        for seed, point, label in self.local_probes:
            if seed != origin or label != 0:
                continue
            pass_distance = self._distance(origin, point)
            if distance > pass_distance and float(v @ self._direction(origin, point)) > 0.9:
                barrier *= self.pass_penalty
        return probability * np.exp(-(distance**2) / (2 * self.sigma**2)) * (
            1 + self.novelty_weight * novelty) * barrier

    def propose(self, probabilities: np.ndarray, queried: set[int]) -> tuple[int, int] | None:
        if not self.failures:
            return None
        proposals = []
        for component in self.components():
            for origin in self.representatives(component):
                near = [(self._distance(i, origin), i) for i in range(len(self.coordinates))
                        if i not in queried and self._distance(i, origin) <= self.radius]
                for _, candidate in sorted(near)[:8]:
                    proposals.append((self.local_score(candidate, origin, float(probabilities[candidate])),
                                      candidate, origin))
        if not proposals:
            return None
        _, candidate, origin = max(proposals, key=lambda item: (item[0], -item[1]))
        return candidate, origin

    def observe(self, candidate: int, label: int | None, origin: int | None) -> None:
        if label == 1 and candidate not in self.failures:
            self.failures.append(candidate)
        if origin is not None and label is not None:
            self.local_probes.append((origin, candidate, label))
