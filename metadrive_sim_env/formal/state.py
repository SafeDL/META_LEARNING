"""Replayable analytic belief state used by the offline teacher.

This module deliberately has no simulator imports.  A state contains only a
frozen prior interpolation, observations already revealed by the source bank,
and the analytic posterior induced by those observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mining.acquisition import diagnostic_scores, mining_scores, novelty_weight
from ..mining.posterior import LatentVulnerabilityPosterior
from ..mining.types import CutInDesign


@dataclass(frozen=True)
class TeacherWorld:
    """Prior predictions and source-bank outcomes aligned to one anchor pool."""

    pool: tuple[CutInDesign, ...]
    base_mean: np.ndarray
    bases: np.ndarray
    noise: np.ndarray
    formal_scores: np.ndarray
    responses: np.ndarray
    evaluability: np.ndarray
    proxy_event_threshold: float

    def __post_init__(self) -> None:
        count = len(self.pool)
        if self.base_mean.shape != (count,) or self.noise.shape != (count,):
            raise ValueError("teacher prior vectors must align with its pool")
        if self.bases.shape[0] != count:
            raise ValueError("teacher basis matrix must align with its pool")
        if self.formal_scores.shape != self.responses.shape:
            raise ValueError("teacher formal and vulnerability matrices must align")
        if self.formal_scores.shape[1] != count:
            raise ValueError("teacher source outcomes must align with its pool")
        if self.evaluability.shape != (count,):
            raise ValueError("teacher evaluability must align with its pool")

    @property
    def rank(self) -> int:
        return int(self.bases.shape[1])

    @property
    def features(self) -> np.ndarray:
        # Candidate identity is scenario geometry, not SUT identity.  It keeps
        # novelty distances meaningful across the two physical route options.
        return np.asarray(
            [
                (float(design.candidate_index), *design.feature_vector())
                for design in self.pool
            ],
            dtype=np.float64,
        )


@dataclass
class TeacherState:
    """Mutable replay state with explicit clone and observe contracts."""

    world: TeacherWorld
    posterior: LatentVulnerabilityPosterior | None = None
    selected_indexes: set[int] = field(default_factory=set)
    history_indexes: list[int] = field(default_factory=list)
    history_responses: list[float] = field(default_factory=list)
    history_formal_scores: list[float] = field(default_factory=list)
    failure_indexes: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.posterior is None:
            self.posterior = LatentVulnerabilityPosterior.standard_normal(self.world.rank)
        if self.posterior.mean.shape != (self.world.rank,):
            raise ValueError("teacher posterior rank does not match its world")

    def clone(self) -> "TeacherState":
        """Return an independent state suitable for counterfactual replay."""
        assert self.posterior is not None
        return TeacherState(
            world=self.world,
            posterior=LatentVulnerabilityPosterior(
                self.posterior.mean.copy(), self.posterior.covariance.copy()
            ),
            selected_indexes=set(self.selected_indexes),
            history_indexes=list(self.history_indexes),
            history_responses=list(self.history_responses),
            history_formal_scores=list(self.history_formal_scores),
            failure_indexes=list(self.failure_indexes),
        )

    def available_mask(self) -> np.ndarray:
        return np.asarray(
            [index not in self.selected_indexes for index in range(len(self.world.pool))],
            dtype=bool,
        )

    def predicted_vulnerability(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.posterior is not None
        mean = self.world.base_mean + self.world.bases @ self.posterior.mean
        variance = np.einsum(
            "ij,jk,ik->i", self.world.bases, self.posterior.covariance, self.world.bases
        ) + self.world.noise
        return mean, variance

    def diagnostic_values(self) -> np.ndarray:
        assert self.posterior is not None
        return diagnostic_scores(
            self.posterior,
            self.world.bases,
            self.world.base_mean,
            self.world.noise,
            self.world.evaluability,
            level_set_threshold=self.world.proxy_event_threshold,
        )

    def mining_values(self) -> np.ndarray:
        mean, variance = self.predicted_vulnerability()
        archive = (
            self.world.features[np.asarray(self.failure_indexes, dtype=int)]
            if self.failure_indexes
            else np.empty((0, self.world.features.shape[1]), dtype=np.float64)
        )
        return mining_scores(
            mean,
            variance,
            self.world.evaluability,
            novelty_weight(self.world.features, archive),
        )

    def shared_risk_values(self) -> np.ndarray:
        return np.square(self.world.evaluability) * np.clip(self.world.base_mean, 0.0, 1.0)

    def select_index(self, policy: str) -> int:
        if policy == "diagnostic":
            scores = self.diagnostic_values()
        elif policy == "mining":
            scores = self.mining_values()
        elif policy == "shared_risk":
            scores = self.shared_risk_values()
        else:
            raise ValueError("unknown analytic teacher selection policy")
        available = self.available_mask()
        if not available.any():
            raise RuntimeError("teacher anchor pool is exhausted")
        viable = np.where(available, scores, -np.inf)
        maximum = np.max(viable)
        ties = np.flatnonzero(np.isclose(viable, maximum))
        return int(min(ties, key=lambda index: self.world.pool[int(index)].design_id))

    def observe(self, index: int, response: float, formal_score: float) -> None:
        """Reveal one source-bank outcome and update the analytic posterior."""
        if index in self.selected_indexes:
            raise ValueError("teacher cannot observe an already selected anchor")
        if not 0 <= index < len(self.world.pool):
            raise IndexError("teacher observation index is outside the anchor pool")
        if not 0.0 <= float(response) <= 1.0:
            raise ValueError("teacher vulnerability response must lie in [0, 1]")
        if float(formal_score) not in (0.0, 0.5, 1.0):
            raise ValueError("teacher formal score must use the frozen Mining scale")
        assert self.posterior is not None
        basis = self.world.bases[index]
        self.posterior.update(
            basis,
            float(response) - float(self.world.base_mean[index]),
            float(self.world.noise[index]),
        )
        self.selected_indexes.add(index)
        self.history_indexes.append(index)
        self.history_responses.append(float(response))
        self.history_formal_scores.append(float(formal_score))
        if formal_score > 0.0:
            self.failure_indexes.append(index)
