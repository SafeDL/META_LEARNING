"""Selection state for diagnostic and online DIVA failure mining."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .acquisition import diagnostic_scores, mining_scores, novelty_weight
from .posterior import LatentVulnerabilityPosterior
from .prior import LowRankVulnerabilityPrior
from .types import DivaCutInDesign, DivaObservation


@dataclass
class DivaMiner:
    prior: LowRankVulnerabilityPrior
    pool: tuple[DivaCutInDesign, ...]
    evaluability: np.ndarray
    proxy_event_threshold: float = 0.75
    posterior: LatentVulnerabilityPosterior = field(init=False)
    selected_ids: set[str] = field(default_factory=set, init=False)
    archive_features: list[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.posterior = LatentVulnerabilityPosterior.standard_normal(self.prior.rank)
        self.evaluability = np.asarray(self.evaluability, dtype=np.float64)
        if len(self.pool) != len(self.evaluability):
            raise ValueError("candidate pool and evaluability vector must align")

    def _available(self) -> np.ndarray:
        return np.asarray([design.design_id not in self.selected_ids for design in self.pool])

    @staticmethod
    def _select(scores: np.ndarray, available: np.ndarray, pool: tuple[DivaCutInDesign, ...]) -> DivaCutInDesign:
        if not available.any():
            raise RuntimeError("DIVA candidate pool is exhausted")
        viable = np.where(available, scores, -np.inf)
        maximum = np.nanmax(viable)
        ties = [index for index, value in enumerate(viable) if np.isclose(value, maximum)]
        return min((pool[index] for index in ties), key=lambda design: design.design_id)

    def select_diagnostic(self) -> DivaCutInDesign:
        mean, _, bases, noise = self.prior.predict(self.pool, self.posterior)
        scores = diagnostic_scores(
            self.posterior,
            bases,
            mean,
            noise,
            self.evaluability,
            level_set_threshold=self.proxy_event_threshold,
        )
        design = self._select(scores, self._available(), self.pool)
        self.selected_ids.add(design.design_id)
        return design

    def select_random(self, rng: np.random.Generator) -> DivaCutInDesign:
        indexes = np.flatnonzero(self._available())
        if not len(indexes):
            raise RuntimeError("DIVA candidate pool is exhausted")
        design = self.pool[int(rng.choice(indexes))]
        self.selected_ids.add(design.design_id)
        return design

    def select_mining(self) -> DivaCutInDesign:
        mean, variance, _, _ = self.prior.predict(self.pool, self.posterior)
        features = np.asarray([design.feature_vector() for design in self.pool])
        archive = np.asarray(self.archive_features, dtype=np.float64).reshape((-1, 7)) if self.archive_features else np.empty((0, 7))
        scores = mining_scores(mean, variance, self.evaluability, novelty_weight(features, archive))
        design = self._select(scores, self._available(), self.pool)
        self.selected_ids.add(design.design_id)
        return design

    def select_source_mean(self) -> DivaCutInDesign:
        """Frozen shared-mean comparator: no target update or latent UCB term."""
        zero = LatentVulnerabilityPosterior.standard_normal(self.prior.rank)
        mean, _, _, _ = self.prior.predict(self.pool, zero)
        scores = np.power(self.evaluability, 2.0) * np.clip(mean, 0.0, 1.0)
        design = self._select(scores, self._available(), self.pool)
        self.selected_ids.add(design.design_id)
        return design

    def observe(self, observation: DivaObservation) -> None:
        if observation.design.design_id not in self.selected_ids:
            raise ValueError("DIVA observation was not selected by this miner")
        if observation.posterior_eligible:
            mean, _, basis, noise = self.prior.predict((observation.design,), self.posterior)
            shared_mean = mean[0] - float(basis[0] @ self.posterior.mean)
            assert observation.vulnerability_response is not None
            self.posterior.update(
                basis[0], observation.vulnerability_response - shared_mean, noise[0]
            )
        if observation.score > 0.0:
            self.archive_features.append(observation.design.feature_vector())
