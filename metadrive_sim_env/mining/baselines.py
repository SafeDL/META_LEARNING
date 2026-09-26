"""Small, explicit Mining comparators with the same no-repeat and budget contracts."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .basis_gp import BasisGPBank
from .types import CutInDesign, MiningObservation


@dataclass
class TargetOnlyGP:
    """Candidate-specific exact GP rebuilt from target observations only."""

    pool: tuple[CutInDesign, ...]
    rng: np.random.Generator
    device: str = "cuda"
    selected_ids: set[str] = field(default_factory=set)
    observations: list[MiningObservation] = field(default_factory=list)
    archive: list[np.ndarray] = field(default_factory=list)

    def _available(self) -> list[CutInDesign]:
        return [design for design in self.pool if design.design_id not in self.selected_ids]

    def select(self) -> CutInDesign:
        available = self._available()
        if not available:
            raise RuntimeError("target-only GP candidate pool is exhausted")
        usable = [row for row in self.observations if row.posterior_eligible]
        by_candidate = {
            candidate: [row for row in usable if row.design.candidate_index == candidate]
            for candidate in (0, 1)
        }
        underrepresented = [candidate for candidate in (0, 1) if len(by_candidate[candidate]) < 2]
        if underrepresented:
            options = [row for row in available if row.candidate_index in underrepresented]
            choice = options[int(self.rng.integers(len(options)))]
            self.selected_ids.add(choice.design_id)
            return choice
        scores = np.full(len(available), -np.inf, dtype=np.float64)
        for candidate in (0, 1):
            train = by_candidate[candidate]
            query_indexes = [
                index for index, design in enumerate(available)
                if design.candidate_index == candidate
            ]
            if not query_indexes:
                continue
            bank = BasisGPBank(device=self.device, fit_steps=40)
            features = np.asarray([row.design.feature_vector() for row in train])
            targets = np.asarray([row.score for row in train])
            bank.fit(candidate, features, targets, np.empty((len(train), 0)))
            query = np.asarray([available[index].feature_vector() for index in query_indexes])
            mean, variance, _, _ = bank.predict(candidate, query)
            scores[query_indexes] = np.clip(mean + 0.5 * np.sqrt(variance), 0.0, 1.0)
        maximum = np.nanmax(scores)
        ties = [
            available[index] for index, value in enumerate(scores) if np.isclose(value, maximum)
        ]
        choice = min(ties, key=lambda design: design.design_id)
        self.selected_ids.add(choice.design_id)
        return choice

    def observe(self, observation: MiningObservation) -> None:
        self.observations.append(observation)
        if observation.score > 0.0:
            self.archive.append(observation.design.feature_vector())
