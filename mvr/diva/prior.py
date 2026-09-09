"""Frozen functional vulnerability prior: global SVD plus candidate GP interpolation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .basis_gp import BasisGPBank
from .factorization import FactorizedVulnerability, fit_low_rank_vulnerability
from .posterior import LatentVulnerabilityPosterior
from .source_bank import SourceBank
from .types import DivaCutInDesign


@dataclass
class LowRankVulnerabilityPrior:
    factorization: FactorizedVulnerability
    gp_bank: BasisGPBank
    noise_floor_var: float = 0.02

    @classmethod
    def fit(
        cls, bank: SourceBank, rank: int, *, device: str = "cuda", gp_fit_steps: int = 100
    ) -> "LowRankVulnerabilityPrior":
        factorization = fit_low_rank_vulnerability(
            bank.scores, bank.eligible, bank.source_refs, bank.design_ids, rank
        )
        gp_bank = BasisGPBank(device=device, fit_steps=gp_fit_steps)
        common = factorization.common_eligible_mask
        for candidate in (0, 1):
            indexes = np.asarray([
                index for index, design in enumerate(bank.designs)
                if common[index] and design.candidate_index == candidate
            ], dtype=int)
            if len(indexes) < 2:
                raise ValueError("each candidate requires two common eligible anchors for GP fitting")
            gp_bank.fit(
                candidate,
                bank.features[indexes],
                factorization.mean[indexes],
                factorization.basis[indexes],
            )
        return cls(factorization, gp_bank)

    @property
    def rank(self) -> int:
        return self.factorization.rank

    def predict(self, designs: tuple[DivaCutInDesign, ...], posterior: LatentVulnerabilityPosterior) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return mean, variance, frozen basis, and effective noise variance."""
        if posterior.mean.shape != (self.rank,):
            raise ValueError("posterior rank does not match prior")
        means = np.empty(len(designs), dtype=np.float64)
        variances = np.empty(len(designs), dtype=np.float64)
        bases = np.empty((len(designs), self.rank), dtype=np.float64)
        noise = np.empty(len(designs), dtype=np.float64)
        for candidate in (0, 1):
            indexes = np.asarray([i for i, design in enumerate(designs) if design.candidate_index == candidate])
            if not len(indexes):
                continue
            features = np.asarray([designs[i].feature_vector() for i in indexes], dtype=np.float64)
            mean, mean_var, basis, basis_var = self.gp_bank.predict(candidate, features)
            latent_variance = np.einsum("ij,jk,ik->i", basis, posterior.covariance, basis)
            interpolation = mean_var
            if self.rank:
                interpolation = interpolation + ((np.square(posterior.mean) + np.diag(posterior.covariance)) * basis_var).sum(axis=1)
            means[indexes] = mean + basis @ posterior.mean
            noise[indexes] = self.noise_floor_var + self.factorization.residual_variance + interpolation
            variances[indexes] = latent_variance + noise[indexes]
            bases[indexes] = basis
        return means, variances, bases, noise

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str = "cpu") -> "LowRankVulnerabilityPrior":
        prior = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(prior, cls):
            raise ValueError("artifact is not a DIVA low-rank vulnerability prior")
        return prior
