"""Global low-rank factorization of eligible source vulnerability responses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorizedVulnerability:
    source_refs: tuple[str, ...]
    design_ids: tuple[str, ...]
    rank: int
    common_eligible_mask: np.ndarray
    mean: np.ndarray
    basis: np.ndarray
    source_latents: np.ndarray
    singular_values: np.ndarray
    residual_variance: float

    def prediction_for_source(self, source_index: int) -> np.ndarray:
        return self.mean + self.basis @ self.source_latents[source_index]


def fit_low_rank_vulnerability(
    scores: np.ndarray,
    eligible: np.ndarray,
    source_refs: tuple[str, ...],
    design_ids: tuple[str, ...],
    rank: int,
) -> FactorizedVulnerability:
    """Fit on columns eligible for every supplied source; never impute censored data."""
    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError("scores and eligible must be equally shaped source-by-design arrays")
    sources, designs = values.shape
    if sources != len(source_refs) or designs != len(design_ids):
        raise ValueError("source/design labels do not match response matrix")
    if len(set(source_refs)) != sources or len(set(design_ids)) != designs:
        raise ValueError("source refs and design ids must be unique")
    if not 0 <= int(rank) <= sources - 1:
        raise ValueError("rank must lie between zero and the centered source rank")
    common = mask.all(axis=0)
    if not common.any():
        raise ValueError("no common posterior-eligible source anchors")
    observed = values[:, common]
    mean_common = observed.mean(axis=0)
    centered = observed - mean_common
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if rank:
        latent = np.sqrt(sources - 1.0) * u[:, :rank]
        basis_common = vt[:rank].T * singular[:rank] / np.sqrt(sources - 1.0)
        reconstruction = latent @ basis_common.T
    else:
        latent = np.zeros((sources, 0), dtype=np.float64)
        basis_common = np.zeros((common.sum(), 0), dtype=np.float64)
        reconstruction = np.zeros_like(centered)
    mean = np.full(designs, np.nan, dtype=np.float64)
    basis = np.full((designs, rank), np.nan, dtype=np.float64)
    mean[common] = mean_common
    basis[common] = basis_common
    residual = centered - reconstruction
    return FactorizedVulnerability(
        source_refs=source_refs,
        design_ids=design_ids,
        rank=int(rank),
        common_eligible_mask=common,
        mean=mean,
        basis=basis,
        source_latents=latent,
        singular_values=singular,
        residual_variance=float(np.mean(np.square(residual))),
    )
