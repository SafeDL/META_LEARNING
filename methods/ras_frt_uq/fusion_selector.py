"""Fuse local historical-response correction with target residual uncertainty."""

from __future__ import annotations

import numpy as np

from methods.ras_frt_uq.coverage_selector import corrected_risk
from methods.ras_frt_uq.transfer_uncertainty import TransferUncertainty


COVERAGE_WEIGHT = 0.2
NEW_CELL_BONUS = 0.2
INFORMATION_WEIGHT = 0.05


def select_fusion_sequence(
    prior: np.ndarray,
    response_similarity: np.ndarray,
    physical_kernel: np.ndarray,
    cell_ids: np.ndarray,
    oracle,
    budget: int,
    regularizer: float,
) -> tuple[list[int], list[int | None], np.ndarray]:
    """Find failures while updating a target-minus-history residual model.

    Uncertain locations retain the local response-similarity correction.
    The Gaussian residual posterior takes more weight as its variance falls.
    """
    model = TransferUncertainty(prior, physical_kernel)
    selected: list[int] = []
    observed: list[int | None] = []
    discovered_cells: set[int] = set()

    for _ in range(budget):
        local_risk = corrected_risk(
            prior, response_similarity, selected, observed, regularizer,
        )
        posterior_variance = np.clip(np.diag(model.covariance), 0, 1)
        risk = (posterior_variance * local_risk +
                (1 - posterior_variance) * model.risk())

        if selected:
            gap = 1 - response_similarity[:, selected].max(axis=1)
            score = (1 - COVERAGE_WEIGHT) * risk + COVERAGE_WEIGHT * gap
        else:
            score = risk.copy()
        if discovered_cells:
            new_cell = ~np.isin(cell_ids, list(discovered_cells))
            score += NEW_CELL_BONUS * risk * new_cell

        score = ((1 - INFORMATION_WEIGHT) * score +
                 INFORMATION_WEIGHT * model.information_gain())
        score[model.selected] = -np.inf

        index = int(np.argmax(score))
        label = oracle.query(index)
        selected.append(index)
        observed.append(label)
        if label == 1:
            discovered_cells.add(int(cell_ids[index]))
        model.observe(index, label)

    return selected, observed, model.risk()
