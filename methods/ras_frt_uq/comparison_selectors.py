"""Simple target-only and coverage baselines for a fixed candidate bank."""

from __future__ import annotations

import numpy as np

from methods.ras_frt_uq.transfer_uncertainty import TransferUncertainty


GP_UCB_BETA = 0.5
GP_INITIAL_QUERIES = 10


def farthest_first(
    coordinates: np.ndarray, oracle, budget: int, seed: int,
) -> tuple[list[int], list[int | None], None]:
    """Cover the scenario coordinates without using target feedback."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    observed: list[int | None] = []
    nearest_distance = np.full(len(coordinates), np.inf)

    for _ in range(budget):
        index = (int(rng.integers(len(coordinates))) if not selected else
                 int(np.argmax(nearest_distance)))
        selected.append(index)
        observed.append(oracle.query(index))
        distance = np.sum((coordinates - coordinates[index]) ** 2, axis=1)
        nearest_distance = np.minimum(nearest_distance, distance)
        nearest_distance[selected] = -np.inf
    return selected, observed, None


def target_gp_ucb(
    kernel: np.ndarray, oracle, budget: int, seed: int,
) -> tuple[list[int], list[int | None], np.ndarray]:
    """GP upper confidence search using only queried target collision labels.

    Binary feedback is treated as a noisy numeric response. The score is an
    adapted GP-UCB heuristic, not a calibrated failure probability bound.
    """
    rng = np.random.default_rng(seed)
    model = TransferUncertainty(np.zeros(len(kernel)), kernel)
    initial = rng.choice(len(kernel), size=min(GP_INITIAL_QUERIES, budget),
                         replace=False)
    selected: list[int] = []
    observed: list[int | None] = []

    for step in range(budget):
        if step < len(initial):
            index = int(initial[step])
        else:
            uncertainty = np.sqrt(np.clip(np.diag(model.covariance), 0, None))
            score = model.mean_residual + GP_UCB_BETA * uncertainty
            score[model.selected] = -np.inf
            index = int(np.argmax(score))
        label = oracle.query(index)
        selected.append(index)
        observed.append(label)
        model.observe(index, label)
    return selected, observed, model.risk()
