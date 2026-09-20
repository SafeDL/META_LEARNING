"""Discrete single-swap adaptation of the paper's continuous set optimizer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


LossFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SetOptimizationResult:
    indices: np.ndarray
    loss: float
    trace: list[dict[str, float | int | str]]
    candidate_evaluations: int


def discrete_single_swap(
    initial_indices: np.ndarray,
    candidate_count: int,
    loss_function: LossFunction,
    max_rounds: int = 3,
    tolerance: float = 1e-12,
) -> SetOptimizationResult:
    """Coordinate descent over unique candidate indices with full-set recomputation."""
    selected = np.asarray(initial_indices, dtype=int).copy()
    if selected.ndim != 1 or len(set(selected.tolist())) != len(selected):
        raise ValueError("initial_indices must be a unique one-dimensional set")
    if np.any(selected < 0) or np.any(selected >= candidate_count):
        raise ValueError("initial index outside candidate pool")
    current = float(loss_function(selected[None, :])[0])
    evaluations = 1
    start = time.perf_counter()
    trace: list[dict[str, float | int | str]] = [{
        "round": 0,
        "position": -1,
        "accepted_index": -1,
        "loss": current,
        "candidate_evaluations": evaluations,
        "elapsed_seconds": 0.0,
        "event": "initial",
    }]
    all_indices = np.arange(candidate_count)
    for round_index in range(1, max_rounds + 1):
        improved = False
        for position in range(len(selected)):
            occupied = np.delete(selected, position)
            candidates = np.setdiff1d(all_indices, occupied, assume_unique=False)
            proposals = np.repeat(selected[None, :], len(candidates), axis=0)
            proposals[:, position] = candidates
            losses = np.asarray(loss_function(proposals), dtype=float)
            evaluations += len(proposals)
            best_position = int(np.argmin(losses))
            best_loss = float(losses[best_position])
            accepted = -1
            if best_loss < current - tolerance:
                accepted = int(candidates[best_position])
                selected[position] = accepted
                current = best_loss
                improved = True
            trace.append({
                "round": round_index,
                "position": position,
                "accepted_index": accepted,
                "loss": current,
                "candidate_evaluations": evaluations,
                "elapsed_seconds": time.perf_counter() - start,
                "event": "accepted" if accepted >= 0 else "no_change",
            })
        if not improved:
            break
    return SetOptimizationResult(selected, current, trace, evaluations)

