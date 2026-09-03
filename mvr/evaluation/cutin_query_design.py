"""Deterministic non-singleton Cut-in validation query design."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.task_spec import CUTIN_LOGICAL_PARAMETER_NAMES, ScenarioMiningTaskSpec


@dataclass(frozen=True)
class CutInValidationQuery:
    query_id: str
    design_kind: str
    candidate_index: int
    action: NormalizedScenarioAction


def _bounds(task: ScenarioMiningTaskSpec) -> tuple[np.ndarray, np.ndarray]:
    if tuple(task.logical_domain_bounds) != CUTIN_LOGICAL_PARAMETER_NAMES:
        raise ValueError("Cut-in validation task does not use canonical 5-D ordering")
    values = np.asarray(
        [task.logical_domain_bounds[name] for name in CUTIN_LOGICAL_PARAMETER_NAMES],
        dtype=float,
    )
    return values[:, 0], values[:, 1]


def _boundary_codes(count: int) -> list[int]:
    corners = 1 << len(CUTIN_LOGICAL_PARAMETER_NAMES)
    if not 1 <= count <= corners:
        raise ValueError("boundary query count exceeds the 5-D corner set")
    selected = [0, corners - 1]
    while len(selected) < count:
        best = max(
            (code for code in range(corners) if code not in selected),
            key=lambda code: (
                min((code ^ other).bit_count() for other in selected),
                -code,
            ),
        )
        selected.append(best)
    return selected[:count]


def build_cutin_validation_queries(
    task: ScenarioMiningTaskSpec,
    *,
    candidates: int,
    sobol_interior: int = 24,
    boundary: int = 8,
    seed: int = 11,
) -> list[CutInValidationQuery]:
    """Build 32 deterministic 5-D points for every discrete candidate."""
    if candidates < 1 or sobol_interior < 1 or boundary < 1:
        raise ValueError("Cut-in query design counts must be positive")
    lower, upper = _bounds(task)
    rows: list[CutInValidationQuery] = []
    for candidate in range(candidates):
        engine = torch.quasirandom.SobolEngine(
            dimension=len(CUTIN_LOGICAL_PARAMETER_NAMES),
            scramble=True,
            seed=int(seed + 1009 * candidate + task.geometry_seed),
        )
        unit = engine.draw(sobol_interior).cpu().numpy().astype(float)
        interior = lower + unit * (upper - lower)
        for index, point in enumerate(interior):
            action = NormalizedScenarioAction(
                candidate, tuple(float(value) for value in point)
            )
            rows.append(CutInValidationQuery(
                f"candidate-{candidate}:sobol-{index:02d}",
                "sobol_interior",
                candidate,
                action,
            ))
        for index, code in enumerate(_boundary_codes(boundary)):
            bits = np.asarray(
                [(code >> axis) & 1 for axis in range(len(lower))], dtype=float
            )
            point = lower + bits * (upper - lower)
            action = NormalizedScenarioAction(
                candidate, tuple(float(value) for value in point)
            )
            rows.append(CutInValidationQuery(
                f"candidate-{candidate}:boundary-{index:02d}",
                "boundary",
                candidate,
                action,
            ))
    actions = {
        (row.candidate_index, tuple(row.action.continuous)) for row in rows
    }
    if len(actions) != len(rows):
        raise RuntimeError("Cut-in validation query design contains duplicates")
    return rows
