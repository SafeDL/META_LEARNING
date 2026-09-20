"""Paper-faithful broad then random-neighbour mutations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corpus import LocalScenarioSeed, ScenarioSpec


@dataclass(frozen=True)
class CandidateBatch:
    candidates: tuple[ScenarioSpec, ...]
    attempted: int
    rejected: tuple[dict, ...]
    duplicates: int
    stage: str
    reference_id: str | None


def _make_candidate(gap: float, speed: float, mode: str) -> ScenarioSpec:
    return ScenarioSpec.create(round(float(gap), 4), round(float(speed), 4), mode)


def generate_candidates(
    seed: LocalScenarioSeed,
    count: int,
    rng: np.random.Generator,
    *,
    reference: ScenarioSpec | None = None,
    gap_step: float = 0.5,
    speed_step: float = 0.25,
    max_attempt_factor: int = 8,
) -> CandidateBatch:
    """Generate unique candidates without querying a SUT or target label."""
    if count < 1:
        raise ValueError("count must be positive")
    stage = "neighbor" if reference is not None else "random"
    accepted: list[ScenarioSpec] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    attempts = 0
    while len(accepted) < count and attempts < count * max_attempt_factor:
        attempts += 1
        if reference is None:
            gap = rng.uniform(*seed.gap_bounds)
            speed = rng.uniform(*seed.relative_speed_bounds)
            mode = str(rng.choice(seed.allowed_modes))
        else:
            gap = np.clip(
                rng.uniform(reference.initial_gap - 5 * gap_step, reference.initial_gap + 5 * gap_step),
                *seed.gap_bounds,
            )
            speed = np.clip(
                rng.uniform(reference.relative_speed - 5 * speed_step, reference.relative_speed + 5 * speed_step),
                *seed.relative_speed_bounds,
            )
            # Mode is categorical: uniformly draw from the legal set, while a
            # deterministic fraction retains the reference mode.
            mode = reference.mode if rng.random() < 0.5 else str(rng.choice(seed.allowed_modes))
        candidate = _make_candidate(gap, speed, mode)
        valid, reason = seed.validate(candidate)
        if not valid:
            rejected.append({"scenario_id": candidate.scenario_id, "reason": reason})
            continue
        if candidate.scenario_id in seen:
            duplicates += 1
            continue
        seen.add(candidate.scenario_id)
        accepted.append(candidate)
    return CandidateBatch(tuple(accepted), attempts, tuple(rejected), duplicates, stage, reference.scenario_id if reference else None)

