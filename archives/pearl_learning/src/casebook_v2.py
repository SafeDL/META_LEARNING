"""Geometry-controlled initial-condition generation for casebook v2."""
from __future__ import annotations

from typing import Any, Callable, Mapping
import numpy as np

from .io import content_hash


MeasureCase = Callable[[dict[str, Any]], Mapping[str, Any]]


def solve_interaction_boundary_case(
    task: Any,
    base_case: Mapping[str, Any],
    target_gap_s: float,
    measure_case: MeasureCase,
    *,
    tolerance_s: float = 0.20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Solve both spawn positions from a target initial conflict ETA gap.

    The solver deliberately uses the instantiated route measurements rather
    than map recipe coordinates.  It chooses the midpoint of the feasible
    common SUT-arrival interval, so a construction grid has no random
    box-sampling component.
    """
    trial = dict(base_case)
    for role in ("adversary", "sut"):
        lo, hi = map(float, task.spawn_regions[role])
        trial[f"{role}_spawn_m"] = (lo + hi) / 2.0
    measured_reference = dict(measure_case(trial))
    intervals: dict[str, tuple[float, float, float]] = {}
    for role in ("adversary", "sut"):
        lo, hi = map(float, task.spawn_regions[role])
        speed = float(base_case[f"{role}_initial_speed_mps"])
        if speed <= 0.0:
            raise ValueError(f"{role} initial speed must be positive")
        offset = float(trial[f"{role}_spawn_m"]) + float(measured_reference[f"{role}_distance_m"])
        intervals[role] = ((offset - hi) / speed, (offset - lo) / speed, offset)
    sut_lo, sut_hi, sut_offset = intervals["sut"]
    adv_lo, adv_hi, adv_offset = intervals["adversary"]
    common_lo = max(sut_lo, adv_lo - float(target_gap_s), 0.0)
    common_hi = min(sut_hi, adv_hi - float(target_gap_s))
    if common_lo > common_hi:
        raise ValueError("target ETA gap is infeasible in the frozen spawn regions")
    sut_time = (common_lo + common_hi) / 2.0
    adv_time = sut_time + float(target_gap_s)
    candidate = dict(trial)
    for role, arrival_time, offset in (
        ("sut", sut_time, sut_offset), ("adversary", adv_time, adv_offset),
    ):
        speed = float(base_case[f"{role}_initial_speed_mps"])
        lo, hi = map(float, task.spawn_regions[role])
        spawn = offset - speed * arrival_time
        if not lo <= spawn <= hi:
            raise ValueError(f"solved {role} spawn lies outside the frozen spawn region")
        candidate[f"{role}_spawn_m"] = float(spawn)
    candidate["adversary_speed_mps"] = float(candidate["adversary_initial_speed_mps"])
    measured = dict(measure_case(candidate))
    actual_gap = float(measured["adversary_time_s"]) - float(measured["sut_time_s"])
    if abs(actual_gap - float(target_gap_s)) > tolerance_s:
        raise ValueError("real geometry did not realize the requested interaction ETA gap")
    if min(float(measured["adversary_signed_distance_m"]), float(measured["sut_signed_distance_m"])) <= 0.0:
        raise ValueError("a vehicle already passed its conflict point")
    if bool(measured.get("initial_target_overlap", False)):
        raise ValueError("initial target vehicles physically overlap")
    return candidate, measured


def solve_adversary_spawn(
    task: Any,
    base_case: Mapping[str, Any],
    target_gap_s: float,
    measure_case: MeasureCase,
    *,
    tolerance_s: float = 0.20,
    adversary_speed_range: tuple[float, float] | list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Solve one spawn using true route/conflict geometry, then verify it."""
    lo, hi = map(float, task.spawn_regions["adversary"])
    trial = dict(base_case)
    trial["adversary_spawn_m"] = (lo + hi) / 2.0
    reference = dict(measure_case(trial))
    sut_time = float(reference["sut_time_s"])
    desired_time = sut_time + float(target_gap_s)
    if desired_time <= 0.0:
        raise ValueError("target arrival gap puts the adversary beyond the conflict point")
    conflict_offset = float(trial["adversary_spawn_m"]) + float(reference["adversary_distance_m"])
    adversary_speed = float(base_case["adversary_initial_speed_mps"])
    if adversary_speed_range is not None:
        feasible_low = max(float(adversary_speed_range[0]), (conflict_offset - hi) / desired_time)
        feasible_high = min(float(adversary_speed_range[1]), (conflict_offset - lo) / desired_time)
        if feasible_low > feasible_high:
            raise ValueError("no adversary speed can realize the target gap in the spawn region")
        generator = rng or np.random.default_rng(0)
        adversary_speed = float(generator.uniform(feasible_low, feasible_high))
    desired_distance = adversary_speed * desired_time
    # Route distance changes one-for-one with spawn longitude on the explicit
    # initial lane. A final real-environment reset below verifies the result.
    spawn = conflict_offset - desired_distance
    if not lo <= spawn <= hi:
        raise ValueError("solved adversary spawn lies outside the frozen spawn region")
    candidate = {
        **trial,
        "adversary_spawn_m": spawn,
        "adversary_initial_speed_mps": adversary_speed,
        "adversary_speed_mps": adversary_speed,
    }
    measured = dict(measure_case(candidate))
    actual_gap = float(measured["adversary_time_s"]) - float(measured["sut_time_s"])
    if abs(actual_gap - float(target_gap_s)) > tolerance_s:
        raise ValueError("real geometry did not realize the requested arrival gap")
    if min(float(measured["adversary_signed_distance_m"]), float(measured["sut_signed_distance_m"])) <= 0.0:
        raise ValueError("a vehicle already passed its conflict point")
    if "initial_target_overlap" in measured:
        if bool(measured["initial_target_overlap"]):
            raise ValueError("initial target vehicles physically overlap")
    elif float(measured["initial_pair_distance_m"]) < 6.0:
        # Backward-compatible fallback for geometry-only test probes. Runtime
        # MetaDrive measurement always supplies the pairwise contact result.
        raise ValueError("initial target vehicles overlap or are too close")
    return candidate, measured


def generate_controlled_cases(
    task: Any,
    split: str,
    count: int,
    config: Mapping[str, Any],
    *,
    arrival_gap_threshold_s: float,
    calibration_hash: str,
    measure_case: MeasureCase,
    reachable_count: int,
    attempt_offset: int = 0,
) -> list[dict[str, Any]]:
    """Generate deterministic cases with controlled signed arrival gaps."""
    if not 0 <= reachable_count <= count:
        raise ValueError("reachable_count must lie in [0, count]")
    seed = int(content_hash({
        "task": task.task_id, "split": split, "calibration_hash": calibration_hash,
        "attempt_offset": int(attempt_offset), "schema": "logical_merge_casebook_v2",
    })[:16], 16)
    rng = np.random.default_rng(seed)
    sampling = dict(config.get("case_sampling", {}))
    sut_range = sampling.get("sut_initial_speed_mps", (10.0, 14.0))
    adv_range = sampling.get("adversary_initial_speed_mps", (10.0, 17.0))
    difficulties = ["heuristic_reachable"] * reachable_count + ["harder"] * (count - reachable_count)
    cases: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    attempts = 0
    while len(cases) < count and attempts < max(200, count * 80):
        difficulty = difficulties[len(cases)]
        attempts += 1
        case_seed = int(rng.integers(1, 2**31 - 1))
        if case_seed in used_seeds:
            continue
        used_seeds.add(case_seed)
        multiplier = float(rng.uniform(1.5, 2.15) if difficulty == "heuristic_reachable" else rng.uniform(2.2, 3.0))
        target_gap = float(rng.choice([-1.0, 1.0]) * multiplier * arrival_gap_threshold_s)
        adversary_speed = float(rng.uniform(*adv_range))
        base = {
            "case_id": f"{task.task_id}_{split}_{len(cases):03d}",
            "case_seed": case_seed,
            "sut_initial_speed_mps": float(rng.uniform(*sut_range)),
            "adversary_initial_speed_mps": adversary_speed,
            "adversary_speed_mps": adversary_speed,
            "sut_spawn_m": float(rng.uniform(*task.spawn_regions["sut"])),
        }
        try:
            candidate, measured = solve_adversary_spawn(
                task, base, target_gap, measure_case,
                adversary_speed_range=adv_range,
                rng=rng,
            )
        except (RuntimeError, ValueError):
            continue
        case = {
            **candidate,
            "target_initial_arrival_gap_s": target_gap,
            "actual_initial_arrival_gap_s": float(measured["adversary_time_s"]) - float(measured["sut_time_s"]),
            "initial_relative_speed_mps": float(measured["initial_relative_speed_mps"]),
            "adversary_initial_conflict_distance_m": float(measured["adversary_distance_m"]),
            "sut_initial_conflict_distance_m": float(measured["sut_distance_m"]),
            "initial_pair_distance_m": float(measured["initial_pair_distance_m"]),
            "initial_target_overlap": bool(measured.get("initial_target_overlap", False)),
            "difficulty_class": difficulty,
            "heuristic_longitudinal_reachable": True,
            "calibration_hash": calibration_hash,
        }
        cases.append(case)
    if len(cases) != count:
        raise RuntimeError(f"only generated {len(cases)}/{count} controlled cases for {task.task_id}/{split}")
    return cases
