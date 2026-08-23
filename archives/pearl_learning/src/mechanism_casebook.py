"""Matched physical cases for the mechanism-identifiability gates.

This module is deliberately separate from :mod:`casebook_v2`.  Benchmark v2
normalizes difficulty against a task-specific calibrated threshold; the
mechanism gate must instead expose the same absolute arrival-gap and relative
speed conditions to each task.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .casebook_v2 import solve_adversary_spawn
from .io import content_hash


MECHANISM_CASEBOOK_PURPOSE = "mechanism_identifiability"
MECHANISM_CASEBOOK_SCHEMA = "logical_merge_mechanism_casebook_v3"

# A small deterministic subset of the prescribed absolute grid.  It covers
# adversary-first, near-simultaneous, and SUT-first arrivals at all three
# relative-speed levels without using any calibrated risk threshold.
DEFAULT_MATCHED_CONDITIONS = (
    # The signs are physically consistent: a faster adversary can be made
    # earlier, while a slower adversary can be made later, within the frozen
    # 2--8 m spawn regions of both selected bottleneck tasks.
    (-0.8, 2.0, 0.25), (-0.8, 2.0, 0.75),
    (-0.4, 2.0, 0.50), (-0.4, 0.0, 0.50),
    (0.0, 0.0, 0.50), (0.4, 0.0, 0.50),
    (0.8, -2.0, 0.25, 12.0), (0.8, -2.0, 0.75, 13.0),
)

# A second, deliberately narrow mechanism-only profile for an opposite-order
# task pair.  Every case begins close enough to the order boundary that both
# accelerate and brake are plausible choices; the profile is not used for the
# benchmark Casebook v2 and does not depend on calibrated thresholds.
ORDER_BOUNDARY_CONDITIONS = (
    (-0.10, 0.0, 0.10, 11.0), (-0.10, 0.0, 0.90, 11.0),
    (-0.05, 0.0, 0.30, 12.0), (-0.05, 0.0, 0.70, 12.0),
    (0.00, 0.0, 0.20, 12.0), (0.00, 0.0, 0.80, 12.0),
    (0.05, 0.0, 0.25, 13.0), (0.10, 0.0, 0.75, 13.0),
)

# A separately versioned, feasibility-screened subset of order-boundary v1.
# The omitted two conditions were observed under the fixed-policy Gate 1
# audit to have no shared collision-free control window for the two opposite
# order objectives.  This is construction data only: it is never a test/OOD
# filter and its provenance is written into the casebook manifest.
ORDER_BOUNDARY_SCREENED_V1_CONDITIONS = (
    (-0.05, 0.0, 0.30, 12.0, "mechanism_grid_02"),
    (-0.05, 0.0, 0.70, 12.0, "mechanism_grid_03"),
    (0.00, 0.0, 0.20, 12.0, "mechanism_grid_04"),
    (0.00, 0.0, 0.80, 12.0, "mechanism_grid_05"),
    (0.05, 0.0, 0.25, 13.0, "mechanism_grid_06"),
    (0.10, 0.0, 0.75, 13.0, "mechanism_grid_07"),
)
MECHANISM_CASE_PROFILES = {
    "absolute_grid": DEFAULT_MATCHED_CONDITIONS,
    "order_boundary": ORDER_BOUNDARY_CONDITIONS,
    "order_boundary_screened_v1": ORDER_BOUNDARY_SCREENED_V1_CONDITIONS,
}

# A three-way few-shot split profile for Gate 3.  The train group is exactly
# the Gate-1-feasibility-screened subset; support and query extend the same
# near-simultaneous order-boundary family with previously unseen signed gaps,
# so every split exposes matched physics while condition ids remain disjoint
# across splits.  The two conditions excluded by the v1 screening are not
# reintroduced anywhere.
ORDER_BOUNDARY_FEWSHOT_V1_CONDITIONS = {
    "train_pool": (
        (-0.05, 0.0, 0.30, 12.0, "mechanism_grid_02"),
        (-0.05, 0.0, 0.70, 12.0, "mechanism_grid_03"),
        (0.00, 0.0, 0.20, 12.0, "mechanism_grid_04"),
        (0.00, 0.0, 0.80, 12.0, "mechanism_grid_05"),
        (0.05, 0.0, 0.25, 13.0, "mechanism_grid_06"),
        (0.10, 0.0, 0.75, 13.0, "mechanism_grid_07"),
    ),
    "validation_support": (
        (-0.04, 0.0, 0.35, 12.5, "mechanism_grid_support_00"),
        (0.04, 0.0, 0.65, 12.5, "mechanism_grid_support_01"),
        (-0.02, 0.0, 0.45, 12.5, "mechanism_grid_support_02"),
        (0.02, 0.0, 0.55, 12.5, "mechanism_grid_support_03"),
    ),
    "validation_query": (
        (-0.06, 0.0, 0.40, 12.5, "mechanism_grid_query_00"),
        (0.06, 0.0, 0.60, 12.5, "mechanism_grid_query_01"),
        (-0.08, 0.0, 0.50, 12.5, "mechanism_grid_query_02"),
        (0.08, 0.0, 0.70, 12.5, "mechanism_grid_query_03"),
    ),
}
# A separately versioned few-shot profile whose validation_query group was
# screened with the two Gate 1B single-task SAC policies (Gate 3 query oracle
# audit).  The unscreened query conditions of order_boundary_fewshot_v1 gave
# the sut_first task no achievable strict VCSR and a negative diagonal
# advantage, so Stage C could not be judged.  The frozen query group is the
# deterministic greedy selection from ORDER_BOUNDARY_QUERY_CANDIDATES_V1;
# provenance lives in the gate3 config's mechanism.query_case_selection and in
# results/.../gate_3_query_screening/gate3_query_candidate_screening.json.
ORDER_BOUNDARY_FEWSHOT_SCREENED_V1_CONDITIONS = {
    "train_pool": ORDER_BOUNDARY_FEWSHOT_V1_CONDITIONS["train_pool"],
    "validation_support": ORDER_BOUNDARY_FEWSHOT_V1_CONDITIONS["validation_support"],
    "validation_query": (
        (0.00, 0.0, 0.35, 12.0, "mechanism_grid_query_candidate_04"),
        (0.00, 0.0, 0.65, 12.0, "mechanism_grid_query_candidate_05"),
        (0.10, 0.0, 0.40, 13.0, "mechanism_grid_query_candidate_02"),
        (-0.05, 0.0, 0.40, 12.0, "mechanism_grid_query_candidate_06"),
    ),
}
FEWSHOT_MECHANISM_PROFILES = {
    "order_boundary_fewshot_v1": ORDER_BOUNDARY_FEWSHOT_V1_CONDITIONS,
    "order_boundary_fewshot_screened_v1": ORDER_BOUNDARY_FEWSHOT_SCREENED_V1_CONDITIONS,
}

# Construction data for the Gate 3 query oracle screening: candidate query
# conditions in the same near-boundary family as the screened train pool.
# The two Gate 1B single-task SACs are used as the screening instrument; only
# conditions where both tasks show own-policy strict VCSR and at least one
# cross-policy failure are frozen back into the few-shot profile.
ORDER_BOUNDARY_QUERY_CANDIDATES_V1 = (
    (0.05, 0.0, 0.30, 12.0, "mechanism_grid_query_candidate_00"),
    (0.05, 0.0, 0.55, 12.0, "mechanism_grid_query_candidate_01"),
    (0.10, 0.0, 0.40, 13.0, "mechanism_grid_query_candidate_02"),
    (0.10, 0.0, 0.60, 13.0, "mechanism_grid_query_candidate_03"),
    (0.00, 0.0, 0.35, 12.0, "mechanism_grid_query_candidate_04"),
    (0.00, 0.0, 0.65, 12.0, "mechanism_grid_query_candidate_05"),
    (-0.05, 0.0, 0.40, 12.0, "mechanism_grid_query_candidate_06"),
    (-0.05, 0.0, 0.60, 12.0, "mechanism_grid_query_candidate_07"),
)

MeasureCase = Callable[[dict[str, Any]], Mapping[str, Any]]


def _condition_rows(profile: str, rows: Sequence[tuple]) -> list[dict[str, float | str]]:
    result = []
    for index, row in enumerate(rows):
        gap, relative_speed, sut_spawn_fraction = row[:3]
        sut_speed = row[3] if len(row) >= 4 else 12.0
        source_condition_id = str(row[4]) if len(row) >= 5 else f"mechanism_grid_{index:02d}"
        result.append({
            "matched_condition_id": source_condition_id,
            "mechanism_case_profile": str(profile),
            "target_initial_arrival_gap_s": float(gap),
            "target_initial_relative_speed_mps": float(relative_speed),
            "target_sut_spawn_fraction": float(sut_spawn_fraction),
            "target_sut_initial_speed_mps": float(sut_speed),
        })
    return result


def matched_conditions(count: int, *, profile: str = "absolute_grid") -> list[dict[str, float | str]]:
    """Return a deterministic prefix of a named absolute mechanism profile."""
    try:
        conditions = MECHANISM_CASE_PROFILES[str(profile)]
    except KeyError as error:
        raise ValueError(f"unsupported mechanism case profile: {profile!r}") from error
    if not 1 <= int(count) <= len(conditions):
        raise ValueError(f"matched case count must lie in [1, {len(conditions)}]")
    return _condition_rows(str(profile), conditions[:count])


def matched_split_conditions(profile: str) -> dict[str, list[dict[str, float | str]]]:
    """Return per-split condition rows for a few-shot mechanism profile."""
    try:
        split_rows = FEWSHOT_MECHANISM_PROFILES[str(profile)]
    except KeyError as error:
        raise ValueError(f"unsupported few-shot mechanism case profile: {profile!r}") from error
    result = {}
    for split, rows in split_rows.items():
        result[split] = _condition_rows(str(profile), rows)
    # Construction-level guard: one condition must never appear in two splits.
    ids = [row["matched_condition_id"] for rows in result.values() for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"few-shot profile {profile} reuses a condition across splits")
    return result


MATCHED_PHYSICAL_FIELDS = (
    "case_seed",
    "sut_spawn_m",
    "adversary_spawn_m",
    "sut_initial_speed_mps",
    "adversary_initial_speed_mps",
    "actual_initial_arrival_gap_s",
    "initial_relative_speed_mps",
    "adversary_initial_conflict_distance_m",
    "sut_initial_conflict_distance_m",
)


def matched_case_seed(condition_id: str) -> int:
    """Use one exogenous simulator seed for a matched condition.

    In a physical task comparison, changing ``case_seed`` changes MetaDrive's
    scenario initialization and the IDM seed.  It must therefore be derived
    from the condition, never from the task id.  Task-local ``case_id`` still
    provides replay namespace isolation.
    """
    return int(content_hash({
        "schema": MECHANISM_CASEBOOK_SCHEMA,
        "condition_id": condition_id,
    })[:16], 16) % (2**31 - 2) + 1


def validate_matched_mechanism_cases(cases_by_task: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Reject a Gate-1 pair unless every condition has identical physics."""
    if len(cases_by_task) != 2:
        raise ValueError("the mechanism audit requires exactly two task case sets")
    by_task = {
        str(task_id): {str(row["matched_condition_id"]): row for row in cases}
        for task_id, cases in cases_by_task.items()
    }
    if any(len(rows) != len(cases_by_task[task_id]) for task_id, rows in by_task.items()):
        raise ValueError("duplicate matched_condition_id in a mechanism case set")
    task_ids = list(by_task)
    left, right = by_task[task_ids[0]], by_task[task_ids[1]]
    if set(left) != set(right):
        raise ValueError("mechanism tasks do not expose the same matched conditions")
    for condition_id in sorted(left):
        for field in MATCHED_PHYSICAL_FIELDS:
            first, second = left[condition_id].get(field), right[condition_id].get(field)
            if isinstance(first, (int, float)) and isinstance(second, (int, float)):
                if not np.isclose(float(first), float(second), rtol=0.0, atol=1e-6):
                    raise ValueError(
                        f"mechanism condition {condition_id} differs in physical field {field}"
                    )
            elif first != second:
                raise ValueError(
                    f"mechanism condition {condition_id} differs in physical field {field}"
                )


def validate_mechanism_split_disjointness(
    cases_by_task: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> None:
    """Require matched physics across tasks and disjoint conditions across splits.

    The generic validator forbids any case-seed reuse across tasks, but matched
    mechanism tasks must deliberately share the exogenous physical seed for the
    same condition.  This validator therefore only forbids condition reuse
    *across splits*; within one split the two tasks must expose exactly the
    same conditions with identical physical fields.
    """
    if len(cases_by_task) != 2:
        raise ValueError("the mechanism split validator requires exactly two task case sets")
    split_sets = {frozenset(book) for book in cases_by_task.values()}
    if len(split_sets) != 1:
        raise ValueError("mechanism tasks must expose the same casebook splits")
    seen: dict[str, set[str]] = {}
    for split in sorted(next(iter(split_sets))):
        per_task: dict[str, Sequence[Mapping[str, Any]]] = {}
        condition_ids: set[str] = set()
        for task_id, book in cases_by_task.items():
            rows = list(book.get(split, []))
            ids = [str(row["matched_condition_id"]) for row in rows]
            if len(set(ids)) != len(rows):
                raise ValueError(f"duplicate matched_condition_id in {task_id}/{split}")
            for condition_id in ids:
                if any(condition_id in previous for previous in seen.values()):
                    raise ValueError(
                        f"mechanism split leakage: condition {condition_id} reused across splits"
                    )
            condition_ids.update(ids)
            per_task[task_id] = rows
        seen[split] = condition_ids
        validate_matched_mechanism_cases(per_task)


def generate_mechanism_cases(
    task: Any,
    config: Mapping[str, Any],
    *,
    measure_case: MeasureCase,
    count: int,
    split: str = "train_pool",
    conditions: Sequence[Mapping[str, float | str]] | None = None,
) -> list[dict[str, Any]]:
    """Generate a task-local realization of shared absolute conditions.

    Speeds and signed initial arrival gaps are identical for all selected
    tasks.  The geometry-specific adversary spawn is solved against a real
    environment reset, then recorded as provenance; no threshold from the v2
    calibration enters generation or filtering.
    """
    if split not in {"train_pool", "validation_support", "validation_query", "test_support", "test_query"}:
        raise ValueError(f"unsupported case split: {split!r}")
    requested = list(conditions or matched_conditions(count))
    if len(requested) != int(count):
        raise ValueError("condition count does not match requested mechanism cases")
    mechanism = dict(config.get("mechanism", {}))
    default_sut_speed = float(mechanism.get("sut_initial_speed_mps", 12.0))
    sampling = dict(config.get("case_sampling", {}))
    adv_range = tuple(map(float, sampling.get("adversary_initial_speed_mps", (10.0, 17.0))))
    sut_region = tuple(map(float, task.spawn_regions["sut"]))
    result: list[dict[str, Any]] = []
    for index, condition in enumerate(requested):
        condition_id = str(condition["matched_condition_id"])
        target_gap = float(condition["target_initial_arrival_gap_s"])
        relative_speed = float(condition["target_initial_relative_speed_mps"])
        sut_speed = float(condition.get("target_sut_initial_speed_mps", default_sut_speed))
        primary_fraction = float(condition.get("target_sut_spawn_fraction", 0.50))
        fractions = (primary_fraction,) + tuple(
            value for value in (0.50, 0.25, 0.75, 0.10, 0.90, 0.00, 1.00)
            if value != primary_fraction
        )
        # Explore a fixed, task-independent normalized-coordinate grid.
        # This never changes the target gap or speed to rescue a geometry.
        sut_spawns = [
            float(sut_region[0] + fraction * (sut_region[1] - sut_region[0]))
            for fraction in fractions
        ]
        adversary_speed = sut_speed + relative_speed
        if not adv_range[0] <= adversary_speed <= adv_range[1]:
            raise ValueError(
                f"shared relative speed {relative_speed} makes adversary speed {adversary_speed} "
                f"outside configured range {adv_range}"
            )
        candidate = None
        measured = None
        for sut_spawn in sut_spawns:
            base = {
                "case_id": f"{task.task_id}_{split}_{index:03d}",
                "case_seed": matched_case_seed(condition_id),
                "sut_initial_speed_mps": sut_speed,
                "adversary_initial_speed_mps": adversary_speed,
                "adversary_speed_mps": adversary_speed,
                "sut_spawn_m": sut_spawn,
            }
            try:
                candidate, measured = solve_adversary_spawn(
                    task, base, target_gap, measure_case,
                    # Deliberately omit adversary_speed_range: the solver
                    # then uses base['adversary_initial_speed_mps'] exactly.
                    # Passing an equal lower/upper range is vulnerable to a
                    # floating-point feasibility roundoff in the v2 helper.
                )
                break
            except (RuntimeError, ValueError):
                continue
        if candidate is None or measured is None:
            raise RuntimeError(
                f"task {task.task_id} cannot realize shared mechanism condition {condition_id}; "
                "do not replace it with a task-specific threshold"
            )
        result.append({
            **candidate,
            "matched_condition_id": condition_id,
            "target_initial_arrival_gap_s": target_gap,
            "actual_initial_arrival_gap_s": float(measured["adversary_time_s"]) - float(measured["sut_time_s"]),
            "target_initial_relative_speed_mps": relative_speed,
            "initial_relative_speed_mps": float(measured["initial_relative_speed_mps"]),
            "adversary_initial_conflict_distance_m": float(measured["adversary_distance_m"]),
            "sut_initial_conflict_distance_m": float(measured["sut_distance_m"]),
            "initial_pair_distance_m": float(measured["initial_pair_distance_m"]),
            "initial_target_overlap": bool(measured.get("initial_target_overlap", False)),
            "mechanism_casebook_purpose": MECHANISM_CASEBOOK_PURPOSE,
            "task_specific_risk_normalization": False,
        })
    return result
