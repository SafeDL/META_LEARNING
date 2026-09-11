"""Aligned DIVA source banks and source-domain Sobol sampling."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import qmc

from ..scenario.catalog import mvr_parameter_spaces, valid_cutin_initial_state
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.task_spec import ScenarioMiningTaskSpec, logical_parameter_names
from .types import DIVA_SCHEMA, DivaCutInDesign, DivaObservation


@dataclass(frozen=True)
class SourceBank:
    """Paired source matrix with formal outcomes separate from learning responses."""

    source_refs: tuple[str, ...]
    designs: tuple[DivaCutInDesign, ...]
    formal_scores: np.ndarray
    responses: np.ndarray
    eligible: np.ndarray
    observations: tuple[DivaObservation, ...]

    def __post_init__(self) -> None:
        shape = (len(self.source_refs), len(self.designs))
        if self.formal_scores.shape != shape or self.responses.shape != shape:
            raise ValueError("source matrices must align with sources and designs")
        if self.eligible.shape != shape:
            raise ValueError("source eligibility matrix must align with responses")

    @property
    def design_ids(self) -> tuple[str, ...]:
        return tuple(design.design_id for design in self.designs)

    @property
    def features(self) -> np.ndarray:
        return np.asarray(
            [design.feature_vector() for design in self.designs], dtype=np.float64
        )

    @classmethod
    def from_observations(
        cls, observations: Iterable[DivaObservation], source_refs: tuple[str, ...]
    ) -> "SourceBank":
        rows = tuple(observations)
        if not rows:
            raise ValueError("source bank requires observations")
        if len(set(source_refs)) != len(source_refs):
            raise ValueError("source refs must be unique")
        designs = {row.design.design_id: row.design for row in rows}
        ordered = tuple(designs[key] for key in sorted(designs))
        source_index = {source: index for index, source in enumerate(source_refs)}
        design_index = {design.design_id: index for index, design in enumerate(ordered)}
        grouped: dict[tuple[int, int], list[DivaObservation]] = {}
        for row in rows:
            if row.sut_ref not in source_index:
                raise ValueError("observation source is outside the declared source split")
            key = source_index[row.sut_ref], design_index[row.design.design_id]
            grouped.setdefault(key, []).append(row)
        shape = len(source_refs), len(ordered)
        formal = np.full(shape, np.nan, dtype=np.float64)
        responses = np.full(shape, np.nan, dtype=np.float64)
        eligible = np.zeros(shape, dtype=bool)
        for key, group in grouped.items():
            formal[key] = float(np.mean([row.score for row in group]))
            usable = [
                float(row.vulnerability_response)
                for row in group
                if row.posterior_eligible and row.vulnerability_response is not None
            ]
            if usable:
                responses[key] = float(np.mean(usable))
                eligible[key] = True
        return cls(source_refs, ordered, formal, responses, eligible, rows)

    def require_common_anchors(self, minimum_per_candidate: int = 16) -> None:
        common = self.eligible.all(axis=0)
        for candidate in (0, 1):
            count = sum(
                design.candidate_index == candidate
                for design, valid in zip(self.designs, common)
                if valid
            )
            if count < minimum_per_candidate:
                raise ValueError("insufficient common eligible anchors for one Cut-in candidate")

    def formal_boundary_counts(self) -> dict[str, dict[str, dict[str, int]]]:
        result: dict[str, dict[str, dict[str, int]]] = {}
        for source_index, source in enumerate(self.source_refs):
            per_candidate: dict[str, dict[str, int]] = {}
            for candidate in (0, 1):
                indexes = [
                    index
                    for index, design in enumerate(self.designs)
                    if design.candidate_index == candidate and self.eligible[source_index, index]
                ]
                values = self.formal_scores[source_index, indexes]
                per_candidate[str(candidate)] = {
                    "eligible": len(indexes),
                    "eligible_zero": int(np.sum(values == 0.0)),
                    "eligible_positive": int(np.sum(values > 0.0)),
                }
            result[source] = per_candidate
        return result

    def source_summary(self) -> dict[str, Any]:
        formal = (
            np.asarray([row.score for row in self.observations], dtype=float)
            if self.observations
            else self.formal_scores[np.isfinite(self.formal_scores)]
        )
        valid_rate = (
            float(np.mean([row.is_valid_episode for row in self.observations]))
            if self.observations else 1.0
        )
        eligible_rate = (
            float(np.mean([row.posterior_eligible for row in self.observations]))
            if self.observations else float(np.mean(self.eligible))
        )
        return {
            "source_refs": list(self.source_refs),
            "observations": len(self.observations),
            "formal_valid_rate": valid_rate,
            "posterior_eligible_rate": eligible_rate,
            "formal_event_rate": float(np.mean(formal > 0.0)),
            "formal_collision_count": int(np.sum(formal == 1.0)),
            "formal_near_miss_count": int(np.sum(formal == 0.5)),
            "common_eligible_anchors": int(self.eligible.all(axis=0).sum()),
            "formal_boundary_counts": self.formal_boundary_counts(),
        }


def retained_task(
    tasks: Iterable[ScenarioMiningTaskSpec], sut_ref: str, logical_domain_id: str
) -> ScenarioMiningTaskSpec:
    matches = [
        task
        for task in tasks
        if task.sut_ref == sut_ref
        and task.functional_scenario == "cutin"
        and task.geometry_id == "cutin-g01"
        and task.logical_domain_id == logical_domain_id
    ]
    if len(matches) != 1:
        raise ValueError("retained DIVA task selector must resolve exactly one task")
    return matches[0]


def study_task(
    task: ScenarioMiningTaskSpec,
    logical_domain_id: str,
    physical_bounds: Mapping[str, tuple[float, float]],
) -> ScenarioMiningTaskSpec:
    """Create a DIVA-only task view for frozen physical study bounds."""
    space = mvr_parameter_spaces()["cutin"]
    normalized = {}
    for name in logical_parameter_names("cutin"):
        lower, upper = physical_bounds[name]
        global_lower, global_upper = space.bounds[name]
        normalized[name] = (
            2.0 * (float(lower) - global_lower) / (global_upper - global_lower) - 1.0,
            2.0 * (float(upper) - global_lower) / (global_upper - global_lower) - 1.0,
        )
    view = replace(
        task,
        logical_domain_id=logical_domain_id,
        logical_domain_bounds=normalized,
    )
    view.validate()
    return view


def sobol_designs_from_physical_bounds(
    physical_bounds: Mapping[str, tuple[float, float]],
    anchors_per_candidate: int,
    seed: int,
) -> tuple[DivaCutInDesign, ...]:
    """Sample the frozen DIVA study domain, then encode globally for execution."""
    names = logical_parameter_names("cutin")
    if anchors_per_candidate < 1 or tuple(physical_bounds) != names:
        raise ValueError("source physical bounds must use all Cut-in parameters in order")
    bounds = np.asarray([physical_bounds[name] for name in names], dtype=np.float64)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("invalid source physical bounds")
    space = mvr_parameter_spaces()["cutin"]
    for name, (lower, upper) in zip(names, bounds):
        global_lower, global_upper = space.bounds[name]
        if lower < global_lower or upper > global_upper:
            raise ValueError("source physical bounds must lie within global Cut-in bounds")
    designs: list[DivaCutInDesign] = []
    for candidate in (0, 1):
        sampler = qmc.Sobol(5, scramble=True, seed=int(seed) + candidate)
        accepted: list[DivaCutInDesign] = []
        while len(accepted) < anchors_per_candidate:
            samples = sampler.random(max(anchors_per_candidate, 8))
            values_batch = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
            for values in values_batch:
                physical: dict[str, float | str] = {
                    "route_or_conflict_candidate": space.candidates[candidate]
                }
                physical.update({name: float(value) for name, value in zip(names, values)})
                if not valid_cutin_initial_state(
                    float(physical["ego_initial_speed_mps"]),
                    float(physical["relative_speed_mps"]),
                    float(physical["initial_gap_m"]),
                    float(physical["cutin_path_length_m"]),
                ):
                    continue
                action = space.encode(physical)
                accepted.append(
                    DivaCutInDesign(candidate, tuple(map(float, action.continuous)))
                )
                if len(accepted) == anchors_per_candidate:
                    break
        designs.extend(accepted)
    return tuple(designs)


def sobol_designs(
    task: ScenarioMiningTaskSpec, anchors_per_candidate: int, seed: int
) -> tuple[DivaCutInDesign, ...]:
    """Sample a task-local normalized domain while retaining global legality checks."""
    if task.functional_scenario != "cutin":
        raise ValueError("DIVA Sobol casebook requires Cut-in tasks")
    names = logical_parameter_names("cutin")
    space = mvr_parameter_spaces()["cutin"]
    local: dict[str, tuple[float, float]] = {}
    for name in names:
        normalized_lower, normalized_upper = task.logical_domain_bounds[name]
        global_lower, global_upper = space.bounds[name]
        local[name] = (
            global_lower + (normalized_lower + 1.0) * 0.5 * (global_upper - global_lower),
            global_lower + (normalized_upper + 1.0) * 0.5 * (global_upper - global_lower),
        )
    return sobol_designs_from_physical_bounds(local, anchors_per_candidate, seed)


def observation_from_dict(payload: Mapping[str, Any]) -> DivaObservation:
    if payload.get("schema") != DIVA_SCHEMA:
        raise ValueError("incompatible observation schema; v1 sparse labels are not accepted")
    design_payload = dict(payload["design"])
    if design_payload.pop("schema", None) != DIVA_SCHEMA:
        raise ValueError("incompatible DIVA design schema")
    design_payload.pop("design_id", None)
    row = dict(payload)
    row.pop("schema", None)
    row["design"] = DivaCutInDesign(**design_payload)
    return DivaObservation(**row)
