"""Source-bank assembly, common anchors, and source-only casebook generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import qmc

from ..scenario.catalog import mvr_parameter_spaces, valid_cutin_initial_state
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.task_spec import ScenarioMiningTaskSpec, logical_parameter_names
from .types import DIVA_SCHEMA, DivaCutInDesign, DivaObservation


@dataclass(frozen=True)
class SourceBank:
    """One aligned source-SUT by DIVA-design table, with censored labels retained."""

    source_refs: tuple[str, ...]
    designs: tuple[DivaCutInDesign, ...]
    scores: np.ndarray
    eligible: np.ndarray
    observations: tuple[DivaObservation, ...]

    @property
    def design_ids(self) -> tuple[str, ...]:
        return tuple(design.design_id for design in self.designs)

    @property
    def features(self) -> np.ndarray:
        return np.asarray([design.feature_vector() for design in self.designs], dtype=np.float64)

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
        ordered_designs = tuple(designs[key] for key in sorted(designs))
        source_index = {name: index for index, name in enumerate(source_refs)}
        design_index = {design.design_id: index for index, design in enumerate(ordered_designs)}
        grouped: dict[tuple[int, int], list[DivaObservation]] = {}
        for row in rows:
            if row.sut_ref not in source_index:
                raise ValueError("observation source is outside the declared source split")
            grouped.setdefault((source_index[row.sut_ref], design_index[row.design.design_id]), []).append(row)
        scores = np.full((len(source_refs), len(ordered_designs)), np.nan, dtype=np.float64)
        eligible = np.zeros_like(scores, dtype=bool)
        for key, group in grouped.items():
            usable = [row.score for row in group if row.posterior_eligible]
            if usable:
                scores[key] = float(np.mean(usable))
                eligible[key] = True
        return cls(source_refs, ordered_designs, scores, eligible, rows)

    def require_common_anchors(self, minimum_per_candidate: int = 16) -> None:
        common = self.eligible.all(axis=0)
        for candidate in (0, 1):
            count = sum(design.candidate_index == candidate for design, valid in zip(self.designs, common) if valid)
            if count < minimum_per_candidate:
                raise ValueError("insufficient common eligible anchors for one Cut-in candidate")

    def response_boundary_counts(self) -> dict[str, dict[str, dict[str, int]]]:
        """Eligible zero/positive counts used by the source-only viability gate."""
        result: dict[str, dict[str, dict[str, int]]] = {}
        for source_index, source in enumerate(self.source_refs):
            per_candidate: dict[str, dict[str, int]] = {}
            for candidate in (0, 1):
                indexes = [
                    index for index, design in enumerate(self.designs)
                    if design.candidate_index == candidate and self.eligible[source_index, index]
                ]
                scores = self.scores[source_index, indexes]
                per_candidate[str(candidate)] = {
                    "eligible": len(indexes),
                    "eligible_zero": int(np.sum(scores == 0.0)),
                    "eligible_positive": int(np.sum(scores > 0.0)),
                }
            result[source] = per_candidate
        return result

    def require_response_boundary(self) -> None:
        counts = self.response_boundary_counts()
        missing = [
            f"{source}/candidate-{candidate}"
            for source, candidates in counts.items()
            for candidate, values in candidates.items()
            if values["eligible_zero"] == 0 or values["eligible_positive"] == 0
        ]
        if missing:
            raise ValueError(
                "source bank lacks eligible zero and positive responses: "
                + ", ".join(missing)
            )

    def source_summary(self) -> dict[str, Any]:
        formal = np.asarray([row.score for row in self.observations], dtype=float)
        return {
            "source_refs": list(self.source_refs),
            "observations": len(self.observations),
            "formal_valid_rate": float(np.mean([row.is_valid_episode for row in self.observations])),
            "posterior_eligible_rate": float(np.mean([row.posterior_eligible for row in self.observations])),
            "event_rate": float(np.mean(formal > 0.0)),
            "common_eligible_anchors": int(self.eligible.all(axis=0).sum()),
            "response_boundary_counts": self.response_boundary_counts(),
        }


def retained_task(
    tasks: Iterable[ScenarioMiningTaskSpec], sut_ref: str, logical_domain_id: str
) -> ScenarioMiningTaskSpec:
    matches = [
        task for task in tasks
        if task.sut_ref == sut_ref
        and task.functional_scenario == "cutin"
        and task.geometry_id == "cutin-g01"
        and task.logical_domain_id == logical_domain_id
    ]
    if len(matches) != 1:
        raise ValueError("retained DIVA task selector must resolve exactly one task")
    return matches[0]


def sobol_designs(
    task: ScenarioMiningTaskSpec, anchors_per_candidate: int, seed: int
) -> tuple[DivaCutInDesign, ...]:
    if task.functional_scenario != "cutin" or anchors_per_candidate < 1:
        raise ValueError("DIVA Sobol casebook requires positive Cut-in anchors")
    names = logical_parameter_names("cutin")
    bounds = np.asarray([task.logical_domain_bounds[name] for name in names], dtype=np.float64)
    parameter_space = mvr_parameter_spaces()["cutin"]
    designs: list[DivaCutInDesign] = []
    for candidate in (0, 1):
        sampler = qmc.Sobol(5, scramble=True, seed=int(seed) + candidate)
        accepted: list[DivaCutInDesign] = []
        while len(accepted) < anchors_per_candidate:
            samples = sampler.random(max(anchors_per_candidate, 8))
            logical = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
            for scene in logical:
                action = NormalizedScenarioAction(candidate, tuple(map(float, scene)))
                physical = parameter_space.decode(action)
                if not valid_cutin_initial_state(
                    float(physical["ego_initial_speed_mps"]),
                    float(physical["relative_speed_mps"]),
                    float(physical["initial_gap_m"]),
                    float(physical["cutin_path_length_m"]),
                ):
                    continue
                accepted.append(DivaCutInDesign(candidate, tuple(map(float, scene))))
                if len(accepted) == anchors_per_candidate:
                    break
        designs.extend(accepted)
    return tuple(designs)


def observation_from_dict(payload: Mapping[str, Any]) -> DivaObservation:
    if payload.get("schema") != DIVA_SCHEMA:
        raise ValueError(
            "incompatible observations cannot be used with the physically-audited "
            "vulnerability model"
        )
    design_payload = dict(payload["design"])
    design_payload.pop("schema", None)
    design_payload.pop("design_id", None)
    payload = dict(payload)
    payload.pop("schema", None)
    payload["design"] = DivaCutInDesign(**design_payload)
    return DivaObservation(**payload)
