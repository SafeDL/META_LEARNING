"""Strict metadata for one transferable scenario-mining task."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping


TASK_SCHEMA = "transferable_scenario_task"
SPLITS = frozenset({"train", "validation", "test"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
INTERACTION_LOGICAL_PARAMETER_NAMES = (
    "adversary_distance_to_conflict_m",
    "sut_distance_to_conflict_m",
    "adversary_initial_speed_mps",
    "sut_initial_speed_mps",
    "maneuver_onset_progress",
)
CUTIN_LOGICAL_PARAMETER_NAMES = (
    "cutin_gap_at_start_m",
    "sut_initial_speed_mps",
    "relative_speed_mps",
    "cutin_start_progress",
    "cutin_start_time_s",
    "lane_change_length_m",
)
# Kept as the public default for Merge and Roundabout callers.  Cut-in must
# explicitly request its own schema rather than silently borrowing conflict
# point coordinates.
LOGICAL_PARAMETER_NAMES = INTERACTION_LOGICAL_PARAMETER_NAMES


def logical_parameter_names(family: str) -> tuple[str, ...]:
    if family == "cutin":
        return CUTIN_LOGICAL_PARAMETER_NAMES
    if family in {"merge", "roundabout"}:
        return INTERACTION_LOGICAL_PARAMETER_NAMES
    raise ValueError(f"unsupported scenario family: {family!r}")


@dataclass(frozen=True)
class ScenarioMiningTaskSpec:
    task_id: str
    sut_ref: str
    functional_scenario: str
    geometry_id: str
    geometry_hash: str
    geometry_seed: int
    adapter_id: str
    interaction_schema_id: str
    sut_split: str
    geometry_split: str
    logical_domain_id: str
    logical_domain_bounds: Mapping[str, tuple[float, float]]
    logical_parameter_mask: tuple[bool, ...]
    logical_split: str
    functional_split: str = "train"
    schema: str = TASK_SCHEMA

    @property
    def map_hash(self) -> str:
        return self.geometry_hash

    def validate(self) -> None:
        if self.schema != TASK_SCHEMA:
            raise ValueError(f"unsupported task schema: {self.schema!r}")
        if not all(str(value) for value in (
            self.task_id, self.sut_ref, self.functional_scenario, self.geometry_id,
            self.adapter_id, self.interaction_schema_id, self.logical_domain_id,
        )):
            raise ValueError("task identifiers must be non-empty")
        if not _SHA256.fullmatch(self.geometry_hash):
            raise ValueError("geometry_hash must be a lowercase SHA-256 digest")
        if not isinstance(self.geometry_seed, int):
            raise ValueError("geometry_seed must be an integer")
        if any(value not in SPLITS for value in (
            self.sut_split, self.geometry_split, self.logical_split, self.functional_split,
        )):
            raise ValueError("task split axes must be train, validation, or test")
        names = logical_parameter_names(self.functional_scenario)
        if tuple(self.logical_domain_bounds) != names:
            raise ValueError("logical domain bounds must use the canonical parameter order")
        if len(self.logical_parameter_mask) != len(names):
            raise ValueError("logical parameter mask must match the canonical parameter count")
        if not all(isinstance(value, bool) for value in self.logical_parameter_mask):
            raise ValueError("logical parameter mask must contain booleans")
        for name in names:
            bounds = self.logical_domain_bounds[name]
            if len(bounds) != 2:
                raise ValueError("logical domain bounds must contain named intervals")
            lower, upper = (float(value) for value in bounds)
            if not lower < upper or lower < -1.0 or upper > 1.0:
                raise ValueError("logical domain bounds must lie in normalized [-1, 1]")

    @property
    def active_logical_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, active in zip(
                logical_parameter_names(self.functional_scenario), self.logical_parameter_mask
            )
            if active
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScenarioMiningTaskSpec":
        fields = set(cls.__dataclass_fields__)
        unknown, missing = set(value) - fields, fields - set(value)
        if unknown or missing:
            raise ValueError(f"task fields mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}")
        payload = dict(value)
        payload["logical_parameter_mask"] = tuple(payload["logical_parameter_mask"])
        task = cls(**payload)
        task.validate()
        return task
