"""Strict metadata for one transferable scenario-mining task."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping


TASK_SCHEMA = "transferable_scenario_task"
SPLITS = frozenset({"train", "validation", "test"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
        if not self.logical_domain_bounds:
            raise ValueError("logical_domain_bounds must be non-empty")
        for name, bounds in self.logical_domain_bounds.items():
            if not str(name) or len(bounds) != 2:
                raise ValueError("logical domain bounds must contain named intervals")
            lower, upper = (float(value) for value in bounds)
            if not lower <= upper or lower < -1.0 or upper > 1.0:
                raise ValueError("logical domain bounds must lie in normalized [-1, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScenarioMiningTaskSpec":
        fields = set(cls.__dataclass_fields__)
        unknown, missing = set(value) - fields, fields - set(value)
        if unknown or missing:
            raise ValueError(f"task fields mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}")
        task = cls(**dict(value))
        task.validate()
        return task
