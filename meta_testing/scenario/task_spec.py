"""Generic, strict task metadata.  It deliberately excludes model inputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping


TASK_SCHEMA = "meta_test_task_v1"
SUPPORTED_SPLITS = frozenset({"meta_train", "meta_validation", "meta_test"})
SUPPORTED_FAMILIES = frozenset({"merge", "cutin", "roundabout"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class MetaTestTaskSpec:
    task_id: str
    split: str
    sut_ref: str
    scenario_family: str
    map_id: str
    map_hash: str
    scenario_template_id: str
    parameter_space_id: str
    seed: int
    schema: str = TASK_SCHEMA

    def validate(self) -> None:
        if self.schema != TASK_SCHEMA:
            raise ValueError(f"unsupported task schema: {self.schema!r}")
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {self.split!r}")
        if self.scenario_family not in SUPPORTED_FAMILIES:
            raise ValueError(f"unsupported scenario family: {self.scenario_family!r}")
        if not all(str(value) for value in (self.task_id, self.sut_ref, self.map_id, self.scenario_template_id, self.parameter_space_id)):
            raise ValueError("task identifiers must be non-empty")
        if not _SHA256.fullmatch(self.map_hash):
            raise ValueError("map_hash must be a lowercase SHA-256 digest")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MetaTestTaskSpec":
        fields = set(cls.__dataclass_fields__)
        unknown, missing = set(value) - fields, fields - set(value)
        if unknown or missing:
            raise ValueError(f"task fields mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}")
        task = cls(**dict(value))
        task.validate()
        return task
