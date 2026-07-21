"""Serializable task specifications; labels never enter learned networks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LogicalScenarioTaskSpec:
    task_id: str
    split: str
    logical_type: str
    map_config: dict[str, Any]
    conflict_spec: dict[str, Any]
    case_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LogicalScenarioTaskSpec":
        return cls(**dict(data))

    def validate(self) -> None:
        if self.logical_type not in {"on_ramp_merge", "lane_drop_merge", "bottleneck_merge", "y_merge"}:
            raise ValueError(f"unsupported logical type: {self.logical_type}")
        if not self.task_id or not self.split:
            raise ValueError("task_id and split are required")
        if not isinstance(self.case_seed, int):
            raise ValueError("case_seed must be an integer")
