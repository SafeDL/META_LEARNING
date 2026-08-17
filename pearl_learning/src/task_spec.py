"""Versioned, strict task contract for audited logical-merge experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping


TASK_SCHEMA = "logical_merge_task"
SUPPORTED_LOGICAL_TYPES = frozenset({"on_ramp_merge", "lane_drop_merge", "bottleneck_merge", "y_merge"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _sha(value: Any, name: str) -> str:
    value = str(value)
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_route(route: Mapping[str, Any], name: str) -> None:
    route = _mapping(route, name)
    if not str(route.get("route_id", "")):
        raise ValueError(f"{name}.route_id is required")
    lanes = route.get("lane_sequence")
    if not isinstance(lanes, list) or not lanes:
        raise ValueError(f"{name}.lane_sequence must be a non-empty list")
    for lane in lanes:
        if not isinstance(lane, (list, tuple)) or len(lane) != 3 or not isinstance(lane[2], int):
            raise ValueError(f"{name}.lane_sequence contains an invalid lane index")
    initial = route.get("initial_lane")
    if initial != lanes[0]:
        raise ValueError(f"{name}.initial_lane must equal the first route lane")


@dataclass(frozen=True)
class LogicalScenarioTaskSpec:
    """A frozen task whose geometry is explicit and independently verifiable."""

    schema: str
    task_id: str
    split: str
    family: str
    logical_type: str
    geometry_id: str
    map_config: dict[str, Any]
    adversary_route: dict[str, Any]
    sut_route: dict[str, Any]
    spawn_regions: dict[str, list[float]]
    priority_spec: dict[str, Any]
    conflict_spec: dict[str, Any]
    case_seed: int
    map_hash: str
    adversary_route_hash: str
    sut_route_hash: str
    conflict_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LogicalScenarioTaskSpec":
        payload = dict(data)
        expected = set(cls.__dataclass_fields__)
        unknown, missing = set(payload) - expected, expected - set(payload)
        if unknown or missing:
            raise ValueError(f"{TASK_SCHEMA} fields mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}")
        if payload.get("schema") != TASK_SCHEMA:
            raise ValueError(
                f"unsupported task schema {payload.get('schema')!r}; "
                f"only {TASK_SCHEMA} is executable"
            )
        task = cls(**payload)
        task.validate()
        return task

    def validate(self) -> None:
        if self.schema != TASK_SCHEMA:
            raise ValueError(f"task schema must be {TASK_SCHEMA}")
        if not self.task_id or not self.split or not self.family or not self.geometry_id:
            raise ValueError("task_id, split, family, and geometry_id are required")
        if self.logical_type not in SUPPORTED_LOGICAL_TYPES:
            raise ValueError(f"unsupported logical type: {self.logical_type}")
        if not isinstance(self.case_seed, int):
            raise ValueError("case_seed must be an integer")
        _mapping(self.map_config, "map_config")
        _validate_route(self.adversary_route, "adversary_route")
        _validate_route(self.sut_route, "sut_route")
        regions = _mapping(self.spawn_regions, "spawn_regions")
        for role in ("adversary", "sut"):
            region = regions.get(role)
            if not isinstance(region, list) or len(region) != 2 or not all(isinstance(x, (int, float)) for x in region):
                raise ValueError(f"spawn_regions.{role} must be [min_m, max_m]")
            if not 0.0 <= float(region[0]) < float(region[1]):
                raise ValueError(f"spawn_regions.{role} must be an increasing non-negative interval")
        priority = _mapping(self.priority_spec, "priority_spec")
        if not isinstance(priority.get("sut_has_priority"), bool):
            raise ValueError("priority_spec.sut_has_priority must be boolean")
        contact_rule = str(priority.get("target_contact_speed_relation", "any"))
        if contact_rule not in {"any", "adversary_faster", "sut_faster"}:
            raise ValueError("priority_spec.target_contact_speed_relation must be any, adversary_faster, or sut_faster")
        margin = float(priority.get("target_contact_speed_margin_mps", 0.0))
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("priority_spec.target_contact_speed_margin_mps must be a finite non-negative number")
        entry_order = str(priority.get("target_contact_entry_order", "any"))
        if entry_order not in {"any", "adversary_first", "sut_first"}:
            raise ValueError("priority_spec.target_contact_entry_order must be any, adversary_first, or sut_first")
        if entry_order != "any" and priority.get("target_contact_entry_order_semantics") not in {
            "pre_step_arrival_time", "continuous_route_entry_interpolation",
        }:
            raise ValueError("arrival-order tasks must declare a supported frozen entry-order semantic")
        conflict = _mapping(self.conflict_spec, "conflict_spec")
        if float(conflict.get("conflict_radius_m", 0.0)) <= 0.0:
            raise ValueError("conflict_spec.conflict_radius_m must be positive")
        for value, name in (
            (self.map_hash, "map_hash"),
            (self.adversary_route_hash, "adversary_route_hash"),
            (self.sut_route_hash, "sut_route_hash"),
            (self.conflict_hash, "conflict_hash"),
        ):
            _sha(value, name)
