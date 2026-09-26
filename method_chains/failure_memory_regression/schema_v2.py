"""Typed records and stable identities for FBRT failure memory v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "fbrt-memory-v2"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BuildSpec:
    build_id: str
    family: str
    parent_build_id: str | None
    adapter_kind: str
    policy_name: str
    control_hz: float
    profile: dict | None = None
    checkpoint_sha256: str | None = None
    mutation: dict | None = None

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


@dataclass
class EpisodeRecord:
    execution_id: str
    scenario_id: str
    build_id: str
    build_fingerprint: str
    scenario_fingerprint: str
    template_id: str
    context_id: str
    scenario: dict
    simulator_seed: int | None
    execution_contract_version: str
    completed: bool | None
    ego_collision: bool | None
    inconclusive: bool
    collision_partner_role: str | None = None
    collision_time_s: float | None = None
    min_ttc: float | None = None
    min_clearance: float | None = None
    public_signature: dict = field(default_factory=dict)
    visibility: str = "historical"
    episode_cost: int = 0
    trajectory_path: str | None = None
    source_file: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PatternCard:
    pattern_id: str
    template_id: str
    context_id: str
    semantic_key: tuple
    center: list[float]
    radius: float
    failure_record_ids: list[str]
    pass_contrast_record_ids: list[str]
    boundary_edges: list[tuple[str, str]]
    occurrence_by_build: dict
    created_in_session: str
    parent_pattern_ids: list[str] = field(default_factory=list)
    evidence_status: str = "observed_region"
    observed_partner_roles: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        item = asdict(self)
        item["semantic_key"] = list(self.semantic_key)
        return item


@dataclass
class Session:
    session_id: str
    task_id: str
    target_build_id: str
    mode: str
    history_snapshot_before: str
    candidate_ids: list[str]
    queried_execution_ids: list[str] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    created_pattern_ids: list[str] = field(default_factory=list)
    history_snapshot_after: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)
