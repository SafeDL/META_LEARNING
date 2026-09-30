"""Stable identifiers for simulator controller builds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


def stable_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
