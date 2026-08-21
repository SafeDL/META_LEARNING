"""Black-box SUT contract; profiles are runner metadata, never learned input."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ControllerProfile:
    profile_id: str
    adapter_name: str
    target_speed_mps: float
    enable_lane_change: bool
    distance_wanted_m: float
    time_headway_s: float

    def validate(self) -> None:
        if not self.profile_id or not self.adapter_name:
            raise ValueError("controller profile id and adapter name are required")
        if self.target_speed_mps <= 0.0 or self.distance_wanted_m <= 0.0 or self.time_headway_s <= 0.0:
            raise ValueError("controller profile parameters must be positive")


class SUTAdapter(Protocol):
    name: str

    def reset(self, env: Any, task: Any, config: Mapping[str, Any], seed: int) -> None: ...
    def attach(self, env: Any, vehicle: Any, profile: ControllerProfile, seed: int) -> Any: ...
    def observe(self, env: Any, vehicle: Any) -> Mapping[str, Any]: ...
    def step(self, observation: Mapping[str, Any]) -> Any: ...
    def metadata(self, profile: ControllerProfile) -> dict[str, Any]: ...
