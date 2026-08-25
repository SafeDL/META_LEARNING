from __future__ import annotations

from dataclasses import dataclass, field
from .base import ControllerProfile, SUTAdapter


@dataclass
class SUTRegistry:
    adapters: dict[str, SUTAdapter] = field(default_factory=dict)
    profiles: dict[str, ControllerProfile] = field(default_factory=dict)

    def register_adapter(self, adapter: SUTAdapter) -> None:
        name = str(adapter.name)
        if not name or name in self.adapters:
            raise ValueError(f"duplicate or empty SUT adapter: {name!r}")
        self.adapters[name] = adapter

    def register_profile(self, profile: ControllerProfile) -> None:
        profile.validate()
        if profile.profile_id in self.profiles:
            raise ValueError(f"duplicate SUT profile: {profile.profile_id}")
        if profile.adapter_name not in self.adapters:
            raise ValueError(f"profile references unknown adapter: {profile.adapter_name}")
        self.profiles[profile.profile_id] = profile

    def create(self, profile_id: str) -> tuple[SUTAdapter, ControllerProfile]:
        try:
            profile = self.profiles[profile_id]
            return self.adapters[profile.adapter_name], profile
        except KeyError as error:
            raise ValueError(f"unknown SUT profile: {profile_id}") from error


def default_registry() -> SUTRegistry:
    from .idm import IDMSUTAdapter

    registry = SUTRegistry()
    registry.register_adapter(IDMSUTAdapter())
    profiles = (
        ControllerProfile("idm_cautious", "idm", 7.0, False, 24.0, 3.0, 1.20, -2.0),
        ControllerProfile("idm_defensive", "idm", 10.0, False, 16.0, 2.2, 1.10, -3.0),
        ControllerProfile("idm_normal", "idm", 13.0, True, 9.0, 1.2, 1.20, -4.5),
        ControllerProfile("idm_assertive", "idm", 18.0, True, 3.0, 0.45, 1.80, -8.0),
        ControllerProfile("idm_fast_small_gap", "idm", 16.0, True, 5.0, 0.75, 1.50, -6.5),
        ControllerProfile("idm_late_response", "idm", 14.0, True, 7.0, 0.35, 0.75, -8.5),
    )
    for profile in profiles:
        registry.register_profile(profile)
    return registry
