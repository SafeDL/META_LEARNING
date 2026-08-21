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
    from .rule_based import RuleBasedSUTAdapter

    registry = SUTRegistry()
    registry.register_adapter(IDMSUTAdapter())
    registry.register_adapter(RuleBasedSUTAdapter())
    profiles = (
        ControllerProfile("idm_cautious", "idm", 8.0, False, 16.0, 2.5),
        ControllerProfile("idm_defensive", "idm", 10.0, False, 13.0, 1.8),
        ControllerProfile("idm_normal", "idm", 12.0, True, 10.0, 1.4),
        ControllerProfile("idm_assertive", "idm", 14.0, True, 7.0, 1.0),
        ControllerProfile("idm_fast_small_gap", "idm", 16.0, True, 5.0, 0.8),
        ControllerProfile("idm_late_response", "idm", 15.0, True, 7.0, 0.6),
    )
    for profile in profiles:
        registry.register_profile(profile)
    return registry
