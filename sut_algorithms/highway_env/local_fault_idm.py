"""Legacy local IDM fault variants supported by the shared SUT adapter."""

from __future__ import annotations

from dataclasses import dataclass

from sut_algorithms.highway_env.idm_profiles import ProfiledIDMVehicle


FAULTS = ("merge_blind06", "merge_brake2", "slow_front_brake2")


@dataclass(frozen=True)
class LocalFault:
    name: str

    def __post_init__(self) -> None:
        if self.name not in FAULTS:
            raise ValueError(self.name)


class LocalFaultIDMVehicle(ProfiledIDMVehicle):
    def __init__(self, *args, fault: LocalFault, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fault = fault
        self.merge_seen_at: float | None = None
        self.fault_active_steps = 0

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None) -> float:
        fault = self.fault.name
        if fault.startswith("merge_") and front_vehicle is not None:
            lateral_offset = abs(float(front_vehicle.position[1] - ego_vehicle.position[1]))
            if self.merge_seen_at is None and lateral_offset >= .25:
                self.merge_seen_at = self.elapsed
        if self.merge_seen_at is not None and fault.startswith("merge_"):
            duration = .60 if fault == "merge_blind06" else .80
            if self.elapsed - self.merge_seen_at < duration:
                self.fault_active_steps += 1
                if fault == "merge_blind06":
                    return super().acceleration(ego_vehicle, None, rear_vehicle)
                baseline = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)
                return max(baseline, -2.0)
        baseline = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)
        if fault == "slow_front_brake2" and front_vehicle is not None \
                and float(front_vehicle.speed) < 18.0:
            self.fault_active_steps += 1
            return max(baseline, -2.0)
        return baseline
