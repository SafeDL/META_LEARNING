"""Heterogeneous IDM-style controllers used as systems under test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highway_env import utils
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


@dataclass(frozen=True)
class IDMProfile:
    """A controller profile with a distinct safety failure mode."""

    name: str
    time_wanted: float
    max_brake: float
    reaction_delay: float
    desired_gap: float
    comfort_acceleration: float = 3.0
    target_speed: float = 27.0


PROFILES = (
    IDMProfile("SUT-A", 2.4, 3.0, 0.0, 8.0),
    IDMProfile("SUT-B", 0.8, 9.0, 0.0, 3.0),
    IDMProfile("SUT-C", 1.3, 4.5, 0.7, 5.0),
    IDMProfile("SUT-D", 1.8, 8.0, 0.0, 8.0),
    IDMProfile("SUT-E", 0.7, 10.0, 0.2, 3.0, comfort_acceleration=4.0),
    IDMProfile("SUT-F", 1.5, 5.0, 1.2, 5.0),
)
PROFILE_NAMES = tuple(profile.name for profile in PROFILES)


def get_profile(name: str) -> IDMProfile:
    """Return a named profile, rejecting accidental controller substitutions."""
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise KeyError(f"Unknown SUT profile: {name}")


class ProfiledIDMVehicle(IDMVehicle):
    """An IDM vehicle whose longitudinal behaviour is fixed by one profile."""

    def __init__(self, *args, profile: IDMProfile, **kwargs) -> None:
        super().__init__(*args, enable_lane_change=False, **kwargs)
        self.profile = profile
        self.elapsed = 0.0
        self.ACC_MAX = max(self.ACC_MAX, profile.max_brake)
        self._front_seen_at: float | None = None

    def step(self, dt: float) -> None:
        self.elapsed += dt
        super().step(dt)

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """Evaluate IDM with a deterministic delayed-reaction interval."""
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0.0
        if front_vehicle is not None and self._front_seen_at is None:
            self._front_seen_at = self.elapsed
        if (
            self._front_seen_at is not None
            and self.elapsed - self._front_seen_at < self.profile.reaction_delay
        ):
            front_vehicle = None

        target_speed = min(self.profile.target_speed, ego_vehicle.lane.speed_limit)
        acceleration = self.profile.comfort_acceleration * (
            1
            - (max(ego_vehicle.speed, 0.0) / utils.not_zero(target_speed))
            ** self.DELTA
        )
        if front_vehicle is not None:
            distance = ego_vehicle.lane_distance_to(front_vehicle)
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            acceleration -= self.profile.comfort_acceleration * (
                desired_gap / utils.not_zero(distance)
            ) ** 2
        return float(np.clip(acceleration, -self.profile.max_brake, self.ACC_MAX))

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> float:
        """Compute a profile-specific IDM desired gap."""
        if front_vehicle is None:
            return self.profile.desired_gap
        relative_speed = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected
            else ego_vehicle.speed - front_vehicle.speed
        )
        braking_product = self.profile.comfort_acceleration * self.profile.max_brake
        return float(
            self.profile.desired_gap
            + ego_vehicle.speed * self.profile.time_wanted
            + ego_vehicle.speed
            * relative_speed
            / (2 * np.sqrt(braking_product))
        )
