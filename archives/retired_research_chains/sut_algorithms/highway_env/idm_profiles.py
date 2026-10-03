"""Heterogeneous IDM and FVDM controllers used as systems under test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


@dataclass(frozen=True)
class SUTProfile:
    """A controller profile with a distinct safety failure mode."""

    name: str
    controller: str
    time_wanted: float = 1.5
    max_brake: float = 5.0
    reaction_delay: float = 0.0
    desired_gap: float = 5.0
    comfort_acceleration: float = 3.0
    target_speed: float = 27.0
    fvdm_sensitivity: float = 0.6
    fvdm_velocity_gain: float = 0.6
    fvdm_transition_gap: float = 5.0
    emergency_ttc: float = 0.0
    emergency_brake_gain: float = 0.0
    emergency_max_brake: float = 6.0


PROFILES = (
    SUTProfile("SUT-A", "IDM", time_wanted=2.4, max_brake=3.0, desired_gap=8.0),
    SUTProfile("SUT-B", "IDM", time_wanted=0.8, max_brake=9.0, desired_gap=3.0),
    SUTProfile("SUT-C", "IDM", time_wanted=1.3, max_brake=4.5, reaction_delay=0.7,
               desired_gap=5.0),
    SUTProfile(
        "SUT-D",
        "FVDM",
        max_brake=3.5,
        desired_gap=8.0,
        fvdm_sensitivity=0.32,
        fvdm_velocity_gain=0.25,
        fvdm_transition_gap=4.0,
    ),
    SUTProfile(
        "SUT-E",
        "FVDM",
        max_brake=8.0,
        desired_gap=2.5,
        fvdm_sensitivity=0.9,
        fvdm_velocity_gain=1.2,
        fvdm_transition_gap=9.0,
    ),
    SUTProfile(
        "SUT-F",
        "FVDM",
        max_brake=4.5,
        reaction_delay=0.4,
        desired_gap=6.0,
        fvdm_sensitivity=0.6,
        fvdm_velocity_gain=0.35,
        fvdm_transition_gap=3.0,
    ),
)
PROFILE_NAMES = tuple(profile.name for profile in PROFILES)

# AdaTE's reference experiment contrasts a normal IDM surrogate with weak- and
# strong-braking FVDM surrogates.  These profiles are isolated from the legacy
# six-SUT benchmark so existing Risk Mining and DETOUR artefacts remain stable.
ADATE_SOURCE_PROFILES = (
    SUTProfile(
        "SM-Normal-IDM",
        "IDM",
        time_wanted=1.5,
        max_brake=5.0,
        desired_gap=5.0,
        comfort_acceleration=2.5,
    ),
    SUTProfile(
        "SM-Limited-FVDM",
        "FVDM",
        max_brake=1.0,
        desired_gap=4.0,
        comfort_acceleration=2.5,
        fvdm_sensitivity=0.65,
        fvdm_velocity_gain=0.55,
        fvdm_transition_gap=5.0,
    ),
    SUTProfile(
        "SM-Strong-FVDM",
        "FVDM",
        max_brake=6.0,
        desired_gap=8.0,
        comfort_acceleration=2.5,
        fvdm_sensitivity=0.45,
        fvdm_velocity_gain=1.0,
        fvdm_transition_gap=8.0,
    ),
)

ADATE_TARGET_PROFILES = (
    SUTProfile(
        "AV-Reference-IDM",
        "IDM",
        time_wanted=1.5,
        max_brake=5.0,
        desired_gap=5.0,
        comfort_acceleration=2.5,
    ),
    SUTProfile(
        "AV-Calibrated-IDM",
        "IDM",
        time_wanted=1.9,
        max_brake=7.0,
        desired_gap=7.0,
        comfort_acceleration=2.2,
        emergency_ttc=1.4,
        emergency_brake_gain=2.5,
        emergency_max_brake=8.0,
    ),
    SUTProfile(
        "AV-Predictive-Brake",
        "IDM",
        time_wanted=1.3,
        max_brake=5.5,
        reaction_delay=0.05,
        desired_gap=4.5,
        comfort_acceleration=2.8,
        emergency_ttc=3.5,
        emergency_brake_gain=8.0,
        emergency_max_brake=12.0,
    ),
)

ALL_PROFILES = PROFILES + ADATE_SOURCE_PROFILES + ADATE_TARGET_PROFILES


def get_profile(name: str) -> SUTProfile:
    """Return a named profile, rejecting accidental controller substitutions."""
    for profile in ALL_PROFILES:
        if profile.name == name:
            return profile
    raise KeyError(f"Unknown SUT profile: {name}")


class ProfiledIDMVehicle(ControlledVehicle):
    """An IDM vehicle whose longitudinal behaviour is fixed by one profile."""

    SPEED_EXPONENT = 4.0

    def __init__(self, *args, profile: SUTProfile, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.profile = profile
        self.elapsed = 0.0
        self._front_seen_at: float | None = None

    def act(self, action: dict | str = None) -> None:
        if self.crashed:
            return
        self.follow_road()
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        command = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.acceleration(self, front_vehicle, rear_vehicle),
        }
        command["steering"] = np.clip(command["steering"], -self.MAX_STEERING_ANGLE,
                                      self.MAX_STEERING_ANGLE)
        Vehicle.act(self, command)

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
        emergency_front = front_vehicle
        if front_vehicle is not None and self._front_seen_at is None:
            self._front_seen_at = self.elapsed
        if (self._front_seen_at is not None
                and self.elapsed - self._front_seen_at < self.profile.reaction_delay):
            front_vehicle = None

        target_speed = min(self.profile.target_speed, ego_vehicle.lane.speed_limit)
        acceleration = self.profile.comfort_acceleration * (
            1 - (max(ego_vehicle.speed, 0.0) / utils.not_zero(target_speed))**self.SPEED_EXPONENT)
        if front_vehicle is not None:
            distance = ego_vehicle.lane_distance_to(front_vehicle)
            desired_gap = self.desired_gap(ego_vehicle, front_vehicle)
            acceleration -= self.profile.comfort_acceleration * (desired_gap /
                                                                 utils.not_zero(distance))**2
        emergency = _emergency_brake(self.profile, ego_vehicle, emergency_front)
        acceleration -= emergency
        lower_limit = max(
            self.profile.max_brake,
            self.profile.emergency_max_brake) if emergency else self.profile.max_brake
        return float(np.clip(acceleration, -lower_limit, self.profile.comfort_acceleration))

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> float:
        """Compute a profile-specific IDM desired gap."""
        if front_vehicle is None:
            return self.profile.desired_gap
        relative_speed = (np.dot(ego_vehicle.velocity -
                                 front_vehicle.velocity, ego_vehicle.direction)
                          if projected else ego_vehicle.speed - front_vehicle.speed)
        braking_product = self.profile.comfort_acceleration * self.profile.max_brake
        return float(self.profile.desired_gap + ego_vehicle.speed * self.profile.time_wanted +
                     ego_vehicle.speed * relative_speed / (2 * np.sqrt(braking_product)))


class ProfiledFVDMVehicle(ControlledVehicle):
    """A full-velocity-difference car-following controller with fixed parameters."""
    def __init__(self, *args, profile: SUTProfile, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.profile = profile
        self.elapsed = 0.0
        self._front_seen_at: float | None = None

    def step(self, dt: float) -> None:
        self.elapsed += dt
        super().step(dt)

    def act(self, action: dict | str = None) -> None:
        if self.crashed:
            return
        self.follow_road()
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        command = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.acceleration(self, front_vehicle, rear_vehicle),
        }
        command["steering"] = np.clip(command["steering"], -self.MAX_STEERING_ANGLE,
                                      self.MAX_STEERING_ANGLE)
        Vehicle.act(self, command)

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """Apply the FVDM optimal-velocity and relative-velocity feedback terms."""
        emergency_front = front_vehicle
        if front_vehicle is not None and self._front_seen_at is None:
            self._front_seen_at = self.elapsed
        if (self._front_seen_at is not None
                and self.elapsed - self._front_seen_at < self.profile.reaction_delay):
            front_vehicle = None
        target_speed = min(self.profile.target_speed, ego_vehicle.lane.speed_limit)
        emergency = _emergency_brake(self.profile, ego_vehicle, emergency_front)
        lower_limit = max(
            self.profile.max_brake,
            self.profile.emergency_max_brake) if emergency else self.profile.max_brake
        if front_vehicle is None:
            return float(
                np.clip(
                    self.profile.fvdm_sensitivity * (target_speed - ego_vehicle.speed) - emergency,
                    -lower_limit,
                    self.profile.comfort_acceleration,
                ))
        gap = ego_vehicle.lane_distance_to(front_vehicle)
        transition = self.profile.fvdm_transition_gap
        desired = self.profile.desired_gap
        normalized_optimal_speed = (np.tanh(
            (gap - desired) / transition) + np.tanh(desired / transition)) / (
                1.0 + np.tanh(desired / transition))
        optimal_speed = target_speed * normalized_optimal_speed
        acceleration = (self.profile.fvdm_sensitivity * (optimal_speed - ego_vehicle.speed) +
                        self.profile.fvdm_velocity_gain *
                        (front_vehicle.speed - ego_vehicle.speed))
        acceleration -= emergency
        return float(np.clip(
            acceleration,
            -lower_limit,
            self.profile.comfort_acceleration,
        ))


def _emergency_brake(profile: SUTProfile, ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle | None) -> float:
    """Apply an explicit short-TTC braking safeguard when the release enables it."""
    if front_vehicle is None:
        return 0.0
    emergency = 0.0
    if profile.emergency_ttc > 0.0 and profile.emergency_brake_gain > 0.0:
        gap = ego_vehicle.lane_distance_to(front_vehicle)
        closing_speed = ego_vehicle.speed - front_vehicle.speed
        if gap > 0.0 and closing_speed > 1e-6:
            ttc = gap / closing_speed
            emergency += profile.emergency_brake_gain * max(profile.emergency_ttc - ttc, 0.0)
    return emergency


def create_profiled_vehicle(*args, profile: SUTProfile, **kwargs) -> ControlledVehicle:
    """Instantiate exactly the controller family named by a SUT profile."""
    if profile.controller == "IDM":
        return ProfiledIDMVehicle(*args, profile=profile, **kwargs)
    if profile.controller == "FVDM":
        return ProfiledFVDMVehicle(*args, profile=profile, **kwargs)
    raise ValueError(f"Unsupported controller family: {profile.controller}")
