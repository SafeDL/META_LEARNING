"""Deterministic Cut-in interaction modes for highway-env."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from mvr.highway.sut.idm_profiles import SUTProfile, create_profiled_vehicle


@dataclass(frozen=True)
class CutInScenario:
    """Anchor coordinates plus a discrete interaction mechanism."""

    initial_gap: float
    relative_speed: float
    mode: str = "single"

    def as_array(self) -> np.ndarray:
        return np.array([self.initial_gap, self.relative_speed], dtype=float)


@dataclass(frozen=True)
class EpisodeResult:
    """Safety response stored for every (SUT, anchor) episode."""

    collision: bool
    near_miss: bool
    min_ttc: float
    min_distance: float
    completed: bool
    terminated: bool
    vulnerability: float


@dataclass(frozen=True)
class LeadVehicleTrace:
    """Per-simulation-step low-level state of the scheduled lead vehicle."""

    time: np.ndarray
    acceleration: np.ndarray
    speed: np.ndarray
    lateral_position: np.ndarray


@dataclass(frozen=True)
class _ModeSchedule:
    """Fixed timing and braking parameters for one interaction mode."""

    cutin_duration: float
    brake_start: float | None = None
    brake_duration: float = 0.0
    braking_deceleration: float = 0.0


class ScheduledCutInVehicle(ControlledVehicle):
    """A lead vehicle that executes one scheduled lane change into the ego lane."""

    _TIME_EPSILON = 1e-9

    def __init__(
        self,
        *args,
        cutin_start: float,
        cutin_duration: float,
        cutin_target_lane_index: tuple[str, str, int],
        brake_start: float | None = None,
        brake_duration: float = 0.0,
        braking_deceleration: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cutin_start = cutin_start
        self.cutin_duration = cutin_duration
        self.cutin_target_lane_index = cutin_target_lane_index
        self.brake_start = brake_start
        self.brake_duration = brake_duration
        self.braking_deceleration = braking_deceleration
        self.elapsed = 0.0
        self.KP_LATERAL = 2.0 / cutin_duration

    def act(self, action: dict | str = None) -> None:
        """Apply the scheduled low-level action without re-running controller logic."""
        self.follow_road()
        if self.elapsed + self._TIME_EPSILON >= self.cutin_start:
            self.target_lane_index = self.cutin_target_lane_index
        acceleration = self.speed_control(self.target_speed)
        if self._is_braking():
            acceleration = -self.braking_deceleration
        action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": acceleration,
        }
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        # ControlledVehicle.act() regenerates acceleration from target_speed and
        # would overwrite the scheduled braking command. Vehicle.act() stores
        # this already constrained low-level action for the next dynamics step.
        Vehicle.act(self, action)

    def _is_braking(self) -> bool:
        """Return whether the current simulation interval is in the brake window."""
        if self.brake_start is None:
            return False
        return (
            self.brake_start - self._TIME_EPSILON <= self.elapsed
            < self.brake_start + self.brake_duration - self._TIME_EPSILON
        )

    def step(self, dt: float) -> None:
        self.elapsed += dt
        super().step(dt)


class CutInEnv(AbstractEnv):
    """One ego SUT and one adjacent lead vehicle on a straight two-lane road."""

    EGO_SPEED = 25.0
    CUTIN_START = 1.0
    FAST_CUTIN_DURATION = 0.45
    DEFAULT_CUTIN_DURATION = 1.5
    BRAKE_DURATION = 1.0
    BRAKING_DECELERATION = 4.5
    FAST_INTRUSION = "fast_intrusion"
    CUTIN_BRAKING = "cutin_braking"

    def __init__(
        self,
        profile: SUTProfile,
        scenario: CutInScenario,
        render_mode: str | None = None,
    ) -> None:
        self.profile = profile
        self.scenario = scenario
        self._min_ttc = np.inf
        self._min_distance = np.inf
        self._cutin_vehicle: ScheduledCutInVehicle | None = None
        self._lead_trace: list[tuple[float, float, float, float]] = []
        super().__init__(render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {"type": "DiscreteMetaAction", "longitudinal": False},
                "lanes_count": 2,
                "duration": 7.0,
                "simulation_frequency": 20,
                "policy_frequency": 5,
                "road_length": 400.0,
                "cutin_duration": cls.DEFAULT_CUTIN_DURATION,
            }
        )
        return config

    def _reset(self) -> None:
        self._min_ttc = np.inf
        self._min_distance = np.inf
        self._lead_trace = []
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"],
                start=0,
                length=self.config["road_length"],
                speed_limit=35,
            ),
            np_random=self.np_random,
        )

    def _create_vehicles(self) -> None:
        ego_lane = ("0", "1", 0)
        adjacent_lane = ("0", "1", 1)
        ego_road_lane = self.road.network.get_lane(ego_lane)
        ego_position = ego_road_lane.position(60.0, 0.0)
        ego = create_profiled_vehicle(
            self.road,
            ego_position,
            heading=ego_road_lane.heading_at(60.0),
            speed=self.EGO_SPEED,
            target_lane_index=ego_lane,
            target_speed=self.profile.target_speed,
            profile=self.profile,
        )
        lead_speed = self.EGO_SPEED + self.scenario.relative_speed
        lead_longitudinal = (
            60.0
            + self.scenario.initial_gap
            - self.scenario.relative_speed * self.CUTIN_START
        )
        adjacent_road_lane = self.road.network.get_lane(adjacent_lane)
        schedule = self._mode_schedule()
        lead = ScheduledCutInVehicle(
            self.road,
            adjacent_road_lane.position(lead_longitudinal, 0.0),
            heading=adjacent_road_lane.heading_at(lead_longitudinal),
            speed=lead_speed,
            target_lane_index=adjacent_lane,
            target_speed=lead_speed,
            cutin_start=self.CUTIN_START,
            cutin_duration=schedule.cutin_duration,
            cutin_target_lane_index=ego_lane,
            brake_start=schedule.brake_start,
            brake_duration=schedule.brake_duration,
            braking_deceleration=schedule.braking_deceleration,
        )
        self.vehicle = ego
        self._cutin_vehicle = lead
        self.road.vehicles = [ego, lead]

    def _mode_schedule(self) -> _ModeSchedule:
        """Map a declared discrete mode to its fixed interaction mechanism."""
        if self.scenario.mode == "single":
            return _ModeSchedule(cutin_duration=self.config["cutin_duration"])
        if self.scenario.mode == self.FAST_INTRUSION:
            return _ModeSchedule(cutin_duration=self.FAST_CUTIN_DURATION)
        if self.scenario.mode == self.CUTIN_BRAKING:
            return _ModeSchedule(
                cutin_duration=self.config["cutin_duration"],
                brake_start=self.CUTIN_START + self.config["cutin_duration"],
                brake_duration=self.BRAKE_DURATION,
                braking_deceleration=self.BRAKING_DECELERATION,
            )
        raise ValueError(f"Unsupported Cut-in interaction mode: {self.scenario.mode}")

    def _simulate(self, action: int | None = None) -> None:
        frames = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )
        simulation_dt = 1 / self.config["simulation_frequency"]
        for _ in range(frames):
            self.road.act()
            self.road.step(simulation_dt)
            self.steps += 1
            self._record_lead_trace()
            self._update_safety_metrics()

    def _record_lead_trace(self) -> None:
        lead = self._cutin_vehicle
        if lead is None:
            return
        self._lead_trace.append(
            (
                float(lead.elapsed),
                float(lead.action["acceleration"]),
                float(lead.speed),
                float(lead.position[1]),
            )
        )

    def _update_safety_metrics(self) -> None:
        lead = self._cutin_vehicle
        if lead is None:
            return
        center_distance = np.linalg.norm(lead.position - self.vehicle.position)
        clearance = max(0.0, center_distance - (lead.LENGTH + self.vehicle.LENGTH) / 2)
        self._min_distance = min(self._min_distance, float(clearance))
        longitudinal_gap = lead.position[0] - self.vehicle.position[0]
        lateral_distance = abs(lead.position[1] - self.vehicle.position[1])
        closing_speed = self.vehicle.speed - lead.speed
        if longitudinal_gap > 0 and lateral_distance < self.vehicle.WIDTH:
            if closing_speed > 1e-6:
                self._min_ttc = min(self._min_ttc, longitudinal_gap / closing_speed)

    def _reward(self, action: int) -> float:
        return float(not self.vehicle.crashed)

    def _is_terminated(self) -> bool:
        return bool(self.vehicle.crashed or self._cutin_vehicle.crashed)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def episode_result(self) -> EpisodeResult:
        """Convert the complete episode trace into the prescribed response value."""
        collision = self._is_terminated()
        near_miss = not collision and (
            self._min_ttc < 1.5 or self._min_distance < 2.0
        )
        ttc = float(self._min_ttc)
        risk_scale = 0.0 if np.isinf(ttc) else float(np.exp(-ttc / 3.0))
        if collision:
            vulnerability = 1.0
        elif near_miss:
            vulnerability = 0.75 + 0.25 * risk_scale
        else:
            vulnerability = 0.75 * risk_scale
        return EpisodeResult(
            collision=collision,
            near_miss=near_miss,
            min_ttc=ttc,
            min_distance=float(self._min_distance),
            completed=not collision and self.time >= self.config["duration"],
            terminated=collision,
            vulnerability=float(vulnerability),
        )

    def lead_vehicle_trace(self) -> LeadVehicleTrace:
        """Return the complete lead trace after an episode has run."""
        if not self._lead_trace:
            raise RuntimeError("No lead-vehicle trace is available before simulation")
        values = np.asarray(self._lead_trace, dtype=float)
        return LeadVehicleTrace(
            time=values[:, 0],
            acceleration=values[:, 1],
            speed=values[:, 2],
            lateral_position=values[:, 3],
        )


def run_cutin_episode(
    profile: SUTProfile, scenario: CutInScenario, seed: int = 0
) -> EpisodeResult:
    """Run one deterministic scenario episode and return only its safety response."""
    result, _ = _run_cutin_episode(profile, scenario, seed, capture_trace=False)
    return result


def run_cutin_episode_with_trace(
    profile: SUTProfile, scenario: CutInScenario, seed: int = 0
) -> tuple[EpisodeResult, LeadVehicleTrace]:
    """Run an episode and retain every low-level lead-vehicle state sample."""
    result, trace = _run_cutin_episode(profile, scenario, seed, capture_trace=True)
    assert trace is not None
    return result, trace


def _run_cutin_episode(
    profile: SUTProfile,
    scenario: CutInScenario,
    seed: int,
    capture_trace: bool,
) -> tuple[EpisodeResult, LeadVehicleTrace | None]:
    """Run one episode and optionally return its complete lead-vehicle trace."""
    env = CutInEnv(profile, scenario)
    try:
        env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(1)
        trace = env.lead_vehicle_trace() if capture_trace else None
        return env.episode_result(), trace
    finally:
        env.close()
