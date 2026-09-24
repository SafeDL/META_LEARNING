"""Deterministic Cut-in and longitudinal interaction modes for highway-env."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from sut_algorithms.highway_env.idm_profiles import SUTProfile, create_profiled_vehicle


@dataclass(frozen=True)
class CutInScenario:
    """Anchor coordinates plus a discrete interaction mechanism."""

    initial_gap: float
    relative_speed: float
    mode: str = "fast_intrusion"
    timing: float | None = None
    intensity: float | None = None

    def __post_init__(self) -> None:
        for name in ("timing", "intensity"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def as_array(self) -> np.ndarray:
        values = [self.initial_gap, self.relative_speed]
        if self.timing is not None and self.intensity is not None:
            values.extend((self.timing, self.intensity))
        return np.asarray(values, dtype=float)


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
    """Fixed lateral and longitudinal behaviour for one functional scenario."""

    initial_lane: int
    target_lane: int
    cutin_start: float | None = None
    cutin_duration: float = 1.5
    brake_start: float | None = None
    brake_duration: float = 0.0
    braking_deceleration: float = 0.0


class ScheduledCutInVehicle(ControlledVehicle):
    """A lead vehicle with an optional lane-change and braking schedule."""

    _TIME_EPSILON = 1e-9

    def __init__(
        self,
        *args,
        cutin_start: float | None,
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
        if (
            self.cutin_start is not None
            and self.elapsed + self._TIME_EPSILON >= self.cutin_start
        ):
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
    """One ego SUT and one scheduled lead vehicle on a straight two-lane road."""

    EGO_SPEED = 25.0
    CUTIN_START = 1.0
    FAST_CUTIN_DURATION = 0.45
    DEFAULT_CUTIN_DURATION = 1.5
    BRAKE_DURATION = 1.0
    BRAKING_DECELERATION = 4.5
    NEAR_MISS_CLEARANCE = 1.0
    FAST_INTRUSION = "fast_intrusion"
    CUTIN_BRAKING = "cutin_braking"
    LEAD_BRAKING = "lead_braking"
    STOP_AND_GO = "stop_and_go"
    SLOW_LEAD_FOLLOWING = "slow_lead_following"
    PASSING_CUTIN = "passing_cutin"

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
        self._leading_vehicle: ScheduledCutInVehicle | None = None
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
        self._leading_vehicle = None
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
        ego = self._create_ego_vehicle(ego_lane, ego_road_lane)
        schedule = self._mode_schedule()
        lead_speed = self.EGO_SPEED + self.scenario.relative_speed
        if (
            self.scenario.mode == self.SLOW_LEAD_FOLLOWING
            and self.scenario.intensity is not None
        ):
            lead_speed -= 2.5 * self.scenario.intensity
        lead_longitudinal = 60.0 + self.scenario.initial_gap
        if schedule.initial_lane == 1:
            lead_longitudinal -= self.scenario.relative_speed * self.CUTIN_START
        lead_lane = ego_lane if schedule.initial_lane == 0 else adjacent_lane
        target_lane = ego_lane if schedule.target_lane == 0 else adjacent_lane
        lead_road_lane = self.road.network.get_lane(lead_lane)
        lead = ScheduledCutInVehicle(
            self.road,
            lead_road_lane.position(lead_longitudinal, 0.0),
            heading=lead_road_lane.heading_at(lead_longitudinal),
            speed=lead_speed,
            target_lane_index=lead_lane,
            target_speed=lead_speed,
            cutin_start=schedule.cutin_start,
            cutin_duration=schedule.cutin_duration,
            cutin_target_lane_index=target_lane,
            brake_start=schedule.brake_start,
            brake_duration=schedule.brake_duration,
            braking_deceleration=schedule.braking_deceleration,
        )
        self.vehicle = ego
        self._cutin_vehicle = lead
        vehicles = [ego, lead]
        if self.scenario.mode == self.PASSING_CUTIN:
            leading_longitudinal = lead_longitudinal + 24.0
            leading_speed = max(lead_speed - 3.0, 8.0)
            leading = ScheduledCutInVehicle(
                self.road,
                lead_road_lane.position(leading_longitudinal, 0.0),
                heading=lead_road_lane.heading_at(leading_longitudinal),
                speed=leading_speed,
                target_lane_index=lead_lane,
                target_speed=leading_speed,
                cutin_start=None,
                cutin_duration=self.DEFAULT_CUTIN_DURATION,
                cutin_target_lane_index=lead_lane,
            )
            self._leading_vehicle = leading
            vehicles.append(leading)
        self.road.vehicles = vehicles

    def _create_ego_vehicle(self, ego_lane, ego_road_lane) -> ControlledVehicle:
        """Create the default profile-controlled ego vehicle."""
        return create_profiled_vehicle(
            self.road,
            ego_road_lane.position(60.0, 0.0),
            heading=ego_road_lane.heading_at(60.0),
            speed=self.EGO_SPEED,
            target_lane_index=ego_lane,
            target_speed=self.profile.target_speed,
            profile=self.profile,
        )

    def _mode_schedule(self) -> _ModeSchedule:
        """Map a mode and optional normalized controls to a physical schedule."""
        timing = self.scenario.timing
        intensity = self.scenario.intensity
        controlled = timing is not None and intensity is not None
        if self.scenario.mode == self.FAST_INTRUSION:
            if controlled:
                return _ModeSchedule(
                    1,
                    0,
                    0.6 + 0.8 * timing,
                    0.25 + 0.55 * (1.0 - intensity),
                )
            return _ModeSchedule(1, 0, self.CUTIN_START, self.FAST_CUTIN_DURATION)
        if self.scenario.mode == self.CUTIN_BRAKING:
            if controlled:
                cutin_duration = 0.8 + timing
                return _ModeSchedule(
                    1,
                    0,
                    self.CUTIN_START,
                    cutin_duration,
                    brake_start=self.CUTIN_START + cutin_duration,
                    brake_duration=0.6 + 0.8 * intensity,
                    braking_deceleration=3.0 + 4.0 * intensity,
                )
            return _ModeSchedule(
                1, 0, self.CUTIN_START, self.config["cutin_duration"],
                brake_start=self.CUTIN_START + self.config["cutin_duration"],
                brake_duration=self.BRAKE_DURATION,
                braking_deceleration=self.BRAKING_DECELERATION,
            )
        if self.scenario.mode == self.LEAD_BRAKING:
            if controlled:
                return _ModeSchedule(
                    0,
                    0,
                    brake_start=0.5 + 1.5 * timing,
                    brake_duration=0.6 + intensity,
                    braking_deceleration=4.0 + 5.0 * intensity,
                )
            return _ModeSchedule(
                0, 0, brake_start=1.0, brake_duration=self.BRAKE_DURATION,
                braking_deceleration=6.5,
            )
        if self.scenario.mode == self.STOP_AND_GO:
            if controlled:
                return _ModeSchedule(
                    0,
                    0,
                    brake_start=0.5 + timing,
                    brake_duration=1.2 + 2.0 * intensity,
                    braking_deceleration=2.5 + 3.0 * intensity,
                )
            return _ModeSchedule(
                0, 0, brake_start=1.0, brake_duration=2.0,
                braking_deceleration=3.5,
            )
        if self.scenario.mode == self.SLOW_LEAD_FOLLOWING:
            return _ModeSchedule(0, 0)
        if self.scenario.mode == self.PASSING_CUTIN:
            return _ModeSchedule(1, 0, self.CUTIN_START, self.DEFAULT_CUTIN_DURATION)
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
        # Euclidean centre distance minus half-lengths gives zero for cars
        # passing side by side in adjacent lanes. Use their rotated outlines.
        clearance = Polygon(lead.polygon()).distance(Polygon(self.vehicle.polygon()))
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
        return bool(
            self.vehicle.crashed
            or self._cutin_vehicle.crashed
            or (self._leading_vehicle is not None and self._leading_vehicle.crashed)
        )

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def episode_result(self) -> EpisodeResult:
        """Convert the complete episode trace into the prescribed response value."""
        collision = self._is_terminated()
        near_miss = not collision and (
            self._min_ttc < 1.5 or self._min_distance < self.NEAR_MISS_CLEARANCE
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
