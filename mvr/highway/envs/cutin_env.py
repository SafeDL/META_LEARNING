"""A deterministic two-dimensional cut-in scenario for highway-env."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle

from mvr.highway.sut.idm_profiles import SUTProfile, create_profiled_vehicle


@dataclass(frozen=True)
class CutInScenario:
    """Anchor coordinates: gap at cut-in start and lead-minus-ego speed."""

    initial_gap: float
    relative_speed: float

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


class ScheduledCutInVehicle(ControlledVehicle):
    """A lead vehicle that executes one scheduled lane change into the ego lane."""

    def __init__(
        self,
        *args,
        cutin_start: float,
        cutin_duration: float,
        cutin_target_lane_index: tuple[str, str, int],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cutin_start = cutin_start
        self.cutin_duration = cutin_duration
        self.cutin_target_lane_index = cutin_target_lane_index
        self.elapsed = 0.0
        self.KP_LATERAL = 2.0 / cutin_duration

    def act(self, action: dict | str = None) -> None:
        if self.elapsed >= self.cutin_start:
            self.target_lane_index = self.cutin_target_lane_index
        action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.speed_control(self.target_speed),
        }
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        super().act(action)

    def step(self, dt: float) -> None:
        self.elapsed += dt
        super().step(dt)


class CutInEnv(AbstractEnv):
    """One ego SUT and one adjacent lead vehicle on a straight two-lane road."""

    EGO_SPEED = 25.0
    CUTIN_START = 1.0

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
                "cutin_duration": 1.5,
            }
        )
        return config

    def _reset(self) -> None:
        self._min_ttc = np.inf
        self._min_distance = np.inf
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
        lead = ScheduledCutInVehicle(
            self.road,
            adjacent_road_lane.position(lead_longitudinal, 0.0),
            heading=adjacent_road_lane.heading_at(lead_longitudinal),
            speed=lead_speed,
            target_lane_index=adjacent_lane,
            target_speed=lead_speed,
            cutin_start=self.CUTIN_START,
            cutin_duration=self.config["cutin_duration"],
            cutin_target_lane_index=ego_lane,
        )
        self.vehicle = ego
        self._cutin_vehicle = lead
        self.road.vehicles = [ego, lead]

    def _simulate(self, action: int | None = None) -> None:
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for _ in range(frames):
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1
            self._update_safety_metrics()

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
        vulnerability = (
            1.0 if collision else 0.75 + 0.25 * risk_scale if near_miss else 0.75 * risk_scale
        )
        return EpisodeResult(
            collision=collision,
            near_miss=near_miss,
            min_ttc=ttc,
            min_distance=float(self._min_distance),
            completed=not collision and self.time >= self.config["duration"],
            terminated=collision,
            vulnerability=float(vulnerability),
        )


def run_cutin_episode(
    profile: SUTProfile, scenario: CutInScenario, seed: int = 0
) -> EpisodeResult:
    """Run one deterministic scenario episode and return only its safety response."""
    env = CutInEnv(profile, scenario)
    env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(1)
    result = env.episode_result()
    env.close()
    return result
