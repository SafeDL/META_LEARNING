"""Functional interactions using the existing highway-env road and IDM SUT."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from highway_sim_env.envs.fbrt_scenarios import FBRTScenario
from highway_sim_env.envs.fbrt_scripted_vehicle import ScriptedVehicle
from methods.core_mine.local_fault_idm import LocalFault, LocalFaultIDMVehicle
from sut_algorithms.highway_env.idm_profiles import SUTProfile, create_profiled_vehicle


class FBRTEnv(AbstractEnv):
    def __init__(self, profile: SUTProfile, scenario: FBRTScenario,
                 fault: str | None = None) -> None:
        self.profile = profile
        self.scenario = scenario
        self.fault = fault
        self.actors: dict[str, Vehicle] = {}
        self.min_clearance = float("inf")
        self.min_ttc = float("inf")
        self.collision_time_s: float | None = None
        self.collision_partner: str | None = None
        self.trace: list[dict] = []
        self.event_times: dict[str, float] = {}
        super().__init__()

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({"action": {"type": "DiscreteMetaAction", "longitudinal": False},
                       "simulation_frequency": 20, "policy_frequency": 20,
                       "road_length": 1000.0, "duration": 24.0})
        return config

    def _reset(self) -> None:
        self.actors = {}
        self.min_clearance = float("inf")
        self.min_ttc = float("inf")
        self.collision_time_s = None
        self.collision_partner = None
        self.trace = []
        self.event_times = {}
        self.config["lanes_count"] = self.scenario.lane_count
        self.config["duration"] = self.scenario.duration_s
        self.road = Road(network=RoadNetwork.straight_road_network(
            self.scenario.lane_count, start=0, length=1000.0, speed_limit=35),
            np_random=self.np_random)
        ego_lane = ("0", "1", 0)
        lane = self.road.network.get_lane(ego_lane)
        ego_position = lane.position(60.0, 0.0)
        ego_args = (self.road, ego_position)
        ego_kwargs = {"heading": lane.heading_at(60.0),
                      "speed": self.scenario.ego_speed_mps,
                      "target_lane_index": ego_lane,
                      "target_speed": self.profile.target_speed,
                      "profile": self.profile}
        if self.fault is None:
            self.vehicle = create_profiled_vehicle(*ego_args, **ego_kwargs)
        else:
            self.vehicle = LocalFaultIDMVehicle(*ego_args, fault=LocalFault(self.fault),
                                                **ego_kwargs)
        self.actors["ego"] = self.vehicle
        gap = self.scenario.initial_clearance_m
        lead_x = 60.0 + self.vehicle.LENGTH + gap
        template = self.scenario.template_id
        if template == "fbrt_cutin":
            adjacent = ("0", "1", 1)
            lead = self._scripted("lead", adjacent, lead_x, self.scenario.lead_speed_mps,
                                  event="cutin", destination_lane=ego_lane,
                                  lane_change_duration_s=self.scenario.lane_change_duration_s)
        elif template == "fbrt_lead_emergency_brake":
            lead = self._scripted("lead", ego_lane, lead_x, self.scenario.lead_speed_mps,
                                  event="brake", deceleration_mps2=self.scenario.lead_deceleration_mps2)
        elif template == "fbrt_stop_hold_go":
            lead = self._scripted("lead", ego_lane, lead_x, self.scenario.lead_speed_mps,
                                  event="stop_hold_go", deceleration_mps2=self.scenario.lead_deceleration_mps2,
                                  hold_s=2.0, restart_acceleration_mps2=1.5)
        elif template == "fbrt_cutout_static":
            adjacent = ("0", "1", 1)
            lead = self._scripted("lead", ego_lane, lead_x, self.scenario.lead_speed_mps,
                                  event="cutout", destination_lane=adjacent)
            static_x = lead_x + self.scenario.lead_speed_mps * (
                1.0 + self.scenario.static_target_ttc_s) + lead.LENGTH
            self.actors["static"] = Vehicle(self.road, lane.position(static_x, 0.0),
                                             heading=lane.heading_at(static_x), speed=0.0)
        else:
            raise ValueError(template)
        self.road.vehicles = list(self.actors.values())
        self._record_step()

    def _scripted(self, name: str, lane_index: tuple, x: float, speed: float,
                  **schedule) -> ScriptedVehicle:
        lane = self.road.network.get_lane(lane_index)
        vehicle = ScriptedVehicle(self.road, lane.position(x, 0.0),
                                  heading=lane.heading_at(x), speed=speed,
                                  target_lane_index=lane_index, target_speed=speed,
                                  **schedule)
        self.actors[name] = vehicle
        return vehicle

    def _simulate(self, action=None) -> None:
        self.road.act()
        self.road.step(0.05)
        self.steps += 1
        self._record_step()

    def _record_step(self) -> None:
        ego = self.vehicle
        stamp = round(self.steps / 20, 3)
        row = {"time_s": stamp}
        for name, vehicle in self.actors.items():
            row[name] = {"x_m": float(vehicle.position[0]),
                         "y_m": float(vehicle.position[1]),
                         "speed_mps": float(vehicle.speed),
                         "crashed": bool(vehicle.crashed)}
            if name == "ego":
                continue
            clearance = Polygon(ego.polygon()).distance(Polygon(vehicle.polygon()))
            self.min_clearance = min(self.min_clearance, float(clearance))
            dx = float(vehicle.position[0] - ego.position[0])
            lateral = abs(float(vehicle.position[1] - ego.position[1]))
            closing = float(ego.speed - vehicle.speed)
            if dx > 0 and lateral < ego.WIDTH and closing > 0:
                self.min_ttc = min(self.min_ttc, dx / closing)
            if ego.crashed and self.collision_time_s is None and clearance < 0.1:
                self.collision_time_s = stamp
                self.collision_partner = name
        if ego.crashed and self.collision_time_s is None:
            self.collision_time_s = stamp
        lead = self.actors["lead"]
        if self.scenario.template_id in ("fbrt_cutin", "fbrt_cutout_static"):
            offset = abs(float(lead.position[1] - ego.position[1]))
            changing = lead.target_lane_index == lead.destination_lane
            if self.scenario.template_id == "fbrt_cutin":
                crossed = offset < ego.WIDTH
                event_name = "first_intrusion_s"
            else:
                crossed = offset > ego.WIDTH
                event_name = "first_exit_s"
            if changing and event_name not in self.event_times and crossed:
                self.event_times[event_name] = stamp
            if changing and "lane_change_complete_s" not in self.event_times:
                target_y = self.road.network.get_lane(lead.destination_lane).position(0, 0)[1]
                if abs(lead.position[1] - target_y) < 0.2:
                    self.event_times["lane_change_complete_s"] = stamp
        self.trace.append(row)

    def _reward(self, action) -> float:
        return float(not self.vehicle.crashed)

    def _is_terminated(self) -> bool:
        return bool(self.vehicle.crashed or
                    any(vehicle.crashed for name, vehicle in self.actors.items()
                        if name != "ego"))

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def outcome(self) -> dict:
        ego_collision = bool(self.vehicle.crashed)
        background_only = (not ego_collision and any(vehicle.crashed for name, vehicle
                           in self.actors.items() if name != "ego"))
        return {"ego_collision": ego_collision,
                "collision_partner": self.collision_partner,
                "collision_time_s": self.collision_time_s,
                "other_vehicle_crashed": any(vehicle.crashed for name, vehicle
                                              in self.actors.items() if name != "ego"),
                "completed": bool(not self._is_terminated() and self._is_truncated()),
                "semantic_valid": not background_only,
                "termination_reason": ("ego_collision" if ego_collision else
                                       "inconclusive_background_termination" if background_only
                                       else "duration"),
                "near_miss": bool(not ego_collision and
                                  (self.min_ttc < 1.5 or self.min_clearance < 1.0)),
                "min_ttc": float(self.min_ttc),
                "min_clearance": float(self.min_clearance),
                "fault_active_steps": int(getattr(self.vehicle, "fault_active_steps", 0)),
                "event_times": self.event_times,
                "lead_events": self.actors["lead"].event_log}


def run_episode(profile: SUTProfile, scenario: FBRTScenario, seed: int,
                fault: str | None = None, with_trace: bool = False) -> tuple[dict, list[dict]]:
    env = FBRTEnv(profile, scenario, fault)
    try:
        env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(1)
        return env.outcome(), env.trace if with_trace else []
    finally:
        env.close()
