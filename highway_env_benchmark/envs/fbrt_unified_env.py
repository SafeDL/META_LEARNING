"""One 20 Hz physical runner for the registered legacy/native/external FBRT builds."""

from __future__ import annotations

from collections import deque

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.envs.common.observation import observation_factory
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
import numpy as np
from shapely.geometry import Polygon

from highway_env_benchmark.envs.fbrt_metrics import (
    longitudinal_bumper_clearance, time_to_collision,
)
from highway_env_benchmark.envs.fbrt_scripted_vehicle import ScriptedVehicle
from method_chains.failure_memory_regression.schema_v2 import BuildSpec, stable_hash
from sut_algorithms.highway_env.fbrt_adapters import adapter_for
from sut_algorithms.highway_env.registry import build_spec_factory


PHYSICS_HZ = 20
DT = 1.0 / PHYSICS_HZ
EXECUTION_CONTRACT = ("fbrt-unified-v4;highway-env-1.9.1;20Hz;"
                      "buildspec-control-v1;ego-initial-v1;event-detector-v2;"
                      "native-mobil-timer-fixed-v1")


class FBRTUnifiedEnv(AbstractEnv):
    def __init__(self, spec: BuildSpec, scenario: dict):
        self.spec = spec
        self.scenario = scenario
        self.adapter = adapter_for(spec)
        self.actors: dict[str, Vehicle] = {}
        self.event_times: dict[str, float] = {}
        self.trace: list[dict] = []
        self.min_ttc = float("inf")
        self.min_clearance = float("inf")
        self.collision_partner_role: str | None = None
        self.collision_time_s: float | None = None
        self.control_actions: list[dict] = []
        self.initial_observation = None
        self.observation_history = deque(maxlen=64)
        config = {"duration": float(self._context().get("duration_s", 12.0))}
        super().__init__(config=config)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {"type": "DiscreteMetaAction", "longitudinal": True, "lateral": True},
            "observation": {
                "type": "Kinematics", "vehicles_count": 5,
                "features": ["x", "y", "vx", "vy", "sin_h", "cos_h"],
                "features_range": {"x": [-50, 50], "y": [-50, 50],
                                   "vx": [-40, 40], "vy": [-40, 40]},
                "absolute": False, "order": "sorted", "normalize": True,
                "see_behind": True,
            },
            "simulation_frequency": 20, "policy_frequency": 5,
            "duration": 12.0, "road_length": 2000.0,
            "collision_reward": -1.0, "right_lane_reward": 0.1,
            "high_speed_reward": 0.4, "lane_change_reward": -0.05,
        })
        return config

    def _context(self) -> dict:
        return dict(self.scenario.get("fixed_context", {}))

    def _parameters(self) -> dict:
        return dict(self.scenario.get("active_parameters", {}))

    def _lane(self, lane_id: int):
        return self.road.network.get_lane(("0", "1", lane_id))

    @property
    def scheduled_vehicle(self):
        return self.actors["lead"]

    def ttc_observation(self) -> np.ndarray:
        return compute_ttc_grid(self, time_quantization=1.0, horizon=6.0)

    def _reset(self) -> None:
        self.adapter.reset()
        self.actors = {}
        self.event_times = {}
        self.trace = []
        self.min_ttc = float("inf")
        self.min_clearance = float("inf")
        self.collision_partner_role = None
        self.collision_time_s = None
        self.control_actions = []
        self.observation_history = deque(maxlen=64)
        ctx, params = self._context(), self._parameters()
        lane_count = int(ctx.get("lane_count", 2))
        self.config["duration"] = float(ctx.get("duration_s", 12.0))
        self.config["lanes_count"] = lane_count
        self.config["simulation_frequency"] = PHYSICS_HZ
        self.config["policy_frequency"] = int(self.spec.control_hz)
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                lane_count, start=0, length=float(self.config["road_length"]), speed_limit=40),
            np_random=self.np_random,
        )
        ego_lane_id = int(params.get("ego_initial_lane_id", ctx.get("ego_lane_id", 0)))
        if ego_lane_id < 0 or ego_lane_id >= lane_count:
            raise ValueError("ego_initial_lane_id is outside the configured road")
        adjacent_lane_id = 1 - ego_lane_id if lane_count == 2 else None
        self.ego_lane_index = ("0", "1", ego_lane_id)
        ego_lane = self._lane(ego_lane_id)
        ego_x = float(params.get("ego_initial_x_m", ctx.get("ego_x_m", 120.0)))
        ego_speed = float(params.get("ego_initial_speed_mps", ctx["ego_speed_mps"]))
        lateral_offset = float(params.get("ego_initial_lateral_offset_m", 0.0))
        heading_offset = float(params.get("ego_initial_heading_offset_rad", 0.0))
        if not (10.0 <= ego_x <= float(self.config["road_length"]) - 10.0
                and 0.0 <= ego_speed <= 40.0
                and abs(lateral_offset) <= 0.5
                and abs(heading_offset) <= 0.1):
            raise ValueError("ego initial state is outside the physical audit bounds")
        ego = self.adapter.create_vehicle(self, ego_lane, ego_x, ego_speed,
                                          float(ctx.get("desired_speed_mps", ego_speed)),
                                          lateral_offset_m=lateral_offset,
                                          heading_offset_rad=heading_offset)
        self.vehicle = ego
        self.initial_mobil_timer = getattr(ego, "timer", None)
        self.initial_ego_lane_index = ego.target_lane_index
        self.controlled_vehicles = [ego]
        self.actors["ego"] = ego
        template = self.scenario["template_id"]
        ego_length = float(ego.LENGTH)

        def lead_center(clearance: float, actor_length: float = 5.0) -> float:
            return ego_x + ego_length / 2.0 + clearance + actor_length / 2.0

        if template == "fbrt_cutin":
            lane_id, event = adjacent_lane_id, "cutin"
            self._add_scripted("lead", lane_id, lead_center(params["initial_clearance_m"]),
                               float(ctx["lead_speed_mps"]), event=event,
                               destination_lane=self.ego_lane_index,
                               lane_change_duration_s=float(params["lane_change_time_scale_s"]),
                               event_start_s=float(ctx.get("event_start_s", 1.0)))
        elif template == "fbrt_cutout_static":
            lead_speed = float(ctx["lead_speed_mps"])
            event_start = float(ctx.get("event_start_s", 1.0))
            gap = float(params["lead_to_static_ttc_start_s"]) * lead_speed
            lead_x = lead_center(params["initial_clearance_m"])
            self._add_scripted("lead", ego_lane_id, lead_x, lead_speed, event="cutout",
                               destination_lane=("0", "1", adjacent_lane_id),
                               lane_change_duration_s=float(ctx.get("lane_change_time_scale_s", 0.8)),
                               event_start_s=event_start)
            static_x = (lead_x + lead_speed * event_start + 5.0 / 2.0 +
                        gap + 5.0 / 2.0)
            static_lane = self._lane(ego_lane_id)
            static = Vehicle(self.road, static_lane.position(static_x, 0.0),
                             heading=static_lane.heading_at(static_x), speed=0.0)
            self.actors["static"] = static
        elif template == "fbrt_lane_change_rear":
            self._add_scripted("lead", ego_lane_id, lead_center(float(ctx["lead_clearance_m"])),
                               float(ctx["lead_speed_mps"]), event="cruise")
            clearance = float(params["rear_clearance_m"])
            rear_speed = ego_speed + float(params["rear_closing_speed_mps"])
            rear_length = 5.0
            rear_x = ego_x - ego_length / 2.0 - clearance - rear_length / 2.0
            self._add_scripted("rear", adjacent_lane_id, rear_x, rear_speed, event="cruise")
        elif template == "fbrt_moving_lead":
            self._add_scripted("lead", ego_lane_id, lead_center(params["initial_clearance_m"]),
                               float(params["lead_speed_mps"]), event="cruise")
        elif template == "fbrt_cutin_then_brake":
            lead_x = lead_center(params["initial_clearance_m"])
            self._add_scripted(
                "lead", adjacent_lane_id, lead_x, float(ctx["lead_speed_mps"]), event="cutin_then_brake",
                destination_lane=self.ego_lane_index,
                lane_change_duration_s=float(ctx["lane_change_time_scale_s"]),
                event_start_s=float(ctx.get("event_start_s", 1.0)),
                deceleration_mps2=float(params["lead_deceleration_mps2"]),
                brake_after_measured_merge_s=float(ctx["brake_after_measured_merge_s"]),
                brake_duration_s=float(ctx["brake_duration_s"]),
                speed_floor_mps=float(ctx["lead_speed_floor_mps"]),
            )
        else:
            raise ValueError(f"scenario template is not implemented in v2 runner: {template}")
        self.road.vehicles = list(self.actors.values())
        self.observation_type = observation_factory(self, self.config["observation"])
        self.initial_observation = np.array(self.observation_type.observe(), copy=True)
        self.observation_history.append((0.0, self.initial_observation))
        self._record_step()

    def _add_scripted(self, role: str, lane_id: int, longitudinal_position: float,
                      speed: float, event: str, destination_lane: tuple | None = None,
                      **schedule) -> ScriptedVehicle:
        lane = self._lane(lane_id)
        vehicle = ScriptedVehicle(
            self.road, lane.position(longitudinal_position, 0.0),
            heading=lane.heading_at(longitudinal_position), speed=speed,
            target_lane_index=("0", "1", lane_id), target_speed=speed,
            event=event, destination_lane=destination_lane, **schedule,
        )
        self.actors[role] = vehicle
        return vehicle

    def _delayed_observation(self, now: float) -> np.ndarray:
        threshold = now - float((self.spec.mutation or {}).get("observation_delay_s", 0.0))
        eligible = [observation for stamp, observation in self.observation_history
                    if stamp <= threshold + 1e-9]
        return np.array(eligible[-1] if eligible else self.initial_observation, copy=True)

    def _advance(self) -> None:
        now = self.steps / PHYSICS_HZ
        observation = np.array(self.observation_type.observe(), copy=True)
        self.observation_history.append((now, observation))
        external = self.spec.adapter_kind == "external_meta_policy"
        decision_frames = max(1, PHYSICS_HZ // int(self.spec.control_hz))
        is_decision = not external or self.steps % decision_frames == 0
        if is_decision:
            policy_observation = (self._delayed_observation(now)
                                  if (self.spec.mutation or {}).get("observation_delay_s")
                                  else observation)
            action = self.adapter.act(self, policy_observation) if external else None
            if external:
                self.action_type.act(action)
                names = getattr(self.action_type, "actions", {})
                self.control_actions.append({"time_s": round(now, 3),
                                             "action": names.get(action, str(action))})
        else:
            action = None

        # Exactly one control owner acts for ego. `road.act()` is deliberately
        # avoided because it would call ego a second time after the policy action.
        if not external or not is_decision:
            self.vehicle.act()
        for role, actor in self.actors.items():
            if role != "ego":
                actor.act()
        self.road.step(DT)
        self.steps += 1
        self.time = self.steps / PHYSICS_HZ
        self._record_step()

    def _record_step(self) -> None:
        ego = self.vehicle
        stamp = round(self.steps / PHYSICS_HZ, 3)
        ego_polygon = Polygon(ego.polygon())
        row = {"time_s": stamp}
        for role, actor in self.actors.items():
            row[role] = {"x_m": float(actor.position[0]), "y_m": float(actor.position[1]),
                         "heading_rad": float(actor.heading),
                         "speed_mps": float(actor.speed), "crashed": bool(actor.crashed)}
            if role == "ego":
                continue
            clearance = float(ego_polygon.distance(Polygon(actor.polygon())))
            self.min_clearance = min(self.min_clearance, clearance)
            lateral_overlap = abs(float(actor.position[1] - ego.position[1])) < (
                float(actor.WIDTH + ego.WIDTH) / 2.0 + 0.25)
            if lateral_overlap:
                if actor.position[0] >= ego.position[0]:
                    bumper_gap = longitudinal_bumper_clearance(
                        float(actor.position[0]), float(actor.LENGTH),
                        float(ego.position[0]), float(ego.LENGTH))
                    closing = float(ego.speed - actor.speed)
                else:
                    bumper_gap = longitudinal_bumper_clearance(
                        float(ego.position[0]), float(ego.LENGTH),
                        float(actor.position[0]), float(actor.LENGTH))
                    closing = float(actor.speed - ego.speed)
                ttc = time_to_collision(bumper_gap, closing)
                if ttc is not None:
                    self.min_ttc = min(self.min_ttc, ttc)
                if ego.crashed and self.collision_partner_role is None and clearance < 0.25:
                    self.collision_partner_role = role
                    self.collision_time_s = stamp

        if ego.target_lane_index != self.initial_ego_lane_index:
            self.event_times.setdefault("ego_lane_change_initiated_s", stamp)
            target = self.road.network.get_lane(ego.target_lane_index)
            if (ego.lane_index == ego.target_lane_index and
                    abs(target.local_coordinates(ego.position)[1]) < 0.2):
                self.event_times.setdefault("ego_lane_change_complete_s", stamp)
        lead = self.actors.get("lead")
        if lead is not None and getattr(lead, "destination_lane", None) is not None:
            lane = self.road.network.get_lane(lead.destination_lane)
            lateral = lane.local_coordinates(lead.position)[1]
            template = self.scenario["template_id"]
            if template == "fbrt_cutin" and abs(lateral) < float(ego.WIDTH):
                self.event_times.setdefault("first_intrusion_s", stamp)
            elif template == "fbrt_cutout_static":
                origin_lane = self.road.network.get_lane(self.ego_lane_index)
                origin_lateral = origin_lane.local_coordinates(lead.position)[1]
                if abs(origin_lateral) > float(ego.WIDTH):
                    self.event_times.setdefault("first_exit_s", stamp)
            elif template == "fbrt_cutin_then_brake" and abs(lateral) < float(ego.WIDTH):
                self.event_times.setdefault("first_intrusion_s", stamp)
            if abs(lateral) < 0.2:
                self.event_times.setdefault("lead_lane_change_complete_s", stamp)
            if getattr(lead, "merge_completed_at", None) is not None:
                self.event_times.setdefault("measured_merge_complete_s",
                                            round(float(lead.merge_completed_at), 3))
        self.trace.append(row)

    def result(self) -> dict:
        ego_collision = bool(self.vehicle.crashed)
        background_only = not ego_collision and any(
            actor.crashed for role, actor in self.actors.items() if role != "ego")
        completed = not ego_collision and not background_only and self.steps >= int(
            round(float(self.config["duration"]) * PHYSICS_HZ))
        lead = self.actors.get("lead")
        mutation_count = int(getattr(self.vehicle, "rear_guard_bypassed_count", 0))
        if lead is not None:
            mutation_count += int(getattr(lead, "rear_guard_bypassed_count", 0))
        return {
            "build_id": self.spec.build_id,
            "build_fingerprint": self.spec.fingerprint,
            "scenario_id": self.scenario["scenario_id"],
            "template_id": self.scenario["template_id"],
            "scenario": self.scenario,
            "execution_contract_version": EXECUTION_CONTRACT,
            "completed": completed,
            "ego_collision": ego_collision,
            "inconclusive": background_only or not completed and not ego_collision,
            "collision_partner_role": self.collision_partner_role,
            "collision_time_s": self.collision_time_s,
            "min_ttc": None if not np.isfinite(self.min_ttc) else float(self.min_ttc),
            "min_clearance": None if not np.isfinite(self.min_clearance) else float(self.min_clearance),
            "public_signature": {"event_times": dict(self.event_times),
                                 "background_phases": [event for role, actor in self.actors.items()
                                                       if role != "ego" for event in
                                                       getattr(actor, "event_log", [])]},
            "event_times": dict(self.event_times),
            "lead_events": list(getattr(lead, "event_log", [])) if lead else [],
            "ego_action_count": len(self.control_actions),
            "ego_actions": list(self.control_actions),
            "ego_runtime": {
                "adapter_kind": self.spec.adapter_kind,
                "vehicle_class": type(self.vehicle).__name__,
                "policy_class": type(self.adapter.policy).__name__ if self.adapter.policy else None,
                "native_act_calls": int(getattr(self.vehicle, "act_call_count", 0)),
                "mobil_decision_calls": int(getattr(self.vehicle, "mobil_decision_count", 0)),
                "initial_mobil_timer_s": self.initial_mobil_timer,
                "checkpoint_sha256": self.spec.checkpoint_sha256,
                "initial_x_m": float(self.trace[0]["ego"]["x_m"]),
                "initial_y_m": float(self.trace[0]["ego"]["y_m"]),
                "initial_heading_rad": float(self.trace[0]["ego"]["heading_rad"]),
                "initial_speed_mps": float(self.trace[0]["ego"]["speed_mps"]),
            },
            "rear_guard_bypassed_count": mutation_count,
            "episode_cost": 1,
            "termination_reason": ("ego_collision" if ego_collision else
                                   "inconclusive_background_termination" if background_only else
                                   "duration" if completed else "event_incomplete"),
        }


def run_build_episode(build_id: str, scenario: dict, seed: int,
                      with_trace: bool = False) -> tuple[dict, list[dict]]:
    spec = build_spec_factory(build_id)
    env = FBRTUnifiedEnv(spec, scenario)
    try:
        env.reset(seed=seed)
        max_steps = int(round(float(env.config["duration"]) * PHYSICS_HZ))
        while env.steps < max_steps:
            if any(actor.crashed for actor in env.actors.values()):
                break
            env._advance()
        result = env.result()
        result["simulator_seed"] = int(seed)
        result["scenario_fingerprint"] = stable_hash({
            "scenario": scenario, "execution_contract": EXECUTION_CONTRACT})
        result["execution_id"] = "new-" + stable_hash({
            "build_fingerprint": spec.fingerprint, "scenario_fingerprint": result["scenario_fingerprint"],
            "seed": int(seed), "contract": EXECUTION_CONTRACT})[:24]
        return result, list(env.trace) if with_trace else []
    finally:
        env.close()
