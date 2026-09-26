"""A genuinely multi-step highway-env adaptation environment for AdaTE A1.

It deliberately leaves :mod:`highway_sim_env.envs.cutin_env` untouched.  The
background vehicle receives one discrete residual-acceleration action per
policy step, and the trace records the *stored* low-level acceleration after
the scheduled controller and residual have been combined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highway_env.vehicle.kinematics import Vehicle

from highway_sim_env.envs.cutin_env import CutInEnv, CutInScenario, ScheduledCutInVehicle
from sut_algorithms.highway_env.idm_profiles import SUTProfile, create_profiled_vehicle

MODE_CODES = {
    name: index
    for index, name in enumerate((
        CutInEnv.FAST_INTRUSION,
        CutInEnv.CUTIN_BRAKING,
        CutInEnv.LEAD_BRAKING,
        CutInEnv.STOP_AND_GO,
        CutInEnv.SLOW_LEAD_FOLLOWING,
        CutInEnv.PASSING_CUTIN,
    ))
}


class ResidualScheduledCutInVehicle(ScheduledCutInVehicle):
    """Scheduled vehicle whose externally chosen residual reaches dynamics."""

    residual_acceleration: float = 0.0
    residual_limit: float = 2.0

    def act(self, action: dict | str = None) -> None:
        self.follow_road()
        if self.cutin_start is not None and self.elapsed + self._TIME_EPSILON >= self.cutin_start:
            self.target_lane_index = self.cutin_target_lane_index
        acceleration = self.speed_control(self.target_speed)
        if self._is_braking():
            acceleration = -self.braking_deceleration
        acceleration += float(
            np.clip(self.residual_acceleration, -self.residual_limit, self.residual_limit))
        command = {
            "steering":
            np.clip(self.steering_control(self.target_lane_index), -self.MAX_STEERING_ANGLE,
                    self.MAX_STEERING_ANGLE),
            "acceleration":
            float(np.clip(acceleration, -10.0, 5.0)),
        }
        # Calling Vehicle.act is essential: ControlledVehicle.act would rebuild
        # its command and silently discard the learned NPC action.
        Vehicle.act(self, command)


@dataclass(frozen=True)
class DenseSnapshot:
    scenario: CutInScenario
    seed: int
    action_history: tuple[int, ...]


@dataclass(frozen=True)
class DenseTransitionTrace:
    time: np.ndarray
    ego_x: np.ndarray
    ego_y: np.ndarray
    ego_speed: np.ndarray
    npc_x: np.ndarray
    npc_y: np.ndarray
    npc_speed: np.ndarray
    npc_acceleration: np.ndarray
    residual_action: np.ndarray
    leading_acceleration: np.ndarray
    distance: np.ndarray
    ttc: np.ndarray


class DenseCutInEnv(CutInEnv):
    """CutInEnv variant with an actual discrete background-vehicle action."""

    ACTIONS = np.asarray([-2.0, 0.0, 2.0], dtype=float)
    PASSING_ACTIONS = np.asarray([-6.0, 0.0, 2.0], dtype=float)

    def __init__(self,
                 profile: SUTProfile,
                 scenario: CutInScenario,
                 render_mode: str | None = None) -> None:
        self._pending_action = 1
        self._action_history: list[int] = []
        self._seed_for_snapshot = 0
        self._dense_trace: list[tuple[float, ...]] = []
        super().__init__(profile, scenario, render_mode=render_mode)

    def reset(self,
              *,
              seed: int | None = None,
              options: dict | None = None):  # type: ignore[override]
        self._seed_for_snapshot = 0 if seed is None else int(seed)
        self._pending_action = 1
        self._action_history = []
        self._dense_trace = []
        return super().reset(seed=seed, options=options)

    def _create_vehicles(self) -> None:
        ego_lane, adjacent_lane = ("0", "1", 0), ("0", "1", 1)
        ego_road_lane = self.road.network.get_lane(ego_lane)
        ego = create_profiled_vehicle(self.road,
                                      ego_road_lane.position(60.0, 0.0),
                                      heading=ego_road_lane.heading_at(60.0),
                                      speed=self.EGO_SPEED,
                                      target_lane_index=ego_lane,
                                      target_speed=self.profile.target_speed,
                                      profile=self.profile)
        schedule = self._mode_schedule()
        lead_speed = self.EGO_SPEED + self.scenario.relative_speed
        lead_longitudinal = 60.0 + self.scenario.initial_gap
        if schedule.initial_lane == 1:
            lead_longitudinal -= self.scenario.relative_speed * self.CUTIN_START
        lead_lane = ego_lane if schedule.initial_lane == 0 else adjacent_lane
        target_lane = ego_lane if schedule.target_lane == 0 else adjacent_lane
        road_lane = self.road.network.get_lane(lead_lane)
        lead = ResidualScheduledCutInVehicle(self.road,
                                             road_lane.position(lead_longitudinal, 0.0),
                                             heading=road_lane.heading_at(lead_longitudinal),
                                             speed=lead_speed,
                                             target_lane_index=lead_lane,
                                             target_speed=lead_speed,
                                             cutin_start=schedule.cutin_start,
                                             cutin_duration=schedule.cutin_duration,
                                             cutin_target_lane_index=target_lane,
                                             brake_start=schedule.brake_start,
                                             brake_duration=schedule.brake_duration,
                                             braking_deceleration=schedule.braking_deceleration)
        self.vehicle, self._cutin_vehicle = ego, lead
        vehicles = [ego, lead]
        if self.scenario.mode == self.PASSING_CUTIN:
            leading_longitudinal = lead_longitudinal + 24.0
            leading_speed = max(lead_speed - 3.0, 8.0)
            leading = ResidualScheduledCutInVehicle(
                self.road,
                road_lane.position(leading_longitudinal, 0.0),
                heading=road_lane.heading_at(leading_longitudinal),
                speed=leading_speed,
                target_lane_index=lead_lane,
                target_speed=leading_speed,
                cutin_start=None,
                cutin_duration=self.DEFAULT_CUTIN_DURATION,
                cutin_target_lane_index=lead_lane,
            )
            self._leading_vehicle = leading
            lead.residual_limit = 6.0
            vehicles.append(leading)
        self.road.vehicles = vehicles

    def _simulate(self, action: int | None = None) -> None:
        if action is not None:
            action_count = len(
                self.PASSING_ACTIONS
                if self.scenario.mode == self.PASSING_CUTIN
                else self.ACTIONS
            )
            if int(action) not in range(action_count):
                raise ValueError("dense action outside the declared discrete action set")
            self._pending_action = int(action)
            self._action_history.append(int(action))
        lead = self._cutin_vehicle
        assert isinstance(lead, ResidualScheduledCutInVehicle)
        leading = self._leading_vehicle
        if self.scenario.mode == self.PASSING_CUTIN:
            lead.residual_acceleration = float(self.PASSING_ACTIONS[self._pending_action])
        else:
            lead.residual_acceleration = float(self.ACTIONS[self._pending_action])
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        dt = 1 / self.config["simulation_frequency"]
        for _ in range(frames):
            self.road.act()
            self.road.step(dt)
            self.steps += 1
            self._record_lead_trace()
            self._record_dense_trace()
            self._update_safety_metrics()

    def _record_dense_trace(self) -> None:
        lead = self._cutin_vehicle
        assert lead is not None
        distance = float(np.linalg.norm(lead.position - self.vehicle.position))
        longitudinal_gap = float(lead.position[0] - self.vehicle.position[0])
        closing = float(self.vehicle.speed - lead.speed)
        ttc = longitudinal_gap / closing if longitudinal_gap > 0 and closing > 1e-8 else np.inf
        leading = self._leading_vehicle
        leading_acceleration = (
            float(leading.action["acceleration"]) if leading is not None else 0.0
        )
        self._dense_trace.append(
            (self.time, self.vehicle.position[0], self.vehicle.position[1], self.vehicle.speed,
             lead.position[0], lead.position[1], lead.speed, lead.action["acceleration"],
             lead.residual_acceleration, leading_acceleration, distance, ttc))

    def state_features(self) -> dict[str, float]:
        lead = self._cutin_vehicle
        assert lead is not None
        schedule = self._mode_schedule()
        if schedule.cutin_start is not None and self.time < schedule.cutin_start:
            schedule_phase = 0.0
        elif schedule.cutin_start is not None and self.time < schedule.cutin_start + schedule.cutin_duration:
            schedule_phase = 1.0
        elif schedule.brake_start is not None and self.time < schedule.brake_start + schedule.brake_duration:
            schedule_phase = 2.0
        else:
            schedule_phase = 3.0
        return {
            "longitudinal_gap": float(lead.position[0] - self.vehicle.position[0]),
            "lateral_gap": float(lead.position[1] - self.vehicle.position[1]),
            "relative_speed": float(lead.speed - self.vehicle.speed),
            "npc_speed": float(lead.speed),
            "leading_gap": (
                float(self._leading_vehicle.position[0] - lead.position[0])
                if self._leading_vehicle is not None else 0.0
            ),
            "leading_relative_speed": (
                float(self._leading_vehicle.speed - lead.speed)
                if self._leading_vehicle is not None else 0.0
            ),
            "ego_speed": float(self.vehicle.speed),
            "ego_heading": float(self.vehicle.heading),
            "npc_heading": float(lead.heading),
            "npc_target_lane": float(lead.target_lane_index[-1]),
            "schedule_phase": schedule_phase,
            "time": float(self.time),
            "mode_code": float(MODE_CODES[self.scenario.mode]),
        }

    def snapshot(self) -> DenseSnapshot:
        return DenseSnapshot(self.scenario, self._seed_for_snapshot, tuple(self._action_history))

    def restore(self, snapshot: DenseSnapshot) -> None:
        if snapshot.scenario != self.scenario:
            raise ValueError("snapshot belongs to a different scenario")
        self.reset(seed=snapshot.seed)
        for action in snapshot.action_history:
            _, _, terminated, truncated, _ = self.step(action)
            if terminated or truncated:
                break

    def dense_trace(self) -> DenseTransitionTrace:
        if not self._dense_trace:
            raise RuntimeError("No dense trace is available before stepping the environment")
        values = np.asarray(self._dense_trace, dtype=float)
        return DenseTransitionTrace(*[values[:, column] for column in range(values.shape[1])])

    def _is_terminated(self) -> bool:
        if self.scenario.mode == self.PASSING_CUTIN:
            return bool(self.vehicle.crashed)
        return super()._is_terminated()
