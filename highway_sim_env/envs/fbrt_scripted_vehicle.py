"""Replayable vehicle actions for the FBRT functional interactions."""

from __future__ import annotations

import numpy as np
from collections import deque

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class ScriptedVehicle(ControlledVehicle):
    def __init__(self, *args, event: str, event_start_s: float = 1.0,
                 lane_change_duration_s: float = 0.8,
                 destination_lane: tuple | None = None,
                 deceleration_mps2: float = 0.0, hold_s: float = 0.0,
                 restart_acceleration_mps2: float = 0.0,
                 brake_after_measured_merge_s: float = 0.3,
                 brake_duration_s: float = 1.5,
                 speed_floor_mps: float = 5.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.event = event
        self.event_start_s = event_start_s
        self.lane_change_duration_s = lane_change_duration_s
        self.destination_lane = destination_lane
        self.deceleration_mps2 = deceleration_mps2
        self.hold_s = hold_s
        self.restart_acceleration_mps2 = restart_acceleration_mps2
        self.brake_after_measured_merge_s = brake_after_measured_merge_s
        self.brake_duration_s = brake_duration_s
        self.speed_floor_mps = speed_floor_mps
        self.cruise_speed = float(self.speed)
        self.elapsed = 0.0
        self.stopped_at: float | None = None
        self.event_log: list[dict] = []
        self._last_phase = "CRUISE"
        self.merge_completed_at: float | None = None
        self.KP_LATERAL = 2.0 / lane_change_duration_s

    def act(self, action=None) -> None:
        if self.crashed:
            return
        phase = "CRUISE"
        acceleration = self.speed_control(self.cruise_speed)
        if self.event in ("cutin", "cutout", "cutin_then_brake") and self.elapsed >= self.event_start_s:
            self.target_lane_index = self.destination_lane
            phase = "LANE_CHANGE"
            if self.event == "cutin_then_brake":
                lane = self.road.network.get_lane(self.destination_lane)
                lateral_offset = lane.local_coordinates(self.position)[1]
                if self.merge_completed_at is None and abs(lateral_offset) < 0.2:
                    self.merge_completed_at = self.elapsed
                    self.event_log.append({"time_s": round(self.elapsed, 3),
                                           "phase": "LANE_CHANGE_COMPLETE"})
                if self.merge_completed_at is not None:
                    since_merge = self.elapsed - self.merge_completed_at
                    if since_merge >= self.brake_after_measured_merge_s:
                        if (since_merge < self.brake_after_measured_merge_s + self.brake_duration_s
                                and self.speed > self.speed_floor_mps):
                            phase = "BRAKE_AFTER_MERGE"
                            acceleration = -min(self.deceleration_mps2,
                                                (self.speed - self.speed_floor_mps) / 0.05)
                        else:
                            phase = "POST_BRAKE_CRUISE"
                            acceleration = self.speed_control(self.speed_floor_mps)
        elif self.event in ("brake", "stop_hold_go") and self.elapsed >= self.event_start_s:
            if self.speed > 0.02 and self.stopped_at is None:
                phase = "BRAKE_TO_STOP"
                acceleration = -min(self.deceleration_mps2, self.speed * 20.0)
            else:
                if self.stopped_at is None:
                    self.stopped_at = self.elapsed
                if self.event == "stop_hold_go" and self.elapsed >= self.stopped_at + self.hold_s:
                    phase = "RESTART" if self.speed < self.cruise_speed - 0.02 else "CRUISE"
                    acceleration = (self.restart_acceleration_mps2 if phase == "RESTART"
                                    else self.speed_control(self.cruise_speed))
                else:
                    phase = "HOLD_STOP"
                    acceleration = 0.0
        elif self.event == "brake_to_floor" and self.elapsed >= self.event_start_s:
            if self.speed > self.speed_floor_mps + 0.02:
                phase = "BRAKE_TO_FLOOR"
                acceleration = -min(self.deceleration_mps2,
                                    (self.speed - self.speed_floor_mps) / 0.05)
            else:
                phase = "FLOOR_CRUISE"
                acceleration = self.speed_control(self.speed_floor_mps)
        if phase != self._last_phase:
            self.event_log.append({"time_s": round(self.elapsed, 3), "phase": phase})
            self._last_phase = phase
        steering = self.steering_control(self.target_lane_index)
        Vehicle.act(self, {"steering": float(np.clip(steering, -self.MAX_STEERING_ANGLE,
                                                    self.MAX_STEERING_ANGLE)),
                           "acceleration": float(acceleration)})

    def step(self, dt: float) -> None:
        self.elapsed += dt
        super().step(dt)


class TimedIDMVehicle(IDMVehicle):
    """Lane-fixed IDM follower with a bounded target-speed event and observed history."""

    def __init__(self, *args, event_start_s: float | None = None,
                 target_speed_increment_mps: float = 0.0,
                 max_acceleration_mps2: float = 1.5,
                 max_speed_mps: float = 35.0, **kwargs):
        kwargs["enable_lane_change"] = False
        super().__init__(*args, **kwargs)
        self.event_start_s = event_start_s
        self.target_speed_increment_mps = target_speed_increment_mps
        self.max_acceleration_mps2 = max_acceleration_mps2
        self.max_speed_mps = max_speed_mps
        self.elapsed = 0.0
        self.event_log: list[dict] = []
        self.state_history = deque(maxlen=64)
        self._remember()
        self._event_fired = False

    def _remember(self) -> None:
        self.state_history.append((self.elapsed, self.position.copy(),
                                   float(self.heading), float(self.speed),
                                   float(self.target_speed)))

    def observed_snapshot(self, age_s: float) -> Vehicle:
        threshold = self.elapsed - age_s
        observed = [row for row in self.state_history if row[0] <= threshold + 1e-9]
        _, position, heading, speed, target_speed = observed[-1] if observed else self.state_history[0]
        proxy = Vehicle(self.road, position.copy(), heading=heading, speed=speed)
        proxy.target_speed = target_speed
        return proxy

    def act(self, action=None) -> None:
        if (not self._event_fired and self.event_start_s is not None
                and self.elapsed >= self.event_start_s - 1e-9):
            self.target_speed = min(self.max_speed_mps,
                                    self.target_speed + self.target_speed_increment_mps)
            self._event_fired = True
            self.event_log.append({"time_s": round(self.elapsed, 3),
                                   "phase": "TARGET_SPEED_INCREASE"})
        super().act(action)
        if isinstance(self.action, dict):
            self.action["acceleration"] = min(float(self.action["acceleration"]),
                                               self.max_acceleration_mps2)

    def step(self, dt: float) -> None:
        super().step(dt)
        self.elapsed += dt
        self._remember()
