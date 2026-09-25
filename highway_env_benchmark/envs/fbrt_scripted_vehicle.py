"""Replayable vehicle actions for the FBRT functional interactions."""

from __future__ import annotations

import numpy as np

from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class ScriptedVehicle(ControlledVehicle):
    def __init__(self, *args, event: str, event_start_s: float = 1.0,
                 lane_change_duration_s: float = 0.8,
                 destination_lane: tuple | None = None,
                 deceleration_mps2: float = 0.0, hold_s: float = 0.0,
                 restart_acceleration_mps2: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.event = event
        self.event_start_s = event_start_s
        self.lane_change_duration_s = lane_change_duration_s
        self.destination_lane = destination_lane
        self.deceleration_mps2 = deceleration_mps2
        self.hold_s = hold_s
        self.restart_acceleration_mps2 = restart_acceleration_mps2
        self.cruise_speed = float(self.speed)
        self.elapsed = 0.0
        self.stopped_at: float | None = None
        self.event_log: list[dict] = []
        self._last_phase = "CRUISE"
        self.KP_LATERAL = 2.0 / lane_change_duration_s

    def act(self, action=None) -> None:
        if self.crashed:
            return
        phase = "CRUISE"
        acceleration = self.speed_control(self.cruise_speed)
        if self.event in ("cutin", "cutout") and self.elapsed >= self.event_start_s:
            self.target_lane_index = self.destination_lane
            phase = "LANE_CHANGE"
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
