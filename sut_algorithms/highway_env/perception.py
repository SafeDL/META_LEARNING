"""Opt-in, observable-state perception for the driving-controller benchmark.

The delay applies to every sampled front state, rather than only its first
appearance. Prediction uses current positions and velocities; scripted event
times, destination lanes and future actions are never inspected.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from highway_env.vehicle.kinematics import Vehicle

from .idm_profiles import SUTProfile, ProfiledIDMVehicle, ProfiledFVDMVehicle


@dataclass(frozen=True)
class PerceptionProfile(SUTProfile):
    perception_mode: str = "lane"
    perception_delay_s: float = 0.0
    prediction_horizon_s: float = 0.0


class PerceptionMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.profile.perception_mode not in ("lane", "constant_velocity"):
            raise ValueError("unsupported perception mode")
        if self.profile.reaction_delay != 0 or not 0 <= self.profile.perception_delay_s <= 2:
            raise ValueError("These controllers use continuous perception delay exclusively")
        if not 0 <= self.profile.prediction_horizon_s <= 3:
            raise ValueError("invalid prediction horizon")
        self.perception_history = deque()
        self.first_detected_s = None
        self.first_used_s = None
        self.predictive_detection_count = 0
        self.commanded_accelerations = []

    def perceived_front(self):
        front, _ = self.road.neighbour_vehicles(self, self.lane_index)
        predictive = False
        if self.profile.perception_mode == "constant_velocity":
            lane = self.lane
            ego_s, _ = lane.local_coordinates(self.position)
            candidates = []
            for actor in self.road.vehicles:
                if actor is self:
                    continue
                s, lateral = lane.local_coordinates(actor.position)
                if s <= ego_s:
                    continue
                lane_heading = lane.heading_at(s)
                normal = np.array([-np.sin(lane_heading), np.cos(lane_heading)])
                lateral_velocity = float(actor.velocity @ normal)
                future_lateral = lateral + lateral_velocity * self.profile.prediction_horizon_s
                overlap = (self.WIDTH + actor.WIDTH) / 2
                enters = (lateral * lateral_velocity < 0 and
                          (abs(future_lateral) <= overlap or lateral * future_lateral < 0))
                if actor is front or abs(lateral) <= overlap or enters:
                    candidates.append((s, actor))
            if candidates:
                selected = min(candidates, key=lambda p: p[0])[1]
                predictive = selected is not front
                front = selected
        snapshot = None
        if front is not None:
            # A new Vehicle copies measured state without a live actor reference.
            snapshot = Vehicle(self.road, np.array(front.position, copy=True),
                               heading=float(front.heading), speed=float(front.speed))
            if self.first_detected_s is None:
                self.first_detected_s = self.elapsed
            self.predictive_detection_count += int(predictive)
        self.perception_history.append((self.elapsed, snapshot))
        cutoff = self.elapsed - self.profile.perception_delay_s
        while len(self.perception_history) > 1 and self.perception_history[1][0] <= cutoff + 1e-9:
            self.perception_history.popleft()
        stamp, delayed = self.perception_history[0]
        used = None
        if delayed is not None and stamp <= cutoff + 1e-9:
            # Carry the stale measured velocity forward; do not freeze an
            # absolute position while the ego continues to move.
            used = Vehicle(self.road, delayed.position + delayed.velocity * (self.elapsed-stamp),
                           heading=delayed.heading, speed=delayed.speed)
        if used is not None and self.first_used_s is None:
            self.first_used_s = self.elapsed
        return used

    def act(self, action=None):
        if self.crashed:
            return
        self.follow_road()
        front = self.perceived_front()
        steering = np.clip(self.steering_control(self.target_lane_index),
                           -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        acceleration = self.acceleration(self, front, None)
        self.commanded_accelerations.append(float(acceleration))
        Vehicle.act(self, {"steering": steering, "acceleration": acceleration})


class PerceivedIDMVehicle(PerceptionMixin, ProfiledIDMVehicle):
    pass


class PerceivedFVDMVehicle(PerceptionMixin, ProfiledFVDMVehicle):
    """FVDM with a physical standstill gap, using a safe standstill gap."""

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        target_speed = min(self.profile.target_speed, ego_vehicle.lane.speed_limit)
        optimal_speed = target_speed
        velocity_difference = 0.0
        if front_vehicle is not None:
            center_distance = ego_vehicle.lane_distance_to(front_vehicle)
            net_gap = center_distance - (ego_vehicle.LENGTH + front_vehicle.LENGTH) / 2
            available = max(net_gap - self.profile.desired_gap, 0.0)
            optimal_speed = target_speed * np.tanh(available / self.profile.fvdm_transition_gap)
            velocity_difference = front_vehicle.speed - ego_vehicle.speed
        acceleration = (self.profile.fvdm_sensitivity * (optimal_speed - ego_vehicle.speed)
                        + self.profile.fvdm_velocity_gain * velocity_difference)
        return float(np.clip(acceleration, -self.profile.max_brake, self.profile.comfort_acceleration))
