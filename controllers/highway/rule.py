"""IDM/MOBIL and FVDM/lane-rule controllers over the shared visible world."""

import math

import numpy as np

from .control import Control, bumper_gap, lane_steering, neighbors


class RuleController:
    def __init__(self, route, parameters):
        if route not in ("idm_mobil", "fvdm_lane"):
            raise ValueError(route)
        self.route = route
        self.parameters = dict(parameters)
        self.target_lane = None
        self.last_change_s = -math.inf

    def acceleration(self, follower, leader):
        p = self.parameters
        desired_speed = p["desired_speed_mps"]
        v = follower.speed
        if self.route == "idm_mobil":
            free = 2 * (1 - (v / desired_speed)**4)
            if leader is None:
                return float(np.clip(free, -8, 3))
            gap = max(bumper_gap(follower, leader), 0.1)
            desired_gap = 2 + max(0, v * p["headway_s"]
                                   + v * (v - leader.speed) / (2 * math.sqrt(2 * 3)))
            return float(np.clip(free - 2 * (desired_gap / gap)**2, -8, 3))
        if leader is None:
            return float(np.clip(0.6 * (desired_speed - v), -8, 3))
        gap = max(bumper_gap(follower, leader) - 4, 0.0)
        spacing = p["headway_s"] * max(v, 1)
        optimal_speed = desired_speed * (math.tanh((gap - spacing) / 10) + math.tanh(spacing / 10))
        optimal_speed /= 1 + math.tanh(spacing / 10)
        acceleration = 0.6 * (optimal_speed - v) + p["velocity_gain"] * (leader.speed - v)
        # FVDM's desired velocity vanishes at zero gap; the same actuator can stop.
        return float(np.clip(acceleration, -8, 3))

    def act(self, world):
        ego = world.ego
        current = min(range(len(world.lane_centers)), key=lambda i: abs(world.lane_centers[i] - ego.y))
        if self.target_lane is None:
            self.target_lane = current
        target_y = world.lane_centers[self.target_lane]
        in_transition = abs(ego.y - target_y) > 0.3 or abs(ego.heading) > 0.04
        current_front, old_rear = neighbors(world, world.lane_centers[current])
        current_a = self.acceleration(ego, current_front)
        if not in_transition and world.time_s - self.last_change_s >= 2:
            choices = []
            for lane in (current - 1, current + 1):
                if not 0 <= lane < len(world.lane_centers):
                    continue
                front, rear = neighbors(world, world.lane_centers[lane])
                if front and bumper_gap(ego, front) < 5:
                    continue
                rear_after = self.acceleration(rear, ego) if rear else 0
                if rear and (bumper_gap(rear, ego) < 5
                             or rear_after < -self.parameters["rear_brake_limit"]):
                    continue
                new_a = self.acceleration(ego, front)
                score = new_a - current_a
                if self.route == "idm_mobil":
                    rear_before = self.acceleration(rear, front) if rear else 0
                    old_before = self.acceleration(old_rear, ego) if old_rear else 0
                    old_after = self.acceleration(old_rear, current_front) if old_rear else 0
                    score += self.parameters["politeness"] * (rear_after - rear_before
                                                              + old_after - old_before)
                if score > 0.3:
                    choices.append((score, -lane, lane))
            if choices:
                self.target_lane = max(choices)[2]
                self.last_change_s = world.time_s
        target_y = world.lane_centers[self.target_lane]
        # During a lane change both the current and destination corridor constrain braking.
        target_front, _ = neighbors(world, target_y)
        acceleration = min(current_a, self.acceleration(ego, target_front)) if in_transition else self.acceleration(ego, target_front)
        return Control(acceleration, lane_steering(ego, target_y))
