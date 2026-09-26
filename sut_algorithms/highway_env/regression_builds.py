"""Explicit controlled build differences used in FBRT-Memory v2."""

from __future__ import annotations

import numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle


class RearGuardIDMVehicle(IDMVehicle):
    """IDM+MOBIL with the new-lane rear-braking veto optionally bypassed."""

    def __init__(self, *args, rear_guard_enabled: bool = True, **kwargs):
        self.rear_guard_enabled = rear_guard_enabled
        self.rear_guard_bypassed_count = 0
        self.mobil_decision_count = 0
        self.act_call_count = 0
        # highway-env otherwise phases lane-change decisions from absolute x+y.
        # Fix the phase for controlled initial-position comparisons.
        kwargs.setdefault("timer", 1e-6)
        super().__init__(*args, **kwargs)

    def act(self, action=None) -> None:
        self.act_call_count += 1
        super().act(action)

    def mobil(self, lane_index) -> bool:
        self.mobil_decision_count += 1
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding
        )
        new_following_pred_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=self
        )
        guard_would_reject = new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED
        if guard_would_reject and self.rear_guard_enabled:
            return False
        if guard_would_reject:
            self.rear_guard_bypassed_count += 1

        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                self.route[0][2] - self.target_lane_index[2]
            ):
                return False
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=self
            )
            old_following_pred_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=old_preceding
            )
            jerk = (
                self_pred_a - self_a
                + self.POLITENESS * (
                    new_following_pred_a - new_following_a
                    + old_following_pred_a - old_following_a
                )
            )
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False
        return True


def make_native_vehicle(build_id: str, road, position, *, heading: float,
                        speed: float, target_lane_index: tuple,
                        target_speed: float):
    guard_enabled = build_id != "mobil_rear_guard_off_v2"
    return RearGuardIDMVehicle(
        road, position, heading=heading, speed=speed,
        target_lane_index=target_lane_index, target_speed=target_speed,
        rear_guard_enabled=guard_enabled,
    )
