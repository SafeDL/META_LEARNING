"""Common physical commands and lane tracking using visible state only."""

import math
from dataclasses import dataclass

import numpy as np

MIN_ACCELERATION = -8.0
MAX_ACCELERATION = 3.0
MAX_STEERING = 0.45
MAX_LATERAL_ACCELERATION = 6.0


@dataclass(frozen=True)
class Control:
    acceleration: float
    steering: float

    def __post_init__(self):
        if not math.isfinite(self.acceleration) or not math.isfinite(self.steering):
            raise ValueError("Physical command must be finite")

    def limited(self, speed=0.0, length=5.0):
        steering_limit = min(
            MAX_STEERING, math.atan(MAX_LATERAL_ACCELERATION * length / max(speed**2, 1))
        )
        return Control(
            float(np.clip(self.acceleration, MIN_ACCELERATION, MAX_ACCELERATION)),
            float(np.clip(self.steering, -steering_limit, steering_limit)),
        )


def lane_steering(ego, target_y):
    heading = np.clip(np.arctan2(0.8 * (target_y - ego.y), max(ego.speed, 2)), -0.35, 0.35)
    steering = np.arctan(ego.length * 2 * (heading - ego.heading) / max(ego.speed, 2))
    return float(np.clip(steering, -MAX_STEERING, MAX_STEERING))


def neighbors(world, y, *, exclude=None):
    vehicles = [v for v in world.vehicles if abs(v.y - y) < 2.2 and v.actor_id != exclude]
    front = [v for v in vehicles if v.x > world.ego.x]
    rear = [v for v in vehicles if v.x <= world.ego.x]
    return (
        min(front, key=lambda v: v.x) if front else None,
        max(rear, key=lambda v: v.x) if rear else None,
    )


def bumper_gap(behind, ahead):
    return ahead.x - behind.x - (ahead.length + behind.length) / 2
