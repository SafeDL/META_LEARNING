"""Synchronous, passive rectangle TTC/longitudinal DRAC/body-distance SSM."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from shapely.geometry import Polygon


@dataclass(frozen=True)
class Body:
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    length: float = 5.0
    width: float = 2.0

    def validate(self):
        if (np.shape(self.position) != (2,) or np.shape(self.velocity) != (2,) or
            not np.all(np.isfinite(np.r_[self.position, self.velocity, self.heading,
                                        self.length, self.width])) or
            self.length <= 0 or self.width <= 0):
            raise ValueError("invalid measured body state")

    def axes(self):
        c, s = math.cos(self.heading), math.sin(self.heading)
        return np.array([[c, s], [-s, c]])

    def radius(self, axis):
        axes = self.axes()
        return .5 * (self.length * abs(axes[0] @ axis) + self.width * abs(axes[1] @ axis))

    def corners(self):
        axes = self.axes()
        return self.position + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) @ (
            np.diag([self.length / 2, self.width / 2]) @ axes)


def rectangle_ttc(first: Body, second: Body, horizon=6.0):
    """Continuous swept SAT under current constant velocity and fixed heading."""
    first.validate()
    second.validate()
    if horizon <= 0 or not np.isfinite(horizon):
        raise ValueError("invalid prediction horizon")
    delta, velocity = second.position - first.position, second.velocity - first.velocity
    entry, exit = 0.0, horizon
    for axis in np.concatenate([first.axes(), second.axes()]):
        distance, speed = float(delta @ axis), float(velocity @ axis)
        radius = first.radius(axis) + second.radius(axis)
        if abs(speed) < 1e-12:
            if abs(distance) > radius + 1e-12:
                return math.inf
            continue
        times = sorted(((-radius - distance) / speed, (radius - distance) / speed))
        entry, exit = max(entry, times[0]), min(exit, times[1])
        if entry > exit + 1e-12:
            return math.inf
    return max(0.0, entry)


def longitudinal_drac(first: Body, second: Body, road_heading=0.0):
    first.validate()
    second.validate()
    longitudinal = np.array([math.cos(road_heading), math.sin(road_heading)])
    lateral = np.array([-longitudinal[1], longitudinal[0]])
    delta = second.position - first.position
    if abs(delta @ lateral) > first.radius(lateral) + second.radius(lateral) + 1e-12:
        return 0.0, "not_applicable_lateral"
    separation = float(delta @ longitudinal)
    if abs(separation) < 1e-12:
        return 0.0, "not_applicable_no_longitudinal_order"
    closing = float((first.velocity - second.velocity) @ longitudinal) * np.sign(separation)
    if closing <= 0:
        return 0.0, "not_applicable_not_closing"
    gap = abs(separation) - first.radius(longitudinal) - second.radius(longitudinal)
    if gap <= 0:
        return math.inf, "closing_longitudinal_overlap"
    return closing * closing / (2 * gap), "applicable"


def pair_risk(first, second, *, ttc_scale=1.5, drac_scale=3.0, gap_scale=1.0,
              horizon=6.0, road_heading=0.0):
    if min(ttc_scale, drac_scale, gap_scale) <= 0:
        raise ValueError("scales must be positive")
    ttc = rectangle_ttc(first, second, horizon)
    drac, status = longitudinal_drac(first, second, road_heading)
    distance = float(Polygon(first.corners()).distance(Polygon(second.corners())))
    components = [0.0 if math.isinf(ttc) else ttc_scale / (ttc + ttc_scale),
                  1.0 if math.isinf(drac) else drac / (drac + drac_scale),
                  gap_scale / (distance + gap_scale)]
    risk = math.sqrt(sum(c * c for c in components) / 3)
    if not np.isfinite(risk) or not 0 <= risk <= 1:
        raise ValueError("nonfinite or out-of-range SSM")
    return {"risk": risk, "components": components, "ttc": None if math.isinf(ttc) else ttc,
            "ttc_status": "no_predicted_contact" if math.isinf(ttc) else "contact_predicted",
            "drac": None if math.isinf(drac) else drac, "drac_status": status,
            "body_distance": distance}


class PassiveRiskObserver:
    """Reads actor states after reset and each original physics step; never acts."""
    def __init__(self, measurement):
        self.options = {"ttc_scale": measurement["ttc_scale_s"],
                        "drac_scale": measurement["drac_scale_mps2"],
                        "gap_scale": measurement["gap_scale_m"],
                        "horizon": measurement["ttc_prediction_horizon_s"]}
        self.frames, self.errors = [], []
        self.peak = None

    def observe(self, env):
        try:
            bodies = {role: Body(np.array(v.position, copy=True), np.array(v.velocity, copy=True),
                                 float(v.heading), float(v.LENGTH), float(v.WIDTH))
                      for role, v in env.actors.items()}
            stamp = env.steps / 20.0
            road_heading = float(env.vehicle.lane.heading_at(
                env.vehicle.lane.local_coordinates(env.vehicle.position)[0]))
            frame = {"time_s": stamp, "actors": {
                role: {"position": b.position.tolist(), "velocity": b.velocity.tolist(),
                       "heading": b.heading, "length": b.length, "width": b.width}
                for role, b in bodies.items()}, "pairs": {}}
            for role, body in bodies.items():
                if role == "ego":
                    continue
                pair = pair_risk(bodies["ego"], body, road_heading=road_heading, **self.options)
                frame["pairs"][role] = pair
                if self.peak is None or pair["risk"] > self.peak["risk"]:
                    self.peak = {**pair, "peak_time_s": stamp, "peak_actor": role}
            self.frames.append(frame)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            self.errors.append(str(exc))

    def summary(self):
        if self.errors or self.peak is None:
            return {"risk": None, "valid_risk": False, "measurement_errors": self.errors or ["no pairs"]}
        return {"risk": self.peak["risk"], "valid_risk": True,
                "peak_time_s": self.peak["peak_time_s"], "peak_actor": self.peak["peak_actor"],
                "component_risks_at_peak": self.peak["components"],
                "raw_components_at_peak": {k: self.peak[k] for k in
                    ("ttc", "ttc_status", "drac", "drac_status", "body_distance")},
                "initial_risk": max(p["risk"] for p in self.frames[0]["pairs"].values())}
