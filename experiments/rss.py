"""ADS-centred RSS distance robustness on recorded straight-road trajectories.

RSS Lemmas 2/4 supply the distance envelopes; max/min supply the quantitative
semantics of separation / all actors / all sampled times. This is a distance
monitor, not the full RSS proper-response or responsibility checker. See
docs/rss.md for the equations, parameter assumptions and scope.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RSSParameters:
    ego_response_s: float = 1.0
    other_response_s: float = 2.0
    acceleration_mps2: float = 3.5
    braking_min_mps2: float = 4.0
    braking_max_mps2: float = 8.0
    lateral_acceleration_mps2: float = 0.2
    lateral_braking_mps2: float = 0.8
    lateral_buffer_m: float = 0.1

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("RSS parameters must be finite and nonnegative")
        if not 0 < self.braking_min_mps2 <= self.braking_max_mps2:
            raise ValueError("RSS requires 0 < minimum braking <= maximum braking")
        if self.lateral_braking_mps2 <= 0:
            raise ValueError("Lateral braking must be positive")


DEFAULT = RSSParameters()


def measurement_contract():
    """One fixed reference for all ADS, NPC roles and functional scenarios.

    These are normative response assumptions, not fitted ADS-specific dynamics.
    In particular the lateral acceleration allowance is not a physical bound.
    """
    return {
        "measure": "ego-normalized-rss-distance-margin",
        "parameters": asdict(DEFAULT),
        "actor_scope": "ego-versus-each-other-object; exclude NPC-NPC pairs",
        "geometry": "road-axis enclosing body projections",
        "normalization": "directional centre separation / required centre separation - 1",
        "aggregation": "max over directions, min over ego-object pairs and recorded times",
        "orientation": "larger margin means safer; risk is negative margin",
        "response_assumption": "Intel fixed roles: evaluated ADS 1 s; every other object 2 s",
        "lateral_stopping": "only inward velocity after response; original paper buffer floor",
    }


def rss_values(margins):
    """Keep the physical zero and sign; never apply the old clearance logarithm."""
    values = np.asarray(margins, dtype=float)
    if not np.isfinite(values).all() or (values < -1 - 1e-12).any():
        raise ValueError("RSS model responses must be finite normalized margins >= -1")
    return values


def longitudinal_distance(rear_speed, front_speed, parameters=DEFAULT, *, rear_is_ego=True):
    """RSS Lemma 2, same driving direction, including stationary obstacles."""
    if rear_speed < -1e-9 or front_speed < -1e-9:
        raise ValueError("This monitor covers nonnegative road-longitudinal velocities")
    rear_speed, front_speed = max(0.0, rear_speed), max(0.0, front_speed)
    p = parameters
    response = p.ego_response_s if rear_is_ego else p.other_response_s
    after_response = rear_speed + response * p.acceleration_mps2
    return max(
        0.0,
        rear_speed * response
        + 0.5 * p.acceleration_mps2 * response**2
        + after_response**2 / (2 * p.braking_min_mps2)
        - front_speed**2 / (2 * p.braking_max_mps2),
    )


def lateral_distance(
    lower_speed, upper_speed, parameters=DEFAULT, *, lower_is_ego=True, upper_is_ego=False
):
    """RSS lateral response with directional stopping, as in ad-rss-lib.

    Only inward post-response velocity needs an additional stopping distance.
    The original paper's positive buffer floor is retained when moving apart.
    """
    p = parameters
    lower_response = p.ego_response_s if lower_is_ego else p.other_response_s
    upper_response = p.ego_response_s if upper_is_ego else p.other_response_s
    lower_after = lower_speed + lower_response * p.lateral_acceleration_mps2
    upper_after = upper_speed - upper_response * p.lateral_acceleration_mps2
    return p.lateral_buffer_m + max(
        0.0,
        (lower_speed + lower_after) * lower_response / 2
        - (upper_speed + upper_after) * upper_response / 2
        + (max(lower_after, 0.0) ** 2 + min(upper_after, 0.0) ** 2) / (2 * p.lateral_braking_mps2),
    )


def road_projection(actor):
    c, s = math.cos(actor["heading"]), math.sin(actor["heading"])
    return (
        actor["speed"] * c,
        actor["speed"] * s,
        (abs(c) * actor["length"] + abs(s) * actor["width"]) / 2,
        (abs(s) * actor["length"] + abs(c) * actor["width"]) / 2,
    )


def pair_margin(ego, actor, parameters=DEFAULT):
    """Four directed separation predicates avoid jumps when actor order changes.

    Road-axis projections enclose the oriented vehicle bodies. Negative values
    mean all four separation predicates fail. The identity of the ego does not
    determine which vehicle is rear, front, left or right.
    """
    evx, evy, ex, ey = road_projection(ego)
    avx, avy, ax, ay = road_projection(actor)
    ego_role, actor_role = ego["actor_id"] == "ego", actor["actor_id"] == "ego"
    dx, dy = actor["x"] - ego["x"], actor["y"] - ego["y"]
    envelopes = {
        "actor_ahead": ex + ax + longitudinal_distance(evx, avx, parameters, rear_is_ego=ego_role),
        "actor_behind": ex
        + ax
        + longitudinal_distance(avx, evx, parameters, rear_is_ego=actor_role),
        "actor_positive_y": ey
        + ay
        + lateral_distance(evy, avy, parameters, lower_is_ego=ego_role, upper_is_ego=actor_role),
        "actor_negative_y": ey
        + ay
        + lateral_distance(avy, evy, parameters, lower_is_ego=actor_role, upper_is_ego=ego_role),
    }
    separations = dict(zip(envelopes, (dx, -dx, dy, -dy)))
    # Normalize by the required centre separation, including body dimensions.
    # This stays finite at zero safe distance and prevents the lateral width
    # from clipping every close-following longitudinal score to one constant.
    margins = {key: separations[key] / size - 1 for key, size in envelopes.items()}
    return {
        "actor_id": actor["actor_id"],
        "margin": max(margins.values()),
        "directional_margins": margins,
        "required_center_separation_m": envelopes,
    }


def frame_margin(frame, parameters=DEFAULT):
    ego = next(a for a in frame["actors"] if a["actor_id"] == "ego")
    pairs = [pair_margin(ego, a, parameters) for a in frame["actors"] if a["actor_id"] != "ego"]
    worst = min(pairs, key=lambda p: p["margin"], default=None)
    return {
        "time_s": frame["time_s"],
        "margin": worst["margin"] if worst else math.inf,
        "critical_actor": worst["actor_id"] if worst else None,
        "pairs": pairs,
    }


def trajectory_margin(trajectory, parameters=DEFAULT):
    if not trajectory:
        raise ValueError("A safety margin requires recorded trajectory samples")
    series = [frame_margin(frame, parameters) for frame in trajectory]
    worst = min(series, key=lambda row: row["margin"])
    negative = [r for r in series if r["margin"] < 0]
    return {
        "minimum_margin": worst["margin"],
        "risk": -worst["margin"],
        "critical_time_s": worst["time_s"],
        "critical_actor": worst["critical_actor"],
        "first_negative_time_s": negative[0]["time_s"] if negative else None,
        "initial_margin": series[0]["margin"],
        "series": series,
    }
