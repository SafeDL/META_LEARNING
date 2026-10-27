"""DriveFuzz penalties used by ScenarioFuzz, with a HighwayEnv signal adapter.

Definitions: AtongWang/ScenarioFuzz commit 49360794846786b133799a62827528bc4644bc62,
src/driving_quality.py and src/fuzz_utils.py. Steering-difference input is corrected
to steering angle; the published call site accidentally passes lateral velocity.
"""

import numpy as np
import skfuzzy
from scipy.signal import butter, filtfilt

OS_GRIDS = (np.arange(0, 55, 5), np.arange(0, 0.55, 0.05), np.arange(0, 50, 5))
OS_TRIANGLES = (
    ((0, 0, 20), (5, 25, 45), (30, 50, 50)),
    ((0, 0, 0.2), (0.05, 0.25, 0.45), (0.3, 0.5, 0.5)),
    ((0, 0, 17.5), (4, 22.5, 41), (27.5, 45, 45)),
)
OS_MEMBERSHIP = tuple(
    tuple(skfuzzy.trimf(grid, shape) for shape in shapes)
    for grid, shapes in zip(OS_GRIDS, OS_TRIANGLES)
)
OUTPUT = np.arange(0, 11)
OS_OUTPUT = tuple(skfuzzy.trimf(OUTPUT, shape) for shape in ((0, 0, 4), (1, 5, 9), (6, 10, 10)))
FD_GRID = np.arange(0, 1.1, 0.1)
FD_HIGH = skfuzzy.trimf(FD_GRID, (0, 1, 1))
US_HIGH = skfuzzy.trimf(OUTPUT, (0, 10, 10))


def filtered(values, cutoff):
    b, a = butter(4, cutoff / 10, btype="lowpass")
    return filtfilt(b, a, values)


def centroid(universe, membership):
    return float(skfuzzy.defuzz(universe, membership, "centroid")) if np.any(membership) else 0.0


def oversteer_level(steering_difference, lateral_difference, yaw_rate):
    levels = [
        [skfuzzy.interp_membership(grid, membership, value) for membership in memberships]
        for grid, memberships, value in zip(
            OS_GRIDS, OS_MEMBERSHIP, (steering_difference, lateral_difference, yaw_rate)
        )
    ]
    aggregate = np.zeros_like(OUTPUT, dtype=float)
    for j, membership in enumerate(OS_OUTPUT):
        strength = max(min(levels[0][j], levels[1][j]), levels[2][j])
        aggregate = np.maximum(aggregate, np.minimum(strength, membership))
    return centroid(OUTPUT, aggregate)


def understeer_level(fraction_drop):
    strength = skfuzzy.interp_membership(FD_GRID, FD_HIGH, fraction_drop)
    return centroid(OUTPUT, np.minimum(strength, US_HIGH))


def penalties(
    speed_kmh, longitudinal_kmh, lateral_kmh, steering_deg, yaw_rate_deg_s, minimum_distance
):
    speed, longitudinal, lateral, steering, yaw = map(
        np.asarray, (speed_kmh, longitudinal_kmh, lateral_kmh, steering_deg, yaw_rate_deg_s)
    )
    if len(speed) < 17 or any(len(a) != len(speed) for a in (longitudinal, lateral, steering, yaw)):
        raise ValueError(
            "DriveFuzz filtering requires aligned signals and at least 17 physics frames"
        )
    acceleration = np.diff(speed)
    vx = filtered(longitudinal, 3.5)
    lateral_acceleration = 20 * np.diff(lateral) * 0.028325450360498
    light = filtered(lateral_acceleration, 3.5)
    lateral_difference = np.abs(light - filtered(light, 0.5))
    lateral_heavy = np.abs(filtered(np.r_[lateral_acceleration, 0], 0.5))
    steering_light = filtered(steering, 3.5)
    steering_difference = np.abs(steering_light - filtered(steering_light, 0.5))
    steering_heavy = np.abs(filtered(steering, 0.5))
    gain = np.divide(
        lateral_heavy, steering_heavy, out=np.zeros_like(steering_heavy), where=steering_heavy > 0
    )
    peaks, peak = np.zeros_like(gain), 0
    for i, change in enumerate(np.diff(gain)):
        if change >= 0:
            peak = gain[i]
        peaks[i] = peak
    fraction_drop = np.divide(gain - peaks, peaks, out=np.zeros_like(peaks), where=peaks > 0)
    yaw = np.abs(filtered(yaw, 0.5))
    distance = int(minimum_distance)
    deductions = {
        "hard_acceleration": int(np.count_nonzero(acceleration >= 21.2 / 20)),
        "hard_braking": int(np.count_nonzero(acceleration <= -21.2 / 20)),
        "hard_turn": int(
            np.count_nonzero(
                np.abs(
                    np.divide(
                        lateral, steering, out=np.zeros_like(lateral), where=np.abs(steering) > 20
                    )
                )
                > 0.18
            )
        ),
        "minimum_distance": 0.0 if distance > 100 or distance == 0 else 1 / distance,
        "oversteer": 0,
        "understeer": 0,
    }
    for i in range(len(lateral) - 2):
        if (
            oversteer_level(steering_difference[i], lateral_difference[i], yaw[i]) >= 4
            and vx[i] > 5
            and lateral_difference[i] > 0.1
        ):
            deductions["oversteer"] += 1
        if understeer_level(fraction_drop[i]) >= 4 and vx[i] > 5 and steering_heavy[i] > 10:
            deductions["understeer"] += 1
    return -float(sum(deductions.values())), deductions


def trajectory_score(trajectory):
    frames = trajectory
    actors = [{actor["actor_id"]: actor for actor in frame["actors"]} for frame in frames]
    times = np.array([frame["time_s"] for frame in frames])
    if len(times) < 17 or not np.allclose(np.diff(times), 0.05, atol=1e-8):
        raise ValueError(
            "HighwayEnv driving quality requires the recorded 20 Hz physics trajectory"
        )
    ego = [row["ego"] for row in actors]
    positions = np.array([[actor["x"], actor["y"]] for actor in ego])
    heading = np.unwrap(np.array([actor["heading"] for actor in ego]))
    speed = np.array([actor["speed"] for actor in ego])
    velocity = np.vstack(
        (
            [speed[0] * np.cos(heading[0]), speed[0] * np.sin(heading[0])],
            np.diff(positions, axis=0) / np.diff(times)[:, None],
        )
    )
    direction = np.r_[heading[0], heading[:-1]]
    longitudinal = (
        np.abs(velocity[:, 0] * np.cos(direction) + velocity[:, 1] * np.sin(direction)) * 3.6
    )
    lateral = np.abs(-velocity[:, 0] * np.sin(direction) + velocity[:, 1] * np.cos(direction)) * 3.6
    steering = np.degrees([frame["control"]["steering"] for frame in frames])
    yaw = np.r_[0, np.degrees(np.diff(heading)) / np.diff(times)]
    distance = min(
        np.hypot(actor["x"] - row["ego"]["x"], actor["y"] - row["ego"]["y"])
        for row in actors
        for name, actor in row.items()
        if name != "ego"
    )
    return penalties(speed * 3.6, longitudinal, lateral, steering, yaw, distance)
