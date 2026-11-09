"""Named highway scenario domains and label-independent coordinate splits."""

from dataclasses import dataclass
from itertools import product
from math import ceil, prod

import numpy as np
from scipy.stats import qmc

from .schema import Scenario


@dataclass(frozen=True)
class Domain:
    bounds: dict[str, tuple[float, float]]
    quantization: dict[str, float]
    categories: dict[str, tuple[str, ...]]

    @property
    def dimension(self):
        return len(self.bounds) + sum(len(v) for v in self.categories.values())


DOMAINS = {
    "front_brake": Domain(
        {
            "ego_speed_mps": (18, 34),
            "gap_m": (8, 65),
            "closing_speed_mps": (-4, 8),
            "deceleration_mps2": (2, 9),
            "event_time_s": (0.5, 3),
            "brake_ramp_s": (0.1, 1.5),
        },
        {
            "ego_speed_mps": 1,
            "gap_m": 2,
            "closing_speed_mps": 1,
            "deceleration_mps2": 0.5,
            "event_time_s": 0.2,
            "brake_ramp_s": 0.1,
        },
        {},
    ),
    "cut_in": Domain(
        {
            "ego_speed_mps": (18, 34),
            "gap_m": (5, 50),
            "closing_speed_mps": (-4, 10),
            "lateral_duration_s": (1.8, 4),
            "event_time_s": (0.5, 3),
        },
        {
            "ego_speed_mps": 1,
            "gap_m": 2,
            "closing_speed_mps": 1,
            "lateral_duration_s": 0.2,
            "event_time_s": 0.2,
        },
        {"side": ("left", "right")},
    ),
    "cut_out": Domain(
        {
            "ego_speed_mps": (18, 34),
            "gap_m": (8, 50),
            "lead_speed_mps": (14, 32),
            "obstacle_distance_m": (65, 160),
            "event_time_s": (0.5, 2),
            "lateral_duration_s": (1.8, 4),
        },
        {
            "ego_speed_mps": 1,
            "gap_m": 2,
            "lead_speed_mps": 1,
            "obstacle_distance_m": 3,
            "event_time_s": 0.2,
            "lateral_duration_s": 0.2,
        },
        {},
    ),
    "lane_change_rear": Domain(
        {
            "ego_speed_mps": (18, 34),
            "lead_gap_m": (10, 55),
            "lead_speed_mps": (10, 26),
            "rear_gap_m": (5, 50),
            "rear_speed_mps": (22, 38),
            "adjacent_front_gap_m": (15, 90),
        },
        {
            "ego_speed_mps": 1,
            "lead_gap_m": 2,
            "lead_speed_mps": 1,
            "rear_gap_m": 2,
            "rear_speed_mps": 1,
            "adjacent_front_gap_m": 3,
        },
        {},
    ),
}

DOMAINS["cut_in_interaction"] = Domain(
    dict(DOMAINS["cut_in"].bounds),
    dict(DOMAINS["cut_in"].quantization),
    {"side": ("left", "right"), "adjacent_space": ("open", "restricted")},
)

# Frozen experiments explicitly enumerate this set instead of following new domains.
RECORDED_SCENARIOS = tuple(DOMAINS)
RECORDED_DOMAINS = dict(DOMAINS)

DOMAINS["car_following"] = Domain(
    {"ego_speed_mps": (18, 34), "lead_speed_mps": (8, 34), "gap_m": (8, 65)},
    {"ego_speed_mps": 1, "lead_speed_mps": 1, "gap_m": 2},
    {},
)
CUT_IN_SCENARIOS = ("cut_in", "cut_in_interaction")
TEST_SCENARIOS = ("car_following", "front_brake", "cut_out", "lane_change_rear")


SPLIT_SEEDS = {
    "policy_train": 11003,
    "qualification": 21001,
    "history": 31013,
    "development": 41011,
    "new": 61001,
    "hard": 71011,
}


def candidate_count(kind):
    domain = DOMAINS[kind]
    dimensions = len(domain.bounds)
    continuous = 10**dimensions if dimensions <= 3 else 1000 * ceil(dimensions / 3)
    return continuous * prod(len(options) for options in domain.categories.values())


def candidate_pool(kind, count=None, *, replicate=10):
    """Current experiment coordinates; retain every physically feasible difficulty level."""
    if kind not in TEST_SCENARIOS:
        raise ValueError(f"Not a current test scenario: {kind}")
    domain = DOMAINS[kind]
    seed = SPLIT_SEEDS["new"] + 101 * replicate
    dimensions = len(domain.bounds)
    grid = count is None and dimensions <= 3
    count = candidate_count(kind) if count is None else count
    if count <= 0:
        raise ValueError("Candidate count must be positive")
    if grid:
        generator = np.random.default_rng(seed)
        points = product(*((np.arange(10) + generator.random(10)) / 10 for _ in domain.bounds))
    else:
        sampler = qmc.Sobol(dimensions, scramble=True, seed=seed)

        def sobol_points():
            while True:
                yield from sampler.random(1024)

        points = sobol_points()
    output = []
    for index, point in enumerate(points):
        parameters = {
            name: float(low + value * (high - low))
            for (name, (low, high)), value in zip(domain.bounds.items(), point)
        }
        scene = Scenario(f"{kind}:candidate:{replicate}:{index:06d}", kind, parameters, {})
        try:
            validate(scene)
        except ValueError:
            continue
        output.append(scene)
        if len(output) == count:
            break
    return output


def difficult(scenario):
    p = scenario.parameters
    if scenario.kind == "front_brake":
        return p["gap_m"] <= 20 and p["deceleration_mps2"] >= 7
    if scenario.kind in ("cut_in", "cut_in_interaction"):
        return p["gap_m"] <= 15 and p["lateral_duration_s"] <= 2.2
    if scenario.kind == "cut_out":
        return p["obstacle_distance_m"] <= 95 and p["event_time_s"] >= 1.5
    if scenario.kind == "lane_change_rear":
        return p["rear_gap_m"] <= 15 and p["rear_speed_mps"] - p["ego_speed_mps"] >= 6
    if scenario.kind == "car_following":
        return p["gap_m"] <= 20 and p["ego_speed_mps"] - p["lead_speed_mps"] >= 6
    raise ValueError(f"Unknown functional scenario: {scenario.kind}")


def validate(scenario):
    domain = DOMAINS[scenario.kind]
    if set(scenario.parameters) != set(domain.bounds):
        raise ValueError("Scenario must contain exactly the named domain parameters")
    if set(scenario.categories) != set(domain.categories):
        raise ValueError("Scenario categories differ from its domain")
    for name, (low, high) in domain.bounds.items():
        if not low <= scenario.parameters[name] <= high:
            raise ValueError(f"Parameter outside physical domain: {name}")
    for name, options in domain.categories.items():
        if scenario.categories[name] not in options:
            raise ValueError(f"Invalid categorical condition: {name}")
    p = scenario.parameters
    if scenario.kind == "cut_out":
        # The scripted lead must clear the obstacle before reaching it.
        if (
            p["gap_m"] + 5 + p["lead_speed_mps"] * (p["event_time_s"] + p["lateral_duration_s"]) + 5
            >= p["obstacle_distance_m"]
        ):
            raise ValueError("Lead cannot clear the obstacle in the specified time")


def features(scenarios):
    if not scenarios or len({s.kind for s in scenarios}) != 1:
        raise ValueError("Build one nonempty functional scenario task at a time")
    domain = DOMAINS[scenarios[0].kind]
    rows = []
    for scenario in scenarios:
        validate(scenario)
        row = [(scenario.parameters[n] - lo) / (hi - lo) for n, (lo, hi) in domain.bounds.items()]
        row.extend(
            float(scenario.categories[n] == v)
            for n, options in domain.categories.items()
            for v in options
        )
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def sample(kind, count, split, *, replicate=0):
    """Generate coordinates only; neither SUT identity nor outcomes enter this function."""
    if kind == "cut_in_interaction":
        base = sample("cut_in", (count + 1) // 2, split, replicate=replicate)
        return [
            Scenario(
                f"{kind}:{split}:{replicate}:{2 * i + j:06d}",
                kind,
                dict(scene.parameters),
                {**scene.categories, "adjacent_space": condition},
            )
            for i, scene in enumerate(base)
            for j, condition in enumerate(("open", "restricted"))
        ][:count]
    domain = DOMAINS[kind]
    seed = SPLIT_SEEDS["history" if split == "reuse" else split] + 101 * replicate
    sampler = qmc.Sobol(len(domain.bounds) + len(domain.categories), scramble=True, seed=seed)
    output = []
    index = 0
    # Use balanced Sobol blocks; rejected physically impossible points have no labels.
    while len(output) < count:
        for point in sampler.random_base2(10) if index == 0 else sampler.random(1024):
            p = {
                n: float(lo + v * (hi - lo))
                for (n, (lo, hi)), v in zip(domain.bounds.items(), point)
            }
            c = {
                n: options[min(int(v * len(options)), len(options) - 1)]
                for (n, options), v in zip(domain.categories.items(), point[len(p) :])
            }
            scenario = Scenario(f"{kind}:{split}:{replicate}:{index:06d}", kind, p, c)
            index += 1
            try:
                validate(scenario)
            except ValueError:
                continue
            if split in ("history", "development", "reuse") and difficult(scenario):
                continue
            if split == "hard" and not difficult(scenario):
                continue
            # The main ID condition shares the ordinary history distribution.
            if split == "new" and difficult(scenario):
                continue
            output.append(scenario)
            if len(output) == count:
                break
    return output
