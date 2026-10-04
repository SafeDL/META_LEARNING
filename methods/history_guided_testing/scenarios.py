"""Two scenario families, each containing exactly 1024 candidates."""
import numpy as np
from scipy.stats import qmc

from .config import BOUNDS, TEMPLATES, TARGET_SAMPLING_SEEDS


def scene_library(kind):
    scenes = []
    for family, template in enumerate(TEMPLATES):
        if kind == "target":
            points = qmc.Sobol(4, scramble=True, seed=TARGET_SAMPLING_SEEDS[family]).random_base2(10)
            seeds = [TARGET_SAMPLING_SEEDS[family]] * 1024
        elif kind == "history":
            points = np.vstack([qmc.Sobol(4, scramble=True, seed=seed + family).random_base2(9)
                                for seed in (74001, 74221)])
            seeds = [74001 + family] * 512 + [74221 + family] * 512
        else:
            raise ValueError("unknown scenario library")
        names = ("initial_clearance_m", "lead_speed_mps",
                 "lane_change_time_scale_s" if family == 0 else "lead_deceleration_mps2",
                 "event_start_s")
        bounds = np.asarray(BOUNDS[family])
        for index, point in enumerate(points):
            values = bounds[:, 0] + point * (bounds[:, 1] - bounds[:, 0])
            scenes.append({
                "scenario_id": f"{kind}:{family}:{index:04d}", "template_id": template,
                "simulator_seed": 4179941,
                "fixed_context": {"duration_s": 12, "ego_speed_mps": 25,
                                  "desired_speed_mps": 23, "lane_count": 2},
                "active_parameters": dict(zip(names, values.tolist())),
                "numeric_input": [*point.tolist(), family],
                "sampling_kind": "scrambled_sobol", "sampling_seed": seeds[index],
            })
    return scenes


def parameter_cells(x):
    x = np.asarray(x)
    digits = np.minimum((x[:, :4] * 4).astype(int), 3)
    return x[:, 4].astype(int) * 256 + np.ravel_multi_index(digits.T, (4,) * 4)
