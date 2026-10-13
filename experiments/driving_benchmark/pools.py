"""Identical physical scenes for all target SUTs; independent pool replicates."""
import numpy as np
from scipy.stats import qmc

from .config import BOUNDS, POOL_SEEDS, TEMPLATES


def scene_pool(pool_id):
    scenes = []
    for family, (template, seed) in enumerate(zip(TEMPLATES, POOL_SEEDS[pool_id])):
        points = qmc.Sobol(4, scramble=True, seed=seed).random_base2(10)
        bounds = np.asarray(BOUNDS[family])
        names = (
            "initial_clearance_m", "lead_speed_mps",
            "lane_change_time_scale_s" if family == 0 else "lead_deceleration_mps2",
            "event_start_s",
        )
        for index, point in enumerate(points):
            values = bounds[:, 0] + point * (bounds[:, 1] - bounds[:, 0])
            scenes.append({
                "scenario_id": f"representative:{pool_id}:{family}:{index:04d}",
                "template_id": template,
                "simulator_seed": 4179941,
                "fixed_context": {"duration_s": 12, "ego_speed_mps": 25,
                                  "desired_speed_mps": 23, "lane_count": 2},
                "active_parameters": dict(zip(names, values.tolist())),
                "numeric_input": [*point.tolist(), family],
                "sampling_kind": "scrambled_sobol", "sampling_seed": seed,
            })
    return scenes
