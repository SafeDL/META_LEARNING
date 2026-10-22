"""MAP-Elites: uniform occupied-cell selection, mutation and within-cell competition."""

import numpy as np


CONFIG = {"bins": 10, "mutation_step": [0.01, 0.1]}
AXES = {"front_brake": ("ego_speed_mps", "gap_m"), "cut_in": ("ego_speed_mps", "gap_m"),
        "cut_out": ("ego_speed_mps", "obstacle_distance_m"),
        "lane_change_rear": ("ego_speed_mps", "rear_gap_m")}


class DeepHyperion:
    def __init__(self, space, *, config=CONFIG):
        self.space = space
        self.config = config
        self.archive = {}

    def cell(self, scenario):
        values = []
        for name in AXES[scenario.kind]:
            low, high = self.space.domain.bounds[name]
            values.append(min(self.config["bins"] - 1, int(
                self.config["bins"] * (scenario.parameters[name] - low) / (high - low))))
        return tuple(values)

    @staticmethod
    def quality(feedback):
        safety = feedback.safety
        return (2 * int(feedback.collision) + 0.5 / (1 + max(0, safety.min_clearance_m))
                + 0.5 / (1 + safety.min_ttc_s))

    def propose(self, remaining):
        occupied = sorted(self.archive)
        cell = occupied[int(self.space.generator.integers(len(occupied)))]
        parent = self.archive[cell][1]
        return self.space.mutate(parent, step=self.config["mutation_step"]), None, {
            "search": "map_elites", "parent_cell": list(cell), "occupied_cells": len(occupied)}

    def observe(self, scenario, feedback):
        cell, quality = self.cell(scenario), self.quality(feedback)
        if cell not in self.archive or quality > self.archive[cell][0]:
            self.archive[cell] = (quality, scenario)
        return {"occupied_cells": len(self.archive)}
