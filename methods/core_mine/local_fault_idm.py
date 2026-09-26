"""Physically executed, predeclared local IDM fault surrogates for a pilot."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from highway_sim_env.envs.cutin_env import CutInEnv, CutInScenario
from methods.core_mine.idm_revision_pilot import REFERENCE, SEED, _episode, scenarios
from sut_algorithms.highway_env.idm_profiles import ProfiledIDMVehicle


ROOT = Path("results/method_chains/core_mine/studies/local_fault_pilot")
FAULTS = ("merge_blind06", "merge_brake2", "slow_front_brake2")


@dataclass(frozen=True)
class LocalFault:
    name: str

    def __post_init__(self) -> None:
        if self.name not in FAULTS:
            raise ValueError(self.name)


class LocalFaultIDMVehicle(ProfiledIDMVehicle):
    def __init__(self, *args, fault: LocalFault, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fault = fault
        self.merge_seen_at: float | None = None
        self.fault_active_steps = 0

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None) -> float:
        fault = self.fault.name
        if fault.startswith("merge_") and front_vehicle is not None:
            lateral_offset = abs(float(front_vehicle.position[1] - ego_vehicle.position[1]))
            if self.merge_seen_at is None and lateral_offset >= .25:
                self.merge_seen_at = self.elapsed
        if self.merge_seen_at is not None and fault.startswith("merge_"):
            duration = .60 if fault == "merge_blind06" else .80
            if self.elapsed - self.merge_seen_at < duration:
                self.fault_active_steps += 1
                if fault == "merge_blind06":
                    return super().acceleration(ego_vehicle, None, rear_vehicle)
                baseline = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)
                return max(baseline, -2.0)
        baseline = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)
        if fault == "slow_front_brake2" and front_vehicle is not None \
                and float(front_vehicle.speed) < 18.0:
            self.fault_active_steps += 1
            return max(baseline, -2.0)
        return baseline


class LocalFaultCutInEnv(CutInEnv):
    def __init__(self, fault: LocalFault, scenario: CutInScenario) -> None:
        self.fault = fault
        super().__init__(REFERENCE, scenario)

    def _create_ego_vehicle(self, ego_lane, ego_road_lane) -> LocalFaultIDMVehicle:
        return LocalFaultIDMVehicle(
            self.road, ego_road_lane.position(60.0, 0.0),
            heading=ego_road_lane.heading_at(60.0), speed=self.EGO_SPEED,
            target_lane_index=ego_lane, target_speed=self.profile.target_speed,
            profile=self.profile, fault=self.fault)


def _fault_episode(fault: str, scenario: CutInScenario, seed: int) -> dict:
    env = LocalFaultCutInEnv(LocalFault(fault), scenario)
    try:
        env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(1)
        result = env.episode_result()
        ego_collision = bool(env.vehicle.crashed)
        near_miss = not ego_collision and (result.min_ttc < 1.5 or result.min_distance < 1.0)
        return {"ego_collision": ego_collision, "near_miss": bool(near_miss),
                "completed": bool(not terminated and env.time >= env.config["duration"]),
                "min_ttc": float(result.min_ttc),
                "min_clearance": float(result.min_distance),
                "fault_active_steps": int(env.vehicle.fault_active_steps)}
    finally:
        env.close()


def _job(args: tuple[str, list[CutInScenario]]) -> list[dict]:
    fault, pool = args
    return [{"build": fault, "seed": SEED, "index": index, "mode": scene.mode,
             "gap": scene.initial_gap, "relative_speed": scene.relative_speed,
             "timing": scene.timing, "intensity": scene.intensity,
             **_fault_episode(fault, scene, SEED + index)}
            for index, scene in enumerate(pool)]


def main() -> None:
    pool = scenarios()
    source = [_episode(REFERENCE, scene, SEED + index)
              for index, scene in enumerate(pool)]
    safe = np.asarray([row["completed"] and not row["ego_collision"]
                       and not row["near_miss"] for row in source])
    if not safe.all():
        raise RuntimeError("pilot source safety changed from archived reference result")
    with ProcessPoolExecutor(max_workers=3) as executor:
        groups = list(executor.map(_job, [(fault, pool) for fault in FAULTS]))
    rows = [row for group in groups for row in group]
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    variants = []
    for fault in FAULTS:
        group = [row for row in rows if row["build"] == fault]
        by_mode = {mode: int(sum(row["ego_collision"] or row["near_miss"]
                                 for row in group if row["mode"] == mode))
                   for mode in dict.fromkeys(scene.mode for scene in pool)}
        variants.append({"build": fault,
                         "new_failures": int(sum(row["ego_collision"] or row["near_miss"]
                                                 for row in group)),
                         "ego_collisions": int(sum(row["ego_collision"] for row in group)),
                         "activated_scenarios": int(sum(row["fault_active_steps"] > 0
                                                        for row in group)),
                         "failure_modes": [mode for mode, count in by_mode.items() if count],
                         "by_mode": by_mode})
    qualified = sum(len(item["failure_modes"]) >= 2 for item in variants) >= 2
    output = {"seed": SEED, "source_safe_candidates": int(safe.sum()),
              "candidate_scenarios": len(pool), "target_physical_episodes": len(rows),
              "ego_control_frequency_hz": 20, "physics_frequency_hz": 20,
              "gate_passed": qualified, "variants": variants}
    (ROOT / "summary.json").write_text(json.dumps(output, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
