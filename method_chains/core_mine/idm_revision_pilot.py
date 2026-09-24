"""Development-only same-family IDM regression-opportunity pilot."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from highway_env_benchmark.envs.cutin_env import CutInEnv, CutInScenario
from sut_algorithms.highway_env.idm_profiles import SUTProfile


SEED = 20291202
ROOT = Path("results/method_chains/core_mine/studies/idm_revision_pilot")
BOUNDS = {
    "fast_intrusion": ((18.0, 45.0), (-8.0, -2.0)),
    "cutin_braking": ((18.0, 50.0), (-7.0, -1.0)),
    "lead_braking": ((20.0, 50.0), (-7.0, -1.0)),
    "stop_and_go": ((18.0, 50.0), (-6.0, -1.0)),
    "slow_lead_following": ((20.0, 50.0), (-8.0, -2.0)),
}
REFERENCE = SUTProfile("idm_ref", "IDM", time_wanted=1.5, max_brake=5.0,
                       reaction_delay=0.0, desired_gap=5.0,
                       comfort_acceleration=3.0, target_speed=27.0)
BUILDS = {
    "idm_ref": REFERENCE,
    "idm_delay07": replace(REFERENCE, name="idm_delay07", reaction_delay=0.7),
    "idm_brake3": replace(REFERENCE, name="idm_brake3", max_brake=3.0),
    "idm_delay07_brake3": replace(REFERENCE, name="idm_delay07_brake3",
                                  reaction_delay=0.7, max_brake=3.0),
}


def scenarios() -> list[CutInScenario]:
    output = []
    for offset, (mode, (gap_bounds, speed_bounds)) in enumerate(BOUNDS.items()):
        unit = qmc.Sobol(d=4, scramble=True, seed=SEED + offset).random_base2(m=4)
        for row in unit:
            gap = gap_bounds[0] + row[0] * (gap_bounds[1] - gap_bounds[0])
            speed = speed_bounds[0] + row[1] * (speed_bounds[1] - speed_bounds[0])
            output.append(CutInScenario(float(gap), float(speed), mode,
                                        float(.05 + .90 * row[2]),
                                        float(.05 + .90 * row[3])))
    return output


def _episode(profile: SUTProfile, scenario: CutInScenario, seed: int) -> dict:
    env = CutInEnv(profile, scenario)
    try:
        env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(1)
        result = env.episode_result()
        ego_collision = bool(env.vehicle.crashed)
        background_collision = any(vehicle.crashed for vehicle in env.road.vehicles
                                   if vehicle is not env.vehicle)
        completed = not terminated and env.time >= env.config["duration"]
        near_miss = not ego_collision and (result.min_ttc < 1.5
                                           or result.min_distance < 1.0)
        return {"ego_collision": ego_collision, "background_collision":
                background_collision, "near_miss": near_miss,
                "completed": bool(completed), "min_ttc": result.min_ttc,
                "min_clearance": result.min_distance}
    finally:
        env.close()


def _job(build: str, pool: list[CutInScenario]) -> list[dict]:
    rows = []
    for index, scenario in enumerate(pool):
        result = _episode(BUILDS[build], scenario, SEED + index)
        rows.append({"build": build, "seed": SEED, "index": index,
                     "mode": scenario.mode, "gap": scenario.initial_gap,
                     "relative_speed": scenario.relative_speed,
                     "timing": scenario.timing, "intensity": scenario.intensity,
                     **result})
    return rows


def main() -> None:
    pool = scenarios()
    if len(pool) != 80 or len({scenario.mode for scenario in pool}) != 5:
        raise RuntimeError("pilot pool must contain 16 scenarios in five modes")
    ROOT.mkdir(parents=True, exist_ok=True)
    records_path = ROOT / "records.csv"
    if records_path.exists():
        raise RuntimeError("existing frozen pilot records are preserved")
    with ProcessPoolExecutor(max_workers=len(BUILDS)) as executor:
        batches = list(executor.map(_job, BUILDS, [pool] * len(BUILDS)))
    rows = [row for batch in batches for row in batch]
    by_build = {build: {row["index"]: row for row in batch}
                for build, batch in zip(BUILDS, batches, strict=True)}
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    eligible = {index for index, row in by_build["idm_ref"].items()
                if row["completed"] and not row["ego_collision"]
                and not row["near_miss"]}
    variants = []
    for build in BUILDS:
        failures = [by_build[build][index] for index in eligible
                    if by_build[build][index]["ego_collision"]
                    or by_build[build][index]["near_miss"]]
        variants.append({"build": build, "new_failures": len(failures),
                         "ego_collisions": sum(row["ego_collision"] for row in failures),
                         "failure_modes": sorted({row["mode"] for row in failures}),
                         "by_mode": {mode: sum(row["mode"] == mode for row in failures)
                                     for mode in BOUNDS}})
    payload = {"seed": SEED, "status": "development_only", "physics_frequency_hz": 20,
               "controller_internal_frequency_hz": 20,
               "candidate_scenarios": len(pool), "physical_episodes": len(rows),
               "reference_safe_candidates": len(eligible),
               "reference_safe_modes": sorted({pool[index].mode for index in eligible}),
               "frozen_builds": {name: {"reaction_delay": profile.reaction_delay,
                                         "max_brake": profile.max_brake}
                                 for name, profile in BUILDS.items()},
               "variants": variants}
    (ROOT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
