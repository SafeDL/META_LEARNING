"""Complete development pools, including outcomes withheld from every selector."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing as mp

import numpy as np

from methods.history_guided_testing.config import TARGET
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.prepare import measure_scene
from methods.history_guided_testing.scenarios import scene_library
from research.history_response_testing.config import CONFIRMATION as PREVIOUS

from .config import OUTPUT

PROFILES = (
    replace(TARGET, name="nominal"),
    replace(TARGET,
            name="delayed_fvdm",
            max_brake=7,
            desired_gap=7,
            perception_delay_s=0.3),
    replace(TARGET,
            name="limited_fvdm",
            max_brake=4,
            desired_gap=8,
            perception_delay_s=0.3),
    replace(TARGET,
            name="delayed_idm",
            controller="IDM",
            max_brake=7,
            desired_gap=8,
            perception_delay_s=0.2),
)


def main():
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=12, mp_context=context) as executor:
        for profile in PROFILES:
            folder = OUTPUT / "development_pools" / profile.name
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "responses.npz").exists():
                print("Full development pool already saved",
                      profile.name,
                      flush=True)
                continue
            scenes = (read_json(PREVIOUS / "pool_0" / "scenarios.json") if
                      profile.name == "nominal" else scene_library("target"))
            write_json(folder / "scenarios.json", scenes)
            cache = {}
            previous_rows = PREVIOUS / "pool_0" / "measurements.jsonl"
            if profile.name == "nominal":
                cache = {
                    row["index"]: row
                    for row in map(
                        json.loads,
                        previous_rows.read_text(encoding="utf-8").splitlines())
                }
            journal = folder / "measurements.jsonl"
            if journal.exists():
                cache.update({
                    row["index"]: row
                    for row in map(
                        json.loads,
                        journal.read_text(encoding="utf-8").splitlines())
                })
            missing = [i for i in range(len(scenes)) if i not in cache]
            arguments = [(scenes[i], profile) for i in missing]
            with journal.open("a", encoding="utf-8") as handle:
                for index, row in zip(
                        missing,
                        executor.map(measure_scene, arguments, chunksize=4)):
                    row["index"] = index
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    cache[index] = row
                    if len(cache) % 128 == 0:
                        print("FULL DEVELOPMENT",
                              profile.name,
                              len(cache),
                              len(scenes),
                              flush=True)
            np.savez_compressed(
                folder / "responses.npz",
                x=[s["numeric_input"] for s in scenes],
                risk=[cache[i]["risk"] for i in range(len(scenes))],
                collision=[cache[i]["collision"] for i in range(len(scenes))])
            write_json(
                folder / "cost.json", {
                    "profile":
                    profile.__dict__,
                    "full_pool_measured":
                    True,
                    "new_physical_calls":
                    len(missing),
                    "pool_collisions":
                    sum(row["collision"] for row in cache.values()),
                    "stage":
                    "development; all outcomes forbidden to selector except queried feedback"
                })
            print("FULL DEVELOPMENT COMPLETE",
                  profile.name,
                  sum(row["collision"] for row in cache.values()),
                  flush=True)


if __name__ == "__main__":
    main()
