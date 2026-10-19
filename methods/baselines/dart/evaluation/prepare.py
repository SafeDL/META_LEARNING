"""Measure the untouched pool only after sources, weights and protocol freeze."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
import time

import numpy as np
from scipy.stats import qmc

from benchmark.data import SCENARIOS
from benchmark.config import (
    BOUNDS,
    OUTPUT,
    TEMPLATES,
    WORKERS,
    sut_spec,
)
from benchmark.prepare import measure
from methods.baselines.hgt.io import read_json, read_rows, write_json

from .config import INDEPENDENT_BANKS, INDEPENDENT_SEEDS, ROOT, SUT_IDS
from .freeze import verify_freeze


def independent_scenes():
    scenes = []
    for family, (template, seed) in enumerate(zip(TEMPLATES, INDEPENDENT_SEEDS)):
        points = qmc.Sobol(4, scramble=True, seed=seed).random_base2(10)
        bounds = np.asarray(BOUNDS[family])
        names = (
            "initial_clearance_m",
            "lead_speed_mps",
            "lane_change_time_scale_s" if family == 0 else "lead_deceleration_mps2",
            "event_start_s",
        )
        for index, point in enumerate(points):
            values = bounds[:, 0] + point * (bounds[:, 1] - bounds[:, 0])
            scenes.append(
                {
                    "scenario_id": f"independent:2:{family}:{index:04d}",
                    "template_id": template,
                    "simulator_seed": 4179941,
                    "fixed_context": {
                        "duration_s": 12,
                        "ego_speed_mps": 25,
                        "desired_speed_mps": 23,
                        "lane_count": 2,
                    },
                    "active_parameters": dict(zip(names, values.tolist())),
                    "numeric_input": [*point.tolist(), family],
                    "sampling_kind": "scrambled_sobol",
                    "sampling_seed": seed,
                }
            )
    return scenes


def main():
    protocol = verify_freeze()
    audit = read_json(OUTPUT / "interface_audit.json")
    if not audit["passed"]:
        raise ValueError("Existing six-system execution interface audit did not pass")
    current = json.loads(json.dumps([asdict(sut_spec(s)) for s in SUT_IDS]))
    if current != protocol["target_builds"]:
        raise ValueError("Target builds differ from the frozen confirmation protocol")
    scenes = independent_scenes()
    x = np.asarray([s["numeric_input"] for s in scenes])
    for pool in (0, 1):
        with np.load(SCENARIOS / f"pool_{pool}" / SUT_IDS[0] / "responses.npz") as old:
            old_coordinates = set(map(tuple, old["x"]))
            if any(tuple(row) in old_coordinates for row in x):
                raise ValueError(
                    "Independent pool contains an existing development coordinate"
                )
    write_json(INDEPENDENT_BANKS / "pool_2" / "scenarios.json", scenes)
    started = time.perf_counter()
    completed = 0
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=mp.get_context("spawn")
    ) as executor:
        for sut in SUT_IDS:
            folder = INDEPENDENT_BANKS / "pool_2" / sut
            folder.mkdir(parents=True, exist_ok=True)
            measurements = folder / "measurements.jsonl"
            rows = read_rows(measurements) if measurements.exists() else []
            mapping = {row["scenario_id"]: row for row in rows}
            if len(mapping) != len(rows):
                raise ValueError(
                    "Independent measurements contain duplicate scenario IDs"
                )
            pending = [(sut, s) for s in scenes if s["scenario_id"] not in mapping]
            with measurements.open("a", encoding="utf-8") as handle:
                for row in executor.map(measure, pending, chunksize=4):
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    mapping[row["scenario_id"]] = row
                    if len(mapping) % 128 == 0:
                        write_json(
                            ROOT / "measurement_status.json",
                            {
                                "status": "running",
                                "sut": sut,
                                "completed_in_system": len(mapping),
                                "completed_calls": completed + len(mapping),
                                "planned_calls": 12288,
                                "elapsed_s": time.perf_counter() - started,
                            },
                        )
                        print(
                            f"MEASURE independent {sut} {len(mapping)}/2048", flush=True
                        )
            ordered = [mapping[s["scenario_id"]] for s in scenes]
            np.savez_compressed(
                folder / "responses.npz",
                x=x,
                scenario_id=[s["scenario_id"] for s in scenes],
                risk=[r["risk"] for r in ordered],
                collision=[r["collision"] for r in ordered],
                risk_components=[r["risk_components"] for r in ordered],
            )
            write_json(
                folder / "physical_cost.json",
                {
                    "physical_calls": len(ordered),
                    "simulator_elapsed_s": sum(r["elapsed_s"] for r in ordered),
                    "ego_collisions": sum(r["collision"] for r in ordered),
                    "build": asdict(sut_spec(sut)),
                },
            )
            completed += len(ordered)
    write_json(
        ROOT / "measurement_status.json",
        {
            "status": "complete",
            "completed_calls": completed,
            "planned_calls": 12288,
            "elapsed_s": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
