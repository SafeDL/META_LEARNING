"""Physically build shared target banks, retaining every event and cost."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import importlib.metadata
import json
import multiprocessing as mp
import time

import numpy as np

from highway_sim_env.envs.unified_env import EXECUTION_CONTRACT
from methods.history_guided_testing.config import ROOT as HISTORY_ROOT
from methods.history_guided_testing.io import read_json, read_rows, write_json
from methods.history_guided_testing.measurements import execute

from .config import (
    BOUNDS, BUDGET, CHECKPOINTS, FEEDBACK, INITIAL_QUERIES,
    MEASUREMENT, OUTPUT, POOL_SEEDS, PROFILES, SEEDS, SUT_IDS, WORKERS, sut_spec,
)
from .pools import scene_pool


def measure(arguments):
    sut_id, scene = arguments
    result, _, summary, _, elapsed = execute(scene, MEASUREMENT, spec=sut_spec(sut_id))
    if result["inconclusive"] or not summary["valid_risk"]:
        raise ValueError(f"Invalid event/risk: {sut_id}, {scene['scenario_id']}")
    return {
        "scenario_id": scene["scenario_id"], "risk": summary["risk"],
        "collision": result["ego_collision"], "elapsed_s": elapsed,
        "risk_components": summary["component_risks_at_peak"],
        "initial_risk": summary["initial_risk"],
        "collision_time_s": result["collision_time_s"],
        "collision_type": result["collision_type"],
        "termination_reason": result["termination_reason"],
        "event_times": result["event_times"],
        "ego_runtime": result["ego_runtime"],
        "ego_action_count": result["ego_action_count"],
    }


def protocol():
    return {
        "stage": "development; no significance claim",
        "budget": BUDGET, "checkpoints": CHECKPOINTS, "seeds": SEEDS,
        "pool_seeds": POOL_SEEDS, "scenes_per_family": 1024,
        "shared_coordinates_across_all_suts": True,
        "bounds": BOUNDS, "physics_hz": 20,
        "execution_contract": EXECUTION_CONTRACT, "measurement": MEASUREMENT,
        "feedback": FEEDBACK,
        "feedback_fairness": "All new mechanism controls can read both queried values",
        "historical_bank": str(HISTORY_ROOT / "history/responses.npz"),
        "historical_profiles": [asdict(profile) for profile in PROFILES],
        "target_builds": [asdict(sut_spec(sut)) for sut in SUT_IDS],
        "target_identity_or_parameters_given_to_selector": False,
        "candidate_options": {
            "initial_queries": INITIAL_QUERIES,
            "planning": "working-model branch values for early and terminal discovery",
            "query_labels": "same-execution risk and collision; queries count toward 200",
            "comparison": "pre-query model-value screen; no adaptive-policy certificate",
        },
        "statistical_plan": {
            "co_primary": ["area_200", "recall_200"],
            "aggregation": "paired pool/seed averages within each SUT, then equal SUT weight",
            "independent_algorithm_sample_size": "six entries with related model families",
            "development_results_are_not_confirmation": True,
            "confirmation_gate": "must be declared before a new untouched evaluation",
        },
        "package_versions": {name: importlib.metadata.version(name) for name in (
            "highway-env", "stable-baselines3", "gymnasium", "torch", "numpy")},
    }


def prepare():
    audit = read_json(OUTPUT / "interface_audit.json")
    if not audit["passed"]:
        raise ValueError("Physical bank requires the six-SUT interface audit to pass")
    destination = OUTPUT / "protocol.json"
    declared = protocol()
    if destination.exists():
        # JSON round-trip makes tuple/list equality independent of Python syntax.
        if read_json(destination) != json.loads(json.dumps(declared)):
            raise ValueError("Existing protocol differs; do not mix response banks")
    else:
        write_json(destination, declared)
    started = time.perf_counter()
    total = len(SUT_IDS) * len(POOL_SEEDS) * 2048
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as executor:
        for pool_id in range(len(POOL_SEEDS)):
            scenes = scene_pool(pool_id)
            write_json(OUTPUT / f"pool_{pool_id}" / "scenarios.json", scenes)
            for sut_id in SUT_IDS:
                folder = OUTPUT / f"pool_{pool_id}" / sut_id
                folder.mkdir(parents=True, exist_ok=True)
                rows_path = folder / "measurements.jsonl"
                rows = read_rows(rows_path) if rows_path.exists() else []
                mapping = {row["scenario_id"]: row for row in rows}
                if len(mapping) != len(rows):
                    raise ValueError("Duplicate physical measurements in response bank")
                pending = [(sut_id, scene) for scene in scenes if scene["scenario_id"] not in mapping]
                with rows_path.open("a", encoding="utf-8") as handle:
                    for row in executor.map(measure, pending, chunksize=4):
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        handle.flush()
                        mapping[row["scenario_id"]] = row
                        count = len(mapping)
                        if count % 128 == 0:
                            write_json(OUTPUT / "measurement_progress.json", {
                                "status": "running", "sut_id": sut_id, "pool": pool_id,
                                "completed_in_bank": count, "expected_in_bank": 2048,
                                "expected_target_calls": total,
                                "elapsed_s": time.perf_counter() - started,
                            })
                            print("MEASURE", pool_id, sut_id, count, flush=True)
                ordered = [mapping[scene["scenario_id"]] for scene in scenes]
                np.savez_compressed(
                    folder / "responses.npz",
                    x=[scene["numeric_input"] for scene in scenes],
                    scenario_id=[scene["scenario_id"] for scene in scenes],
                    risk=[row["risk"] for row in ordered],
                    collision=[row["collision"] for row in ordered],
                    risk_components=[row["risk_components"] for row in ordered],
                )
                write_json(folder / "physical_cost.json", {
                    "physical_calls": len(ordered),
                    "simulator_elapsed_s": sum(row["elapsed_s"] for row in ordered),
                    "ego_collisions": sum(row["collision"] for row in ordered),
                    "build": asdict(sut_spec(sut_id)),
                })
    write_json(OUTPUT / "measurement_progress.json", {
        "status": "complete", "target_calls": total,
        "elapsed_s": time.perf_counter() - started,
        "interface_audit_calls_reported_separately": True,
    })


if __name__ == "__main__":
    prepare()
