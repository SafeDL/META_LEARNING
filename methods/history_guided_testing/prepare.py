"""Prepare two scenario libraries and physically measure missing response banks."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp

import numpy as np

from methods.ras_frt_uq.unified import SIMILARITY, TRAINING as RAS_TRAINING
from .config import (BASELINES, BUDGET, CHECKPOINTS, FEEDBACK, MEASUREMENT, PROFILES,
                     ROOT, SEEDS, TARGET, TARGET_SAMPLING_SEEDS, TRAINING, WORKERS, build_spec)
from .io import read_json, read_rows, write_json
from .measurements import execute
from .scenarios import scene_library


def measure_scene(arguments):
    scene, profile = arguments
    result, _, summary, _, elapsed = execute(scene, MEASUREMENT, spec=build_spec(profile))
    if result["inconclusive"] or not summary["valid_risk"]:
        raise ValueError(f"invalid measurement: {scene['scenario_id']}")
    return {"scenario_id": scene["scenario_id"], "risk": summary["risk"],
            "collision": bool(result["ego_collision"]), "elapsed_s": elapsed}


def measure_bank(profile, scenes, folder):
    folder.mkdir(parents=True, exist_ok=True)
    pending_path = folder / f"{profile.name}.jsonl"
    rows = read_rows(pending_path) if pending_path.exists() else []
    completed = {row["scenario_id"] for row in rows}
    remaining = [(scene, profile) for scene in scenes if scene["scenario_id"] not in completed]
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn")) as pool:
        with pending_path.open("a", encoding="utf-8") as handle:
            for count, row in enumerate(pool.map(measure_scene, remaining, chunksize=4), len(rows) + 1):
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                rows.append(row)
                if count % 128 == 0:
                    print(f"{profile.name} physical measurements: {count}/2048", flush=True)
    mapping = {row["scenario_id"]: row for row in rows}
    return [mapping[scene["scenario_id"]] for scene in scenes]


def prepare_history(scenes):
    destination = ROOT / "history/responses.npz"
    if destination.exists():
        return
    arrays = {}
    for profile in PROFILES:
        rows = measure_bank(profile, scenes, destination.parent)
        arrays[profile.name] = np.asarray([row["risk"] for row in rows])
        arrays[profile.name + "_collision"] = np.asarray([row["collision"] for row in rows])
    np.savez_compressed(destination, x=[scene["numeric_input"] for scene in scenes], **arrays)
    cost_path = ROOT / "physical_cost.json"
    cost = read_json(cost_path) if cost_path.exists() else {}
    cost.update({"new_history_physical_calls": 12288, "reused_historical_responses": 0})
    write_json(cost_path, cost)
    for profile in PROFILES:
        (destination.parent / f"{profile.name}.jsonl").unlink()


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "protocol.json", {
        "history_scenes_per_family": 1024, "target_scenes_per_family": 1024,
        "historical_train_count": 1638, "historical_validation_count": 410,
        "historical_profiles": [asdict(profile) for profile in PROFILES],
        "target_profile": asdict(TARGET), "seeds": SEEDS, "budget": BUDGET,
        "checkpoints": CHECKPOINTS, "feedback": FEEDBACK, "training": TRAINING,
        "baselines": BASELINES, "measurement": MEASUREMENT,
        "final_target_used_for_training_or_selection": False,
        "development_target_used_to_simplify_acquisition": True,
        "history_sampling_seeds": [[74001, 74221], [74002, 74222]],
        "target_sampling_seeds": TARGET_SAMPLING_SEEDS,
        "kernel_selection": "lowest five-seed historical validation RMSE",
        "acquisition": "fixed pure risk acquisition before final target measurement",
        "selection_feedback": {"ras_frt_uq": "queried binary collision only",
                               "other_methods": "queried continuous risk only"},
        "ras_frt_uq": {"training": RAS_TRAINING, "similarity": SIMILARITY,
                       "sources": "same six historical algorithms",
                       "family_models": "two four-input encoders per seed",
                       "target_labels_used_for_training_or_selection": False,
                       "fusion_parameters": "original RAS-FRT-UQ values, unchanged"},
    })
    historical_scenes = scene_library("history")
    scenes = scene_library("target")
    write_json(ROOT / "history/scenarios.json", historical_scenes)
    prepare_history(historical_scenes)
    x = np.asarray([scene["numeric_input"] for scene in scenes])
    history_x = np.load(ROOT / "history/responses.npz")["x"]
    if not np.array_equal(history_x, [scene["numeric_input"] for scene in historical_scenes]):
        raise ValueError("history responses do not match the scenario library")
    if set(map(tuple, x)) & set(map(tuple, history_x)):
        raise ValueError("historical and target coordinates overlap")
    write_json(ROOT / "target/scenarios.json", scenes)
    output = ROOT / "target/responses.npz"
    if output.exists():
        return
    rows = measure_bank(TARGET, scenes, output.parent)
    np.savez_compressed(output, x=x, risk=[row["risk"] for row in rows],
                        collision=[row["collision"] for row in rows])
    cost_path = ROOT / "physical_cost.json"
    cost = read_json(cost_path) if cost_path.exists() else {}
    cost.setdefault("reused_historical_responses", 12288)
    cost.update({
                 "final_target_physical_calls": len(rows),
                 "target_simulator_seconds": sum(row["elapsed_s"] for row in rows)})
    cost.pop("new_target_physical_calls", None)
    write_json(cost_path, cost)
    (output.parent / f"{TARGET.name}.jsonl").unlink()
    print("Target physical measurements complete", flush=True)


if __name__ == "__main__":
    prepare()
