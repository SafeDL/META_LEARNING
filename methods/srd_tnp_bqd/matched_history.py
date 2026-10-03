"""One speed-matched historical IDM source on the original 2048 A scenes."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import multiprocessing as mp

import numpy as np
from threadpoolctl import threadpool_limits

from .common import append_jsonl, read_jsonl, write_json
from .data import SourceContext, risk_logit
from .measurements import execute
from .s01 import (COUNT, ROOT, TARGET, build_spec, configuration, load_scenes,
                  numeric_inputs)

OUTPUT = ROOT / "history"
SOURCE = "idm_source_speed_23_mps"


def source_spec():
    original = build_spec("idm_source_strong_brake")
    profile = {**original.profile, "name": SOURCE,
               "target_speed": build_spec(TARGET).profile["target_speed"]}
    return replace(original, build_id=SOURCE, parent_build_id=original.build_id, profile=profile)


def measure_source(scene):
    threadpool_limits(limits=1)
    spec = source_spec()
    result, trace, risk, _, elapsed = execute(scene, configuration()["measurement"], spec=spec)
    if not risk["valid_risk"]:
        raise ValueError(f"Invalid supplemental source risk: {scene['scenario_id']}")
    event_start = scene["active_parameters"]["event_start_s"]
    frame = next(row for row in trace if row["time_s"] >= event_start)
    return {"scenario_id": scene["scenario_id"], "ego_collision": bool(result["ego_collision"]),
            **risk, "build_fingerprint": spec.fingerprint, "elapsed_s": elapsed,
            "physics_steps": len(trace) - 1, "ego_speed_at_event_mps": frame["ego"]["speed_mps"]}


def prepare_source():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spec = source_spec()
    write_json(OUTPUT / "protocol.json", {
        "source_build": asdict(spec), "source_fingerprint": spec.fingerprint,
        "only_source_parameter_changed": "target_speed: 27 -> 23 m/s",
        "ego_initial_speed_mps": 25, "A_scenes": COUNT,
        "original_five_sources_and_target_unchanged": True,
        "mean_source_choice": "nearest available desired speed and max_brake to known target profile",
        "h_sources": "original five sources; weights frozen",
        "D_labels_used_for_source_specification": False,
        "independent_mean_readout": "A-only cross-fitted TNP/local response linear readout",
        "main_acquisition": "original one-EAI/two-risk cycle",
        "new_scene_count": 0, "evaluation": "existing-D development",
    })
    path = OUTPUT / "history.jsonl"
    completed = read_jsonl(path) if path.exists() else []
    identifiers = {row["scenario_id"] for row in completed}
    assert len(identifiers) == len(completed)
    assert all(row["build_fingerprint"] == spec.fingerprint for row in completed)
    scenes = load_scenes("A")
    assert identifiers <= {s["scenario_id"] for s in scenes}
    remaining = [s for s in scenes if s["scenario_id"] not in identifiers]
    if remaining:
        with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as executor:
            for row in executor.map(measure_source, remaining, chunksize=8):
                append_jsonl(path, row)
                completed.append(row)
                if len(completed) % 128 == 0:
                    print("Supplemental IDM physical responses", len(completed), "/", COUNT, flush=True)
    assert len(completed) == COUNT
    write_json(OUTPUT / "source_measurement.json", {
        "status": "COMPLETE", "source": SOURCE, "physical_executions": len(completed),
        "collisions": sum(r["ego_collision"] for r in completed),
        "source_elapsed_s_sum": sum(r["elapsed_s"] for r in completed),
        "mean_ego_speed_at_event_mps": float(np.mean([r["ego_speed_at_event_mps"] for r in completed])),
        "new_scenes": 0, "new_target_physical_executions": 0,
    })


def source_context():
    scenes = load_scenes("A")
    rows = {r["scenario_id"]: r for r in read_jsonl(OUTPUT / "history.jsonl")}
    assert len(rows) == len(scenes) == COUNT
    risks = [rows[s["scenario_id"]]["risk"] for s in scenes]
    return SourceContext(SOURCE, numeric_inputs(scenes), risk_logit(risks)[:, None], np.ones(COUNT, bool))


if __name__ == "__main__":
    prepare_source()
