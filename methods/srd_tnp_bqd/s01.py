"""The original RAS-FRT-UQ S01 A/D benchmark, without resampling."""
import os
from pathlib import Path
import sys

import numpy as np

from highway_sim_env.s01_parameters import coordinates, valid_label
from highway_sim_env.build_spec import BuildSpec
from sut_algorithms.highway_env.registry import build_spec_factory
from .benchmark import GRID_BINS, parameter_cells
from .common import DEFAULT_CONFIG, REPO, config_at, read_json, read_jsonl
from .data import SourceContext, risk_logit
from .measurements import execute


ROOT = REPO / "results/srd_tnp_bqd"
BENCHMARK_ROOT = REPO / "benchmarks/s01"
HISTORY_ROOT = BENCHMARK_ROOT / "historical_idm"
TARGET_ROOT = BENCHMARK_ROOT / "s01_fvdm"
SOURCES = ("idm_source_short_headway", "idm_source_nominal", "idm_source_long_headway",
           "idm_source_limited_brake", "idm_source_strong_brake")
TARGET = "fvdm_safety_speed_23_mps"
COUNT = 2048
INPUT_DIM = 4
CELL_COUNT = GRID_BINS ** INPUT_DIM


def configuration():
    return config_at(DEFAULT_CONFIG)


def build_spec(name):
    if name in SOURCES:
        return build_spec_factory(name)
    if name != TARGET:
        raise ValueError("SUT is outside the original S01 benchmark")
    profile = read_json(TARGET_ROOT / "protocol.json")["target_profile"]
    return BuildSpec(TARGET, "profiled_fvdm", None, "legacy_profile", "Profiled-FVDM", 20., profile=profile)


def load_scenes(bank):
    paths = {"A": HISTORY_ROOT / "history_manifest.jsonl",
             "D": TARGET_ROOT / "candidate_manifest.jsonl"}
    return read_jsonl(paths[bank])


def numeric_inputs(scenes):
    return coordinates(scenes)


def cells_for(scenes, bins=GRID_BINS):
    return parameter_cells(numeric_inputs(scenes), bins)


def response_rows(bank, source=None):
    name = TARGET if bank == "D" else source
    if bank == "A" and name not in SOURCES:
        raise ValueError("unknown historical SUT")
    raw_path = TARGET_ROOT / f"{name}.jsonl" if bank == "D" else HISTORY_ROOT / "banks" / f"{name}.jsonl"
    raw = {r["scenario_id"]: r for r in read_jsonl(raw_path)}
    risks = {r["scenario_id"]: r for r in read_jsonl(ROOT / "measurements" / bank / f"{name}.risk.jsonl")}
    scenes = load_scenes(bank)
    identifiers = {s["scenario_id"] for s in scenes}
    if len(scenes) != COUNT or set(raw) != identifiers or set(risks) != identifiers:
        raise ValueError("S01 response bank is incomplete")
    rows = []
    for scene in scenes:
        result, risk = raw[scene["scenario_id"]], risks[scene["scenario_id"]]
        if valid_label(result) is None or not risk["valid_risk"] or risk["inconclusive"]:
            raise ValueError("invalid S01 response")
        if result["ego_collision"] != risk["ego_collision"]:
            raise ValueError("passive measurement changed original collision labels")
        if risk["build_fingerprint"] != build_spec(name).fingerprint:
            raise ValueError("original SUT build changed")
        rows.append({**result, **risk})
    return scenes, rows


def source_contexts():
    output = []
    for name in SOURCES:
        scenes, rows = response_rows("A", name)
        x = numeric_inputs(scenes)
        z = risk_logit([r["risk"] for r in rows])[:, None]
        output.append(SourceContext(name, x, z, np.ones(COUNT, bool)))
    return output


def response_guard():
    """Block every target-label file in training and selector processes."""
    forbidden = (TARGET_ROOT / f"{TARGET}.jsonl", ROOT / "measurements/D",
                 REPO / "results/ras_frt_uq/legacy_s01")

    def guard(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            if any(path == p or path.is_relative_to(p) for p in forbidden):
                raise PermissionError("undisclosed target responses are forbidden")
    sys.addaudithook(guard)


def measure_target(scene):
    result, trace, summary, _, elapsed = execute(scene, configuration()["measurement"], spec=build_spec(TARGET))
    return {**result, **summary, "elapsed_s": elapsed, "physics_steps": len(trace) - 1}
