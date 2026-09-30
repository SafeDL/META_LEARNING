"""Confirm a calibrated FVDM collision rate on fresh uniform S01 scenes."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scipy.stats import qmc

from highway_sim_env.envs.fbrt_unified_env import (
    EXECUTION_CONTRACT, FBRTUnifiedEnv, PHYSICS_HZ,
)
from highway_sim_env.s01_parameters import NAMES, bounds
from highway_sim_env.build_spec import BuildSpec
from methods.ras_frt.protocol import read_jsonl, write_json, write_jsonl
from sut_algorithms.highway_env.idm_profiles import SUTProfile


OUTPUT_ROOT = Path(
    "results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation"
)
SOBOL_SEED = 43105
BUILD_ID = "fvdm_safety_speed_23_mps"
COUNT = 2048
WORKERS = 16
SCENARIO_TEMPLATE = OUTPUT_ROOT / "scenario_template.json"
PROFILE = SUTProfile(
    BUILD_ID, "FVDM", max_brake=8.0, desired_gap=8.0,
    target_speed=23.0, fvdm_sensitivity=0.6,
    fvdm_velocity_gain=1.0, fvdm_transition_gap=8.0,
)


def execute_profile(
    profile: SUTProfile, scenes: list[dict],
    include_analysis_fields: bool = False,
) -> tuple[str, list[dict]]:
    """Keep the frozen D response schema; add diagnostics for new banks."""
    spec = BuildSpec(
        profile.name, f"profiled_{profile.controller.lower()}", None,
        "legacy_profile", f"Profiled-{profile.controller}", 20.0,
        profile=profile.__dict__.copy(),
    )
    rows = []
    for scene in scenes:
        env = FBRTUnifiedEnv(spec, scene)
        try:
            env.reset(seed=scene["simulator_seed"])
            limit = int(round(float(env.config["duration"]) * PHYSICS_HZ))
            while env.steps < limit:
                if any(actor.crashed for actor in env.actors.values()):
                    break
                env._advance()
            result = env.result()
            row = {
                "scenario_id": scene["scenario_id"],
                "build_id": profile.name,
                "build_fingerprint": spec.fingerprint,
                "completed": result["completed"],
                "ego_collision": result["ego_collision"],
                "inconclusive": result["inconclusive"],
                "min_clearance": result["min_clearance"],
                "collision_partner_role": result["collision_partner_role"],
                "ego_distance_m": env.trace[-1]["ego"]["x_m"] -
                                  env.trace[0]["ego"]["x_m"],
            }
            if include_analysis_fields:
                row.update({key: result[key] for key in (
                    "min_ttc", "collision_type", "collision_time_s",
                    "observed_maneuver_phase",
                )})
            rows.append(row)
        finally:
            env.close()
    return profile.name, rows


def execute_bank(profile: SUTProfile, scenes: list[dict],
                 workers: int = WORKERS,
                 include_analysis_fields: bool = False) -> list[dict]:
    parts = [scenes[i::workers] for i in range(workers)]
    collected = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute_profile, profile, part,
                               include_analysis_fields) for part in parts]
        for future in as_completed(futures):
            name, chunk = future.result()
            print("completed", name, len(chunk), flush=True)
            collected.update({row["scenario_id"]: row for row in chunk})
    return [collected[scene["scenario_id"]] for scene in scenes]


def make_manifest(
    sobol_seed: int = SOBOL_SEED,
    scenario_prefix: str = "ras_frt:s01:uniform_confirmation",
    context_id: str = "S01:research_v3:ras_frt_profile_confirmation",
) -> list[dict]:
    template = json.loads(SCENARIO_TEMPLATE.read_text(encoding="utf-8"))
    limits = bounds()
    samples = qmc.Sobol(d=len(NAMES), scramble=True,
                        seed=sobol_seed).random_base2(m=11)
    rows = []
    for index, point in enumerate(samples):
        row = {key: value for key, value in template.items()
               if key not in {"scenario_id", "sample_index", "active_parameters",
                              "context_id"}}
        row["scenario_id"] = f"{scenario_prefix}:{index:04d}"
        row["sample_index"] = index
        row["context_id"] = context_id
        row["active_parameters"] = {
            name: float(limits[name][0] + point[j] *
                        (limits[name][1] - limits[name][0]))
            for j, name in enumerate(NAMES)
        }
        rows.append(row)
    return rows


def main() -> None:
    profile = PROFILE
    manifest_path = OUTPUT_ROOT / "candidate_manifest.jsonl"
    if manifest_path.exists():
        scenes = read_jsonl(manifest_path)
        if scenes != make_manifest():
            raise ValueError("frozen confirmation manifest changed")
    else:
        scenes = make_manifest()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        write_jsonl(manifest_path, scenes)
        write_json(OUTPUT_ROOT / "protocol.json", {
            "purpose": (
                "build the fixed-profile FVDM target library D for "
                "budgeted discovery evaluation"
            ),
            "target_profile_selection": (
                "profile fixed using prior controller calibration before D "
                "sampling; D scenes and outcomes did not select the profile"
            ),
            "family": "S01",
            "sampling": "uniform scrambled Sobol over unchanged four-dimensional bounds",
            "sobol_seed": SOBOL_SEED,
            "candidate_count": COUNT,
            "target_build_id": BUILD_ID,
            "target_profile": profile.__dict__,
            "simulator_seed": scenes[0]["simulator_seed"],
            "execution_contract": EXECUTION_CONTRACT,
            "target_labels_used_to_select_D_scenes": False,
        })
    if len(scenes) != COUNT:
        raise ValueError("confirmation bank is incomplete")
    bank_path = OUTPUT_ROOT / f"{BUILD_ID}.jsonl"
    if bank_path.exists():
        rows = read_jsonl(bank_path)
    else:
        rows = execute_bank(profile, scenes)
        write_jsonl(bank_path, rows)
    fingerprint = BuildSpec(
        profile.name, "profiled_fvdm", None, "legacy_profile",
        "Profiled-FVDM", 20.0, profile=profile.__dict__.copy(),
    ).fingerprint
    if len(rows) != COUNT or any(
        row["scenario_id"] != scene["scenario_id"] or
        row["build_fingerprint"] != fingerprint
        for row, scene in zip(rows, scenes)
    ):
        raise ValueError("confirmation responses do not match the manifest")
    if any(row["inconclusive"] for row in rows):
        raise ValueError("inconclusive confirmation episode")
    collisions = sum(row["ego_collision"] for row in rows)
    completed = [row for row in rows if row["completed"]]
    summary = {
        "candidate_count": COUNT,
        "valid_count": COUNT,
        "collisions": collisions,
        "collision_rate": collisions / COUNT,
        "completed_count": len(completed),
        "mean_ego_distance_m_completed": sum(
            row["ego_distance_m"] for row in completed) / len(completed),
    }
    write_json(OUTPUT_ROOT / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
