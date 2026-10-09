"""Measure a finite set of identical scenes across validation SUTs."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.prepare import measure_scene
from research.behavior_response_testing.confirmation import profiles
from research.risk_feedback_meta_testing.evaluate import verify_or_lock

from ..config import COHORT, PREVIOUS, RESULTS


OUTPUT = RESULTS / "matched_scene_diagnostic"
PER_STRATUM = 4
RISK_ANCHORS = (0.99, 0.65, 0.20)


def main():
    verify_or_lock()
    protocol = read_json(PREVIOUS / "results" / "protocol.json")
    names = protocol["split"]["validation"]
    selected_profiles = [profile for profile in profiles() if profile.name in names]
    reference = names[0]
    folder = COHORT / reference / "pool_0"
    all_scenes = read_json(folder / "scenarios.json")
    with np.load(folder / "responses.npz") as bank:
        reference_risk = bank["risk"].copy()
        reference_collision = bank["collision"].copy()
        family = bank["x"][:, 4].copy()
    chosen, strata = [], []
    for value in (0, 1):
        for anchor in RISK_ANCHORS:
            available = [i for i in np.argsort(abs(reference_risk - anchor))
                         if family[i] == value and i not in chosen]
            indices = available[:PER_STRATUM]
            chosen += indices
            strata += [{"index": int(i), "family": value, "risk_anchor": anchor}
                       for i in indices]
    scenes = [all_scenes[i] for i in chosen]
    assert len(scenes) == len(set(chosen)) == 24
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "protocol.json", {
        "role": "matched validation response diagnostic; not training or performance benchmark",
        "reference_profile": reference, "reference_pool": 0,
        "profiles": names, "strata": strata, "scenes": scenes,
        "expected_physical_measurements": 288,
        "physical_settings_and_risk_unchanged": True,
        "use_for_model_training": False,
    })
    arguments = [(scene, profile) for profile in selected_profiles for scene in scenes]
    rows = []
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn")) as pool:
        for number, ((scene, profile), measurement) in enumerate(
                zip(arguments, pool.map(measure_scene, arguments)), 1):
            rows.append({"profile": profile.name, "controller": profile.controller,
                         "probe_index": (number - 1) % len(scenes), **measurement})
            if number % len(scenes) == 0:
                print("MATCHED PHYSICAL", profile.name, number, "/", len(arguments), flush=True)
    reference_rows = [row for row in rows if row["profile"] == reference]
    if any(row["risk"] != float(reference_risk[index])
           or row["collision"] != bool(reference_collision[index])
           for row, index in zip(reference_rows, chosen)):
        raise ValueError("Matched-scene rerun differs from the original physical bank")
    risks = np.asarray([[row["risk"] for row in rows if row["profile"] == name]
                        for name in names])
    collisions = np.asarray([[row["collision"] for row in rows if row["profile"] == name]
                             for name in names])
    diagnostics = []
    for index, stratum in enumerate(strata):
        values = risks[:, index]
        diagnostics.append({**stratum, "probe_index": index,
                            "risk_min": float(values.min()), "risk_max": float(values.max()),
                            "risk_std_across_suts": float(values.std()),
                            "collision_count_across_suts": int(collisions[:, index].sum()),
                            "collision_disagreement": bool(collisions[:, index].any()
                                                           and not collisions[:, index].all())})
    write_json(OUTPUT / "summary.json", {
        "role": "response-identifiability diagnosis, no policy or significance claims",
        "physical_measurements": len(rows), "reference_rerun_matches": True,
        "profiles": names, "records": rows, "per_scene": diagnostics,
        "risk_matrix": risks.tolist(), "collision_matrix": collisions.tolist(),
    })
    verify_or_lock()
    print("MATCHED SCENE DIAGNOSTIC COMPLETE", len(rows), diagnostics, flush=True)


if __name__ == "__main__":
    main()
