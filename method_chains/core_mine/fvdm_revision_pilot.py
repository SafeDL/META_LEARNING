"""Frozen FVDM-family physical opportunity pilot on the original 80 cases."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from method_chains.core_mine.idm_revision_pilot import SEED, _episode, scenarios
from sut_algorithms.highway_env.idm_profiles import ADATE_SOURCE_PROFILES


ROOT = Path("results/method_chains/core_mine/studies/fvdm_revision_pilot")
REFERENCE = next(profile for profile in ADATE_SOURCE_PROFILES
                 if profile.name == "SM-Strong-FVDM")
BUILDS = {
    "fvdm_ref": REFERENCE,
    "fvdm_delay05": replace(REFERENCE, name="fvdm_delay05", reaction_delay=.5),
    "fvdm_brake3": replace(REFERENCE, name="fvdm_brake3", max_brake=3.0),
    "fvdm_delay05_brake3": replace(REFERENCE, name="fvdm_delay05_brake3",
                                   reaction_delay=.5, max_brake=3.0),
}


def _job(build: str) -> list[dict]:
    return [{"build": build, "seed": SEED, "index": index, "mode": scene.mode,
             "gap": scene.initial_gap, "relative_speed": scene.relative_speed,
             "timing": scene.timing, "intensity": scene.intensity,
             **_episode(BUILDS[build], scene, SEED + index)}
            for index, scene in enumerate(scenarios())]


def main() -> None:
    with ProcessPoolExecutor(max_workers=4) as executor:
        groups = list(executor.map(_job, BUILDS))
    rows = [row for group in groups for row in group]
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source = [row for row in rows if row["build"] == "fvdm_ref"]
    safe = {row["index"] for row in source if row["completed"]
            and not row["ego_collision"] and not row["near_miss"]}
    modes = tuple(dict.fromkeys(row["mode"] for row in source))
    source_safe_by_mode = {mode: sum(row["index"] in safe
                                     for row in source if row["mode"] == mode)
                           for mode in modes}
    variants = []
    for build in tuple(BUILDS)[1:]:
        subset = [row for row in rows if row["build"] == build and row["index"] in safe]
        by_mode = {mode: sum(row["ego_collision"] or row["near_miss"]
                             for row in subset if row["mode"] == mode)
                   for mode in modes}
        variants.append({"build": build,
                         "new_failures": sum(by_mode.values()),
                         "ego_collisions": sum(row["ego_collision"] for row in subset),
                         "failure_modes": [mode for mode, count in by_mode.items() if count],
                         "by_mode": by_mode})
    gate = (all(count >= 8 for count in source_safe_by_mode.values())
            and sum(item["new_failures"] >= 2 and len(item["failure_modes"]) >= 2
                    for item in variants) >= 2)
    output = {"seed": SEED, "source": "SM-Strong-FVDM",
              "candidate_scenarios": 80, "physical_episodes": len(rows),
              "source_safe_total": len(safe),
              "source_safe_by_mode": source_safe_by_mode,
              "physics_frequency_hz": 20,
              "ego_control_frequency_hz": 20,
              "gate_passed": gate, "variants": variants}
    (ROOT / "summary.json").write_text(json.dumps(output, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
