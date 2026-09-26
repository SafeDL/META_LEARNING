"""Development-only opportunity check for frozen same-family VI revisions."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from methods.core_mine import sparse_sut_experiment as sparse
from methods.core_mine.control_frequency_audit import _simulate
from methods.core_mine.versioned_vi import VERSIONS, version_policy
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


SEED = 20291118
ROOT = Path("results/method_chains/core_mine/studies/vi_revision_pilot")
MODES = ("fast_intrusion", "cutin_braking", "lead_braking", "stop_and_go",
         "slow_lead_following")
BUILDS = ("vi_ttc", *VERSIONS)


def _job(build: str, indices: np.ndarray, anchors: np.ndarray,
         modes: np.ndarray, controls: np.ndarray) -> list[dict]:
    policy = policy_factory(build, ASSETS) if build == "vi_ttc" else version_policy(build)
    rows = []
    for index in indices:
        anchor, control = anchors[index], controls[index]
        result = _simulate(build, SEED, int(index), float(anchor[0]), float(anchor[1]),
                           str(modes[index]), float(control[0]), float(control[1]),
                           20, policy=policy)
        rows.append({"build": build, "seed": SEED, "index": int(index),
                     "mode": str(modes[index]),
                     "ego_collision": bool(result["ego_collision"]),
                     "background_collision": bool(result["background_collision"]),
                     "near_miss": bool(result["near_miss"]),
                     "completed": bool(result["completed"]),
                     "min_ttc": float(result["min_ttc"]),
                     "min_clearance": float(result["min_distance"])})
    return rows


def main() -> None:
    sparse.configure_proposal("v8_source_safe")
    anchors, modes, controls, _ = sparse.sparse_scenarios(SEED)
    indices = np.concatenate([np.flatnonzero(modes == mode)[:16] for mode in MODES])
    if len(indices) != 80 or len(set(indices.tolist())) != 80:
        raise RuntimeError("pilot must use 16 unique candidates per mode")
    ROOT.mkdir(parents=True, exist_ok=True)
    record_path = ROOT / "records.csv"
    if record_path.exists():
        raise RuntimeError("existing frozen pilot records are preserved")
    with ProcessPoolExecutor(max_workers=5) as pool:
        batches = list(pool.map(_job, BUILDS, [indices] * len(BUILDS),
                                [anchors] * len(BUILDS), [modes] * len(BUILDS),
                                [controls] * len(BUILDS)))
    rows = [row for batch in batches for row in batch]
    by_build = {build: {row["index"]: row for row in batch}
                for build, batch in zip(BUILDS, batches, strict=True)}
    for index in indices:
        base = by_build["vi_ttc"][int(index)]
        copy = by_build["vi_reference"][int(index)]
        for key in ("ego_collision", "background_collision", "near_miss",
                    "completed", "min_ttc", "min_clearance"):
            if base[key] != copy[key]:
                raise RuntimeError(f"reference adapter is not identical at {index}: {key}")
    with record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    eligible = {int(index) for index in indices
                if by_build["vi_ttc"][int(index)]["completed"]
                and not by_build["vi_ttc"][int(index)]["ego_collision"]
                and not by_build["vi_ttc"][int(index)]["near_miss"]}
    summaries = []
    for build in VERSIONS:
        failures = [by_build[build][index] for index in eligible
                    if by_build[build][index]["ego_collision"]
                    or by_build[build][index]["near_miss"]]
        summaries.append({"build": build, "eligible_reference_safe": len(eligible),
                          "new_failures": len(failures),
                          "ego_collisions": sum(row["ego_collision"] for row in failures),
                          "failure_modes": sorted({row["mode"] for row in failures}),
                          "by_mode": {mode: sum(row["mode"] == mode for row in failures)
                                      for mode in MODES}})
    output = {"seed": SEED, "status": "development_only", "frequency_hz": 20,
              "physics_frequency_hz": 20, "original_scenarios": 80,
              "physical_episodes": len(rows), "reference_adapter_parity": True,
              "reference_safe_candidates": len(eligible),
              "reference_safe_modes": sorted(set(modes[list(eligible)].astype(str))),
              "frozen_versions": VERSIONS, "variants": summaries}
    (ROOT / "summary.json").write_text(json.dumps(output, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
