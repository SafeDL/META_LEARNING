"""Replay B=50 archived hits with the corrected vehicle-outline metric."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from highway_env_benchmark.envs.cutin_env import CutInScenario
from replications.highway_sut_selection.runner import ASSETS, run_episode
from sut_algorithms.highway_env.registry import policy_factory


ROOT = Path("results/method_chains/core_mine/studies/sparse_continuous_suts")
METHOD = "MeanResidual-Risk"


def main() -> None:
    with (ROOT / "validate" / "records.csv").open(encoding="utf-8", newline="") as handle:
        records = [row for row in csv.DictReader(handle)
                   if row["method"] == METHOD and int(row["budget"]) == 50 and int(row["repeat"]) == 0]
    cases = []
    for row in records:
        seed = int(row["seed"])
        with np.load(ROOT / "banks" / "sparse_continuous" / f"sparse_sut_bank_{seed}.npz", allow_pickle=False) as bank:
            target = row["heterogeneity"]
            sut_index = list(bank["sut_names"].astype(str)).index(target)
            for index in (int(value) for value in row["queried_indices"].split(";")):
                if not (bank["ego_collision"][sut_index, index] or bank["near_miss"][sut_index, index]):
                    continue
                cases.append((seed, target, index, CutInScenario(
                    float(bank["anchors"][index, 0]), float(bank["anchors"][index, 1]),
                    str(bank["modes"][index]), float(bank["controls"][index, 0]),
                    float(bank["controls"][index, 1]),
                )))
    policies = {sut: policy_factory(sut, ASSETS) for sut in sorted({case[1] for case in cases})}
    rows = []
    for ordinal, (seed, target, index, scenario) in enumerate(cases, 1):
        result = run_episode(policies[target], scenario, seed + index)
        rows.append({"seed": seed, "target": target, "index": index, "mode": scenario.mode,
                     "ego_collision": result["ego_collision"], "corrected_near_miss": result["near_miss"],
                     "polygon_clearance_m": result["min_distance"],
                     "min_ttc_s": result["min_ttc"] if np.isfinite(result["min_ttc"]) else None})
        if ordinal % 25 == 0 or ordinal == len(cases):
            print(f"replayed {ordinal}/{len(cases)} archived hits", flush=True)
    by_mode = {}
    for mode in sorted({row["mode"] for row in rows}):
        selected = [row for row in rows if row["mode"] == mode]
        by_mode[mode] = {"archived_hits": len(selected),
                         "corrected_events": sum(row["ego_collision"] or row["corrected_near_miss"] for row in selected),
                         "smallest_polygon_clearance_m": min(row["polygon_clearance_m"] for row in selected)}
    report = {"method": METHOD, "budget_per_target": 50, "target_units": len(records),
              "archived_hits_replayed": len(rows),
              "corrected_ego_collisions": sum(row["ego_collision"] for row in rows),
              "corrected_near_misses": sum(row["corrected_near_miss"] for row in rows),
              "by_mode": by_mode, "by_target": dict(Counter(row["target"] for row in rows)),
              "cases": rows}
    path = ROOT / "geometry_audit.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
