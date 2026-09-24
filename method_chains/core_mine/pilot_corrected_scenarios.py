"""Development-only pilot for genuinely critical two-vehicle scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from highway_env_benchmark.envs.cutin_env import CutInScenario
from replications.highway_sut_selection.runner import ASSETS, run_episode
from sut_algorithms.highway_env.registry import policy_factory


MODES = (
    "fast_intrusion", "cutin_braking", "lead_braking", "stop_and_go", "slow_lead_following",
)
SUTS = ("idm_mobil", "mcts_cv", "ppo_ece")
# Explicit development search space. Bounds are physical inputs, not outcome labels.
BOUNDS = {
    "fast_intrusion": ((6.0, 25.0), (-8.0, 0.0)),
    "cutin_braking": ((6.0, 25.0), (-8.0, 0.0)),
    "lead_braking": ((8.0, 30.0), (-7.0, 0.0)),
    "stop_and_go": ((8.0, 30.0), (-7.0, 0.0)),
    "slow_lead_following": ((5.0, 22.0), (-8.0, -1.0)),
}
WIDE_BOUNDS = {
    "fast_intrusion": ((8.0, 40.0), (-8.0, 1.0)),
    "cutin_braking": ((8.0, 42.0), (-8.0, 1.0)),
    "lead_braking": ((10.0, 42.0), (-7.0, 1.0)),
    "stop_and_go": ((8.0, 42.0), (-8.0, 1.0)),
    "slow_lead_following": ((7.0, 36.0), (-8.0, 0.0)),
}
MIXED_BOUNDS = {
    "fast_intrusion": ((6.0, 25.0), (-8.0, 4.0)),
    "cutin_braking": ((6.0, 25.0), (-8.0, 4.0)),
    "lead_braking": ((8.0, 30.0), (-7.0, 4.0)),
    "stop_and_go": ((5.0, 24.0), (-8.0, 4.0)),
    "slow_lead_following": ((5.0, 22.0), (-8.0, 3.0)),
}
BALANCED_BOUNDS = {
    "fast_intrusion": ((6.0, 25.0), (-8.0, 8.0)),
    "cutin_braking": ((6.0, 25.0), (-8.0, 8.0)),
    "lead_braking": ((8.0, 30.0), (-7.0, 8.0)),
    "stop_and_go": ((5.0, 24.0), (-8.0, 8.0)),
    "slow_lead_following": ((5.0, 22.0), (-8.0, 7.0)),
}


def _write_summary(rows: list[dict], seed: int, per_mode: int, output: Path,
                   bounds: dict, suts: tuple[str, ...]) -> None:
    summary = []
    for mode in MODES:
        for sut in suts:
            subset = [row for row in rows if row["mode"] == mode and row["sut"] == sut]
            summary.append({"mode": mode, "sut": sut, "episodes": len(subset),
                            "ego_collisions": sum(str(row["ego_collision"]).lower() == "true" for row in subset),
                            "near_misses": sum(str(row["near_miss"]).lower() == "true" for row in subset),
                            "smallest_polygon_clearance": min(float(row["min_polygon_clearance"]) for row in subset)})
    quartiles = []
    for sut in suts:
        counts = [0, 0, 0, 0]
        for row in rows:
            if row["sut"] != sut or not (str(row["ego_collision"]).lower() == "true" or
                                         str(row["near_miss"]).lower() == "true"):
                continue
            low, high = bounds[row["mode"]][0]
            fraction = (float(row["initial_gap"]) - low) / (high - low)
            counts[min(3, max(0, int(fraction * 4)))] += 1
        quartiles.append({"sut": sut, "critical_by_gap_quartile": counts,
                          "max_share": max(counts) / sum(counts) if sum(counts) else None})
    (output / "summary.json").write_text(json.dumps({"seed": seed, "per_mode": per_mode,
                                                       "bounds": bounds, "summary": summary,
                                                       "quartile_diagnostics": quartiles}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20280901)
    parser.add_argument("--per-mode", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/method_chains/core_mine/studies/corrected_pilot"))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--range-set", choices=("initial", "wide", "mixed", "balanced"), default="initial")
    parser.add_argument("--include-vi", action="store_true")
    args = parser.parse_args()
    if args.per_mode < 1 or args.per_mode > 128:
        raise ValueError("per-mode must be in [1,128]")
    bounds = {"initial": BOUNDS, "wide": WIDE_BOUNDS, "mixed": MIXED_BOUNDS,
              "balanced": BALANCED_BOUNDS}[args.range_set]
    suts = (*SUTS, "vi_ttc") if args.include_vi else SUTS
    if args.summarize_only:
        with (args.output / "episodes.csv").open(encoding="utf-8", newline="") as handle:
            prior = list(csv.DictReader(handle))
        if len(prior) != args.per_mode * len(MODES) * len(suts) or {int(row["seed"]) for row in prior} != {args.seed}:
            raise ValueError("existing episode CSV does not match requested pilot")
        _write_summary(prior, args.seed, args.per_mode, args.output, bounds, suts)
        return
    rows = []
    policies = {name: policy_factory(name, ASSETS) for name in suts}
    for mode_number, mode in enumerate(MODES):
        exponent = int(np.ceil(np.log2(args.per_mode)))
        unit = qmc.Sobol(4, scramble=True, seed=args.seed + mode_number).random_base2(exponent)[:args.per_mode]
        (gap_low, gap_high), (speed_low, speed_high) = bounds[mode]
        for local_index, values in enumerate(unit):
            scenario = CutInScenario(
                gap_low + (gap_high - gap_low) * float(values[0]),
                speed_low + (speed_high - speed_low) * float(values[1]),
                mode,
                0.05 + 0.90 * float(values[2]),
                0.05 + 0.90 * float(values[3]),
            )
            for name in suts:
                result = run_episode(policies[name], scenario, args.seed + mode_number * args.per_mode + local_index)
                rows.append({"seed": args.seed, "mode": mode, "local_index": local_index, "sut": name,
                             "initial_gap": scenario.initial_gap, "relative_speed": scenario.relative_speed,
                             "timing": scenario.timing, "intensity": scenario.intensity,
                             "ego_collision": result["ego_collision"], "background_collision": result["background_collision"],
                             "near_miss": result["near_miss"], "min_ttc": result["min_ttc"],
                             "min_polygon_clearance": result["min_distance"], "completed": result["completed"]})
        print(f"completed {mode}: {len(rows)} episodes", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, args.seed, args.per_mode, args.output, bounds, suts)


if __name__ == "__main__":
    main()
