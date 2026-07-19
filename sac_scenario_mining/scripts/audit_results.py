"""Audit held-out SAC results against the Stage 1 acceptance criteria."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..src.utils import dump_json


def _event_rate(path: Path, key: str) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sum(row[key].lower() == "true" for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root",
                        default="results/sac_scenario_mining/final_eval")
    args = parser.parse_args()
    root = Path(args.results_root)
    summaries = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in root.glob("*/summary.json")
    }
    baseline = summaries["random"]
    seed_results = []
    for name in sorted(key for key in summaries if key.startswith("sac_seed")):
        summary = summaries[name]
        vcr_gain = float(summary["valid_critical_rate"]) - float(
            baseline["valid_critical_rate"])
        ttc_reduction = 1.0 - float(summary["median_min_ttc"]) / float(
            baseline["median_min_ttc"])
        non_target = _event_rate(root / name / "episodes.csv",
                                 "non_target_collision")
        outroad = _event_rate(root / name / "episodes.csv",
                              "adversary_out_of_road")
        improvement = vcr_gain >= 0.10 or ttc_reduction >= 0.20
        valid = float(summary["invalid_rate"]) <= 0.25
        target_driven = float(summary["target_collision_rate"]) > max(
            non_target, outroad)
        seed_results.append({
            "run":
            name,
            "valid_critical_rate_gain":
            vcr_gain,
            "median_ttc_reduction":
            ttc_reduction,
            "invalid_rate":
            summary["invalid_rate"],
            "non_target_collision_rate":
            non_target,
            "adversary_out_of_road_rate":
            outroad,
            "improvement_pass":
            improvement,
            "validity_pass":
            valid,
            "target_driven_pass":
            target_driven,
            "seed_pass":
            bool(improvement and valid and target_driven),
        })
    report = {
        "baseline": {
            "run": "random",
            **baseline
        },
        "criteria": {
            "valid_critical_rate_gain":
            ">= 0.10 OR median TTC reduction >= 0.20",
            "invalid_rate": "<= 0.25",
            "target_driven":
            "target collision rate exceeds each non-target collision and adversary-out-of-road rate",
            "required_passing_seeds": 2,
        },
        "seeds": seed_results,
        "passing_seeds": sum(result["seed_pass"] for result in seed_results),
        "accepted": sum(result["seed_pass"] for result in seed_results) >= 2,
    }
    dump_json(root / "acceptance_audit.json", report)
    print(report)
    if not report["accepted"]:
        raise SystemExit("Stage 1 acceptance criteria not met")


if __name__ == "__main__":
    main()
