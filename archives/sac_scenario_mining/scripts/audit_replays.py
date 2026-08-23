"""Replay exported critical scenarios and write a deterministic audit report."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ..src.utils import dump_json, load_config


def _is_critical(record: dict, threshold: float) -> bool:
    return bool(record["target_collision"]
                or float(record["min_ttc"]) <= threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios-root", required=True)
    parser.add_argument("--config",
                        default="archives/sac_scenario_mining/configs/merge_sac.yaml")
    args = parser.parse_args()

    root = Path(args.scenarios_root)
    config = load_config(args.config)
    tolerance = float(config["evaluation"]["replay_ttc_tolerance"])
    threshold = float(config["evaluation"]["critical_ttc_threshold"])
    entries = []
    for scenario in sorted(root.glob("rank_*")):
        manifest = scenario / "manifest.json"
        original = json.loads(
            (scenario / "metrics.json").read_text(encoding="utf-8"))
        result = subprocess.run(
            [
                sys.executable, "-m", "archives.sac_scenario_mining.scripts.replay",
                "--manifest",
                str(manifest), "--config", args.config
            ],
            text=True,
            capture_output=True,
        )
        replay_path = scenario / "replay_metrics.json"
        replay = json.loads(replay_path.read_text(
            encoding="utf-8"
        )) if result.returncode == 0 and replay_path.exists() else None
        same_collision = replay is not None and bool(
            original["target_collision"]) == bool(replay["target_collision"])
        same_critical = replay is not None and _is_critical(
            original, threshold) == _is_critical(replay, threshold)
        ttc_error = abs(
            float(original["min_ttc"]) -
            float(replay["min_ttc"])) if replay is not None else float("inf")
        entries.append({
            "scenario":
            scenario.name,
            "replay_returncode":
            result.returncode,
            "original_target_collision":
            bool(original["target_collision"]),
            "replay_target_collision":
            bool(replay["target_collision"]) if replay else None,
            "original_min_ttc":
            original["min_ttc"],
            "replay_min_ttc":
            replay["min_ttc"] if replay else None,
            "ttc_error":
            ttc_error,
            "collision_consistent":
            same_collision,
            "critical_consistent":
            same_critical,
            "passed":
            bool(same_collision and same_critical and ttc_error <= tolerance),
            "stderr":
            result.stderr[-1000:] if result.returncode else "",
        })
    report = {
        "scenarios": len(entries),
        "passed": sum(entry["passed"] for entry in entries),
        "ttc_tolerance": tolerance,
        "entries": entries,
    }
    dump_json(root.parent / "replay_audit.json", report)
    print(report)
    if report["passed"] != report["scenarios"]:
        raise SystemExit("One or more scenario replays failed")


if __name__ == "__main__":
    main()
