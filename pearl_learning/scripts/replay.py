"""Replay a saved PEARL critical scenario without loading policy weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pearl_learning.src.io import read_config, write_json
from pearl_learning.src.scenario_manifest import load_manifest
from pearl_learning.src.task_env import LogicalMergeEnv
from pearl_learning.src.task_spec import LogicalScenarioTaskSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="pearl_learning/configs/merge_family_pearl.yaml")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest)
    manifest = load_manifest(manifest_dir)
    config = read_config(args.config)
    task = LogicalScenarioTaskSpec.from_dict(manifest["task"])
    env = LogicalMergeEnv(task, config, [manifest["case"]])
    try:
        env.reset(options={"case": manifest["case"]})
        for action in np.load(manifest_dir / "actions.npy"):
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        replay = env.episode_record()
    finally:
        env.close()

    original = json.loads((manifest_dir / "metrics.json").read_text(encoding="utf-8"))
    tolerance = float(config["evaluation"]["replay_ttc_tolerance"])
    audit = {
        "original_target_collision": original["target_collision"],
        "replay_target_collision": replay["target_collision"],
        "original_min_ttc": original["min_ttc"],
        "replay_min_ttc": replay["min_ttc"],
        "ttc_error": abs(float(original["min_ttc"]) - float(replay["min_ttc"])),
    }
    audit["passed"] = audit["original_target_collision"] == audit["replay_target_collision"] and audit["ttc_error"] <= tolerance
    write_json(manifest_dir / "replay_metrics.json", replay)
    write_json(manifest_dir / "replay_audit.json", audit)
    if not audit["passed"]:
        raise SystemExit(f"replay audit failed: {audit}")
    print(audit)


if __name__ == "__main__":
    main()
