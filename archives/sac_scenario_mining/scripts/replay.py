"""Action-trace replay for saved parameterized critical scenarios."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..src.env import Stage1AdversarialMergeEnv
from ..src.scenario_manifest import load_manifest
from ..src.utils import dump_json, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="archives/sac_scenario_mining/configs/merge_sac.yaml")
    parser.add_argument("--render", choices=["topdown"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.render:
        config["environment"]["use_render"] = True
    manifest = load_manifest(args.manifest)
    actions = np.load(Path(args.manifest).with_name("actions.npy"))
    env = Stage1AdversarialMergeEnv(config, split="test", seed=manifest["policy_seed"])
    try:
        env.reset(options={"case": manifest["case"]})
        for action in actions:
            _, _, terminated, truncated, _ = env.step(action)
            if args.render:
                env.render(view=args.render)
            if terminated or truncated:
                break
        record = env.episode_record()
        dump_json(Path(args.manifest).with_name("replay_metrics.json"), record)
        print(record)
    finally:
        env.close()


if __name__ == "__main__":
    main()
