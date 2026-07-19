"""Run a trained SAC policy in a visible MetaDrive test scenario."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sac_scenario_mining.src.env import Stage1AdversarialMergeEnv
from sac_scenario_mining.src.utils import load_config

DEFAULT_MODEL = PROJECT_ROOT / "results/sac_scenario_mining/merge_sac_seed2/best_model.zip"
DEFAULT_SCENARIO_SEED = 2016
DEFAULT_CONFIG = PROJECT_ROOT / "sac_scenario_mining/configs/merge_sac.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one deterministic SAC rollout in MetaDrive.")
    parser.add_argument("--policy-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--scenario-seed",
                        type=int,
                        default=DEFAULT_SCENARIO_SEED)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    from stable_baselines3 import SAC

    config = load_config(args.config)
    config["environment"]["use_render"] = True
    env = Stage1AdversarialMergeEnv(config, split="test", seed=0)
    model = SAC.load(Path(args.policy_path), device="auto")
    try:
        observation, info = env.reset(
            options={"scenario_seed": args.scenario_seed})
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            env.render()
            done = terminated or truncated
        print({"scenario_seed": info["scenario_seed"], **env.episode_record()})
    finally:
        env.close()


if __name__ == "__main__":
    main()
