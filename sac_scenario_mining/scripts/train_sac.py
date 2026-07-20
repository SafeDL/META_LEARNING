"""Train SAC on the explicit training case table with continuous diagnostics."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from ..src.casebook import build_case_table
from ..src.env import Stage1AdversarialMergeEnv
from ..src.utils import dump_json, load_config, versions
from .evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sac_scenario_mining/configs/merge_sac.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="merge_sac_seed0")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = load_config(args.config)
    total = args.total_timesteps or int(cfg["sac"]["total_timesteps"])
    root = Path(cfg["project"]["output_root"]) / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    dump_json(root / "config_resolved.json", cfg)
    dump_json(root / "versions.json", versions())
    dump_json(root / "train_case_table.json", build_case_table(cfg, "train"))
    dump_json(root / "validation_case_table.json", build_case_table(cfg, "validation"))

    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(Stage1AdversarialMergeEnv(cfg, split="train", seed=args.seed), filename=str(root / "train_monitor"))
    sac = cfg["sac"]
    model = SAC(sac["policy"], env, learning_rate=sac["learning_rate"], buffer_size=sac["buffer_size"],
                learning_starts=sac["learning_starts"], batch_size=sac["batch_size"], tau=sac["tau"],
                gamma=sac["gamma"], train_freq=sac["train_freq"], gradient_steps=sac["gradient_steps"],
                ent_coef=sac["ent_coef"], policy_kwargs={"net_arch": sac["policy_hidden_sizes"]},
                seed=args.seed, device=args.device or cfg["experiment"]["device"], verbose=1)
    model.set_logger(configure(str(root), ["stdout", "csv"]))
    best_score, completed = None, 0
    interval = min(int(sac["validation_interval"]), total)
    try:
        while completed < total:
            chunk = min(interval, total - completed)
            model.learn(total_timesteps=chunk, reset_num_timesteps=completed == 0, log_interval=10)
            completed += chunk
            model.save(root / "checkpoint")
            summary = evaluate(cfg, "zero", str(root / "checkpoint.zip"), "validation", None,
                               args.seed, True, root / "validation" / f"step_{completed}", env=env.env,
                               save_case_table=False)
            # This project mines accident-inducing cases: a valid target
            # collision is therefore the primary model-selection objective.
            # Valid low-TTC cases remain the secondary signal.
            score = (float(summary["target_collision_rate"]), float(summary["valid_critical_rate"]),
                     -float(summary["invalid_rate"]), -float(summary["median_min_ttc"]))
            if best_score is None or score > best_score:
                best_score = score
                shutil.copy2(root / "checkpoint.zip", root / "best_model.zip")
            if completed < total:
                # Validation reuses MetaDrive's singleton physical engine.  It
                # leaves that engine at a held-out terminal state, whereas SB3
                # caches the previous training observation.  Start a fresh
                # training episode and replace the cache before learning.
                model._last_obs = model.get_env().reset()
                model._last_episode_starts = np.ones((model.n_envs,), dtype=bool)
        model.save(root / "final_model")
    finally:
        env.close()
    print(f"Saved {root / 'best_model.zip'} and {root / 'final_model.zip'}")


if __name__ == "__main__":
    main()
