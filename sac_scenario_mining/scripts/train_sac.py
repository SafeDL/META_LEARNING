"""Train SB3 SAC; independent validation selects the best checkpoint."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ..src.env import Stage1AdversarialMergeEnv
from ..src.utils import dump_json, load_config, versions
from .evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="sac_scenario_mining/configs/merge_sac.yaml")
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

    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    def make_training_env() -> Monitor:
        return Monitor(
            Stage1AdversarialMergeEnv(cfg, split="train", seed=args.seed),
            filename=str(root / "train_monitor"),
        )

    env = make_training_env()
    sac = cfg["sac"]
    model = SAC(
        sac["policy"],
        env,
        learning_rate=sac["learning_rate"],
        buffer_size=sac["buffer_size"],
        learning_starts=sac["learning_starts"],
        batch_size=sac["batch_size"],
        tau=sac["tau"],
        gamma=sac["gamma"],
        train_freq=sac["train_freq"],
        gradient_steps=sac["gradient_steps"],
        ent_coef=sac["ent_coef"],
        policy_kwargs={"net_arch": sac["policy_hidden_sizes"]},
        seed=args.seed,
        device=args.device or cfg["experiment"]["device"],
        verbose=1,
    )
    best_score, completed, interval = -float("inf"), 0, min(5000, total)
    try:
        while completed < total:
            step = min(interval, total - completed)
            model.learn(total_timesteps=step,
                        reset_num_timesteps=completed == 0)
            completed += step
            model.save(root / "checkpoint")
            env.close()
            summary = evaluate(
                cfg,
                "zero",
                str(root / "checkpoint.zip"),
                "validation",
                int(cfg["scenario_split"]["validation"]["num_scenarios"]),
                args.seed,
                True,
                root / "validation" / f"step_{completed}",
            )
            score = (float(summary["valid_critical_rate"]),
                     -float(summary["median_min_ttc"]))
            if completed == step or score > best_score:
                best_score = score
                shutil.copy2(root / "checkpoint.zip", root / "best_model.zip")
            if completed < total:
                env = make_training_env()
                model.set_env(env)
        model.save(root / "final_model")
    finally:
        env.close()
    print(f"Saved {root / 'best_model.zip'} and {root / 'final_model.zip'}")


if __name__ == "__main__":
    main()
