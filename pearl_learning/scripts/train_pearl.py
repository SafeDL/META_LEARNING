from __future__ import annotations
import argparse
from pearl_learning.src.io import read_config
from pearl_learning.src.pearl_trainer import train
from pearl_learning.src.taskbook import build_taskbook


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--max-env-steps", type=int); parser.add_argument("--run-name", required=True); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); cfg = read_config(args.config); taskbook = build_taskbook(cfg); tasks = taskbook["meta_train"]
    if args.smoke: tasks = tasks[:4]
    max_steps = args.max_env_steps or int(cfg["meta_training"]["total_environment_steps"])
    result = train(cfg, tasks, taskbook["meta_validation"], max_steps, args.seed, args.run_name, args.smoke); print(f"PEARL training completed: {result}")


if __name__ == "__main__": main()
