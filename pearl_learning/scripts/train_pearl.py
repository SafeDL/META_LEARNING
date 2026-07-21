from __future__ import annotations
import argparse
from pathlib import Path

from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.io import read_config
from pearl_learning.src.pearl_trainer import train
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload
from pearl_learning.src.io import content_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--taskbook", required=True); parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--max-env-steps", type=int); parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--gate-manifest"); parser.add_argument("--output-root")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.output_root:
        cfg["project"] = {**cfg["project"], "output_root": args.output_root}
    taskbook = load_taskbook(args.taskbook)
    tasks = taskbook["meta_train"][: min(2, len(taskbook["meta_train"]))] if args.smoke else taskbook["meta_train"]
    validation = taskbook["meta_validation"][:1] if args.smoke else taskbook["meta_validation"]
    selected = tasks + validation
    casebooks = {task.task_id: load_casebook(task, args.casebook_root) for task in selected}
    max_steps = args.max_env_steps or int(cfg["meta_training"]["total_environment_steps"])
    result = train(cfg, tasks, validation, casebooks, content_hash(taskbook_payload(taskbook)), max_steps, args.seed, args.run_name, args.smoke, args.gate_manifest)
    print(f"PEARL training completed: {result}")


if __name__ == "__main__":
    main()
