"""Train one SAC task and evaluate checkpoints on validation only."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from archives.pearl_learning.scripts.run_baselines import _evaluate_sac, _implementation_hash, _new_sac
from archives.pearl_learning.src.casebook import load_casebook
from archives.pearl_learning.src.io import content_hash, read_config, write_json
from archives.pearl_learning.src.task_env import LogicalMergeEnv
from archives.pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--environment-steps", type=int, default=10_000)
    parser.add_argument("--checkpoint-interval", type=int, default=2_500)
    parser.add_argument("--priority-alignment-weight", type=float)
    parser.add_argument("--rule-mismatch-penalty", type=float)
    parser.add_argument("--invalid-event-penalty", type=float)
    args = parser.parse_args()
    if args.environment_steps <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("training budgets must be positive")

    cfg = copy.deepcopy(read_config(args.config))
    if args.priority_alignment_weight is not None:
        cfg["reward"]["priority_alignment_weight"] = float(args.priority_alignment_weight)
    if args.rule_mismatch_penalty is not None:
        cfg["reward"]["rule_mismatch_penalty"] = float(args.rule_mismatch_penalty)
    if args.invalid_event_penalty is not None:
        for key in (
            "non_target_collision_penalty", "out_of_road_penalty", "wrong_route_penalty",
        ):
            cfg["reward"][key] = float(args.invalid_event_penalty)
    taskbook = load_taskbook(args.taskbook)
    matches = [task for task in taskbook["meta_train"] if task.task_id == args.task_id]
    if len(matches) != 1:
        raise SystemExit("--task-id must identify exactly one frozen meta-train task")
    task = matches[0]
    book = load_casebook(task, args.casebook_root)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    env = LogicalMergeEnv(task, cfg, book["train_pool"])
    model = _new_sac(env, args.seed)
    checkpoints = []
    try:
        trained = 0
        while trained < args.environment_steps:
            increment = min(args.checkpoint_interval, args.environment_steps - trained)
            model.learn(total_timesteps=increment, reset_num_timesteps=False)
            trained += increment
            path = root / f"{task.task_id}_{trained}_steps"
            model.save(path)
            # MetaDrive owns a process-global engine. Close the training
            # environment before opening the independent validation one.
            env.close()
            metrics = _evaluate_sac(model, task, cfg, book["validation_query"])
            checkpoints.append({
                "steps": trained,
                "checkpoint": str(path.with_suffix(".zip")),
                "summary": metrics["summary"],
            })
            if trained < args.environment_steps:
                env = LogicalMergeEnv(task, cfg, book["train_pool"])
                model.set_env(env)
    finally:
        env.close()

    write_json(root / "diagnostic.json", {
        "schema": "single_task_sac_validation_diagnostic",
        "task_id": task.task_id,
        "seed": args.seed,
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "config_hash": content_hash(cfg),
        "implementation_hash": _implementation_hash(),
        "training_split": "train_pool",
        "evaluation_split": "validation_query",
        "test_query_accessed": False,
        "checkpoints": checkpoints,
    })
    print(f"completed validation-only SAC diagnostic for {task.task_id}")


if __name__ == "__main__":
    main()
