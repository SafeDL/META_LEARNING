"""Compare PEARL with SAC baselines under matched new-task interaction budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch

from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.checkpoint import load_checkpoint
from pearl_learning.src.evaluator import evaluate_fewshot
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.metrics import summarize
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.task_env import LogicalMergeEnv
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


class FixedSupportCases(gym.Wrapper):
    """Cycle only frozen support cases while an online baseline spends its budget."""

    def __init__(self, env: LogicalMergeEnv, cases: list[Mapping[str, Any]]):
        super().__init__(env)
        if not cases:
            raise ValueError("matched-budget training requires at least one support case")
        self.cases = [dict(case) for case in cases]
        self.index = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        case = self.cases[self.index % len(self.cases)]
        self.index += 1
        return self.env.reset(seed=seed, options={"case": case})


def _evaluate_sac(model: Any, task: Any, config: Mapping[str, Any], cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    env = LogicalMergeEnv(task, config, cases)
    records: list[dict[str, Any]] = []
    try:
        for case in cases:
            observation, _ = env.reset(options={"case": case})
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
            records.append(env.episode_record())
    finally:
        env.close()
    return {"summary": summarize(records), "records": records}


def _new_online_sac(env: gym.Env, seed: int) -> Any:
    """An intentionally strong online SAC baseline: one update after each new step."""
    from stable_baselines3 import SAC
    return SAC(
        "MlpPolicy", env, seed=seed, verbose=0, learning_starts=1,
        batch_size=8, train_freq=(1, "step"), gradient_steps=1,
    )


def _mean_strict(tasks: Mapping[str, Mapping[str, Any]]) -> float:
    return float(np.mean([float(value["summary"]["valid_critical_strict_rate"]) for value in tasks.values()]))


def _restore_checkpoint_rng(checkpoint: Mapping[str, Any]) -> None:
    state = checkpoint["rng_state"]
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu").clone())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([torch.as_tensor(item, dtype=torch.uint8, device="cpu").clone() for item in state["cuda"]])


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the matched-budget comparison, not its per-query rollout records."""
    budgets: dict[str, Any] = {}
    for shot, entry in result["budgets"].items():
        budgets[shot] = {
            "mean_support_environment_steps": entry["mean_support_environment_steps"],
            "support_environment_steps_by_task": entry["support_environment_steps_by_task"],
            "pearl": {"strict_mean": entry["pearl"]["strict_mean"]},
            **{
                method: {key: entry[method][key] for key in ("strict_mean", "strict_std", "strict_mean_by_seed")}
                for method in ("scratch_sac", "pooled_finetune_sac")
            },
        }
    return {key: result[key] for key in ("schema", "taskbook_hash", "checkpoint", "pooled_model", "split", "protocol")} | {"budgets": budgets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--pooled-model", required=True, help="Frozen pooled SAC pre-training checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260725, 20260726, 20260727], help="Independent online SAC seeds")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--resume", action="store_true", help="retain completed K budgets in --output")
    parser.add_argument("--detailed-output", action="store_true", help="retain per-query records and resumable state")
    args = parser.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

    config = read_config(args.config)
    taskbook = load_taskbook(args.taskbook)
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    tasks = list(taskbook["meta_test_logical"])
    casebooks = {task.task_id: load_casebook(task, args.casebook_root) for task in tasks}
    device = torch.device("cuda" if torch.cuda.is_available() and config["experiment"].get("device") != "cpu" else "cpu")
    agent = PEARLAgent(int(config["environment"]["observation_dim"]), int(config["environment"]["action_dim"]), config, device)
    checkpoint = load_checkpoint(args.checkpoint, agent, device)
    if checkpoint["taskbook_hash"] != taskbook_hash:
        raise ValueError("PEARL checkpoint belongs to a different frozen taskbook")
    _restore_checkpoint_rng(checkpoint)
    pearl = evaluate_fewshot(agent, config, tasks, casebooks, "meta_test_logical", provenance={"evaluation_rng": "checkpoint_rng_state"})
    if pearl["parameter_hash_before"] != pearl["parameter_hash_after"]:
        raise RuntimeError("few-shot profiling changed PEARL parameters")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / ".equal_budget_state.json"
    result: dict[str, Any] = {
        "schema": "pearl_equal_new_task_budget",
        "taskbook_hash": taskbook_hash,
        "checkpoint": str(args.checkpoint),
        "pooled_model": str(args.pooled_model),
        "split": "meta_test_logical",
        "protocol": {
            "budget": "per-task cumulative environment steps observed in PEARL's first K frozen support episodes",
            "support_cases": "the same frozen test_support pool; online baselines cycle only the first K cases",
            "query_cases": "the same disjoint frozen test_query cases used by PEARL",
            "scratch_initialization": "random",
            "pooled_finetune_initialization": "frozen pooled SAC checkpoint",
            "online_sac_updates": "one gradient update after each environment step; favors online SAC baselines computationally",
            "online_sac_seeds": [int(seed) for seed in args.seeds],
            "pearl_parameter_updates": 0,
        },
        "pearl": pearl,
        "budgets": {},
    }
    if args.resume:
        if not args.detailed_output:
            raise ValueError("--resume requires --detailed-output")
        if not state_path.exists():
            raise ValueError("no resumable detailed state exists")
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing.get("schema") != result["schema"] or existing.get("taskbook_hash") != taskbook_hash:
            raise ValueError("resume artifact belongs to a different equal-budget protocol or taskbook")
        if existing.get("checkpoint") != str(args.checkpoint) or existing.get("pooled_model") != str(args.pooled_model):
            raise ValueError("resume artifact uses different PEARL or pooled checkpoints")
        if existing.get("protocol", {}).get("online_sac_seeds") != result["protocol"]["online_sac_seeds"]:
            raise ValueError("resume artifact uses different online SAC seeds")
        result = existing
    requested = sorted(set(int(shot) for shot in args.shots))
    if any(shot < 1 or str(shot) not in next(iter(pearl["tasks"].values())) for shot in requested):
        raise ValueError("shots must be positive entries in evaluation.shots")
    for shot in requested:
        budget_by_task = {task.task_id: int(pearl["tasks"][task.task_id][str(shot)]["support_environment_steps"]) for task in tasks}
        pearl_tasks = {task.task_id: pearl["tasks"][task.task_id][str(shot)] for task in tasks}
        entry = result["budgets"].setdefault(str(shot), {
            "support_environment_steps_by_task": budget_by_task,
            "mean_support_environment_steps": float(np.mean(list(budget_by_task.values()))),
            "pearl": {"strict_mean": _mean_strict(pearl_tasks), "tasks": pearl_tasks},
            "online_by_seed": {},
        })
        if entry["support_environment_steps_by_task"] != budget_by_task:
            raise RuntimeError(f"resume support budget mismatch at K={shot}")
        for online_seed in args.seeds:
            seed_key = str(online_seed)
            if seed_key in entry["online_by_seed"]:
                continue
            scratch_tasks: dict[str, Any] = {}
            finetune_tasks: dict[str, Any] = {}
            scratch_steps: dict[str, int] = {}
            finetune_steps: dict[str, int] = {}
            for index, task in enumerate(tasks):
                budget = budget_by_task[task.task_id]
                support = casebooks[task.task_id]["test_support"][:shot]
                if not support or budget < 1:
                    raise RuntimeError(f"task {task.task_id} has no usable support budget at K={shot}")
                scratch_env = FixedSupportCases(LogicalMergeEnv(task, config, support), support)
                try:
                    scratch = _new_online_sac(scratch_env, online_seed + index + 1000 * shot)
                    before_steps = int(scratch.num_timesteps)
                    scratch.learn(total_timesteps=budget)
                    scratch_steps[task.task_id] = int(scratch.num_timesteps) - before_steps
                finally:
                    scratch_env.close()
                if scratch_steps[task.task_id] != budget:
                    raise RuntimeError(f"scratch budget mismatch for {task.task_id}: {scratch_steps[task.task_id]} != {budget}")
                scratch_tasks[task.task_id] = _evaluate_sac(scratch, task, config, casebooks[task.task_id]["test_query"])

                finetune_env = FixedSupportCases(LogicalMergeEnv(task, config, support), support)
                try:
                    finetune = SAC.load(args.pooled_model, env=finetune_env, device="auto")
                    finetune.set_random_seed(online_seed + 500 + index + 1000 * shot)
                    finetune.learning_starts = 1
                    finetune.batch_size = 8
                    finetune.train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)
                    finetune.gradient_steps = 1
                    before_steps = int(finetune.num_timesteps)
                    finetune.learn(total_timesteps=budget, reset_num_timesteps=False)
                    finetune_steps[task.task_id] = int(finetune.num_timesteps) - before_steps
                finally:
                    finetune_env.close()
                if finetune_steps[task.task_id] != budget:
                    raise RuntimeError(f"finetune budget mismatch for {task.task_id}: {finetune_steps[task.task_id]} != {budget}")
                finetune_tasks[task.task_id] = _evaluate_sac(finetune, task, config, casebooks[task.task_id]["test_query"])
            entry["online_by_seed"][seed_key] = {
                "scratch_sac": {"strict_mean": _mean_strict(scratch_tasks), "tasks": scratch_tasks, "environment_steps_by_task": scratch_steps},
                "pooled_finetune_sac": {"strict_mean": _mean_strict(finetune_tasks), "tasks": finetune_tasks, "environment_steps_by_task": finetune_steps},
            }
            write_json(state_path, result)
        for method in ("scratch_sac", "pooled_finetune_sac"):
            values = [float(entry["online_by_seed"][str(seed)][method]["strict_mean"]) for seed in args.seeds]
            entry[method] = {
                "strict_mean": float(np.mean(values)), "strict_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "strict_mean_by_seed": {str(seed): float(entry["online_by_seed"][str(seed)][method]["strict_mean"]) for seed in args.seeds},
            }
        write_json(state_path, result)
    metrics_path = output / ("equal_budget_metrics.json" if args.detailed_output else "equal_budget_summary.json")
    write_json(metrics_path, result if args.detailed_output else _compact_result(result))
    if not args.detailed_output:
        state_path.unlink(missing_ok=True)
    print(metrics_path)


if __name__ == "__main__":
    main()
