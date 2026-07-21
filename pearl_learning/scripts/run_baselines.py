"""Run baselines under one frozen taskbook, casebooks, and metric protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from pearl_learning.src.baselines import (
    BASELINE_NAMES, OracleTaskObservation, PooledLogicalMergeEnv, write_baseline_manifest,
)
from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.checkpoint import load_checkpoint
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.metrics import summarize
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.task_env import LogicalMergeEnv
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _evaluation_tasks(taskbook: Mapping[str, list[Any]]) -> list[Any]:
    return list(taskbook["meta_test_template"]) + list(taskbook["meta_test_logical"])


def _evaluate_sac(model: Any, task: Any, config: Mapping[str, Any], cases: list[Mapping[str, Any]],
                  transform: Callable[[np.ndarray, Any], np.ndarray] | None = None) -> dict[str, Any]:
    """Evaluate one deterministic policy on the frozen query cases only."""
    env = LogicalMergeEnv(task, config, cases)
    records: list[dict[str, Any]] = []
    try:
        for case in cases:
            observation, _ = env.reset(options={"case": case})
            terminated = truncated = False
            while not (terminated or truncated):
                model_input = transform(observation, task) if transform else observation
                action, _ = model.predict(model_input, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
            records.append(env.episode_record())
    finally:
        env.close()
    return {"summary": summarize(records), "records": records}


def _evaluate_no_context(agent: PEARLAgent, task: Any, config: Mapping[str, Any],
                         cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    """The PEARL ablation: query every episode from the fixed unit-normal prior."""
    env = LogicalMergeEnv(task, config, cases)
    records: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            mu, _ = agent.prior()
            for case in cases:
                observation, _ = env.reset(options={"case": case})
                terminated = truncated = False
                while not (terminated or truncated):
                    obs = torch.as_tensor(observation[None], dtype=torch.float32, device=agent.device)
                    action = agent.act(obs, mu, deterministic=True)[0].detach().cpu().numpy()
                    observation, _, terminated, truncated, _ = env.step(action)
                records.append(env.episode_record())
    finally:
        env.close()
    return {"summary": summarize(records), "records": records, "context": "unit_normal_prior"}


def _save_metrics(output: Path, payload: Mapping[str, Any]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, dict(payload))
    return output


def _new_sac(env: Any, seed: int) -> Any:
    from stable_baselines3 import SAC
    return SAC("MlpPolicy", env, seed=seed, verbose=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--taskbook", required=True); parser.add_argument("--casebook-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", choices=BASELINE_NAMES, required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--env-steps", type=int, required=True)
    parser.add_argument("--pretrain-steps", type=int, help="pooled pre-training budget; defaults to --env-steps")
    parser.add_argument("--pearl-checkpoint", help="required by pearl_no_context")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    taskbook = load_taskbook(args.taskbook)
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    train_tasks, evaluation_tasks = taskbook["meta_train"], _evaluation_tasks(taskbook)
    all_tasks = list(train_tasks) + list(taskbook["meta_validation"]) + evaluation_tasks
    casebooks = {task.task_id: load_casebook(task, args.casebook_root) for task in all_tasks}
    case_hashes = {task_id: content_hash(book) for task_id, book in casebooks.items()}
    query_cases = lambda task: casebooks[task.task_id]["test_query"][:1 if args.smoke else None]
    root = Path(args.output) / args.baseline
    artifacts: dict[str, str] = {}
    checkpoint_hash: str | None = None

    if args.baseline == "pearl_no_context":
        if not args.pearl_checkpoint:
            raise SystemExit("pearl_no_context requires --pearl-checkpoint")
        device = torch.device("cuda" if torch.cuda.is_available() and cfg["experiment"].get("device") != "cpu" else "cpu")
        agent = PEARLAgent(int(cfg["environment"]["observation_dim"]), int(cfg["environment"]["action_dim"]), cfg, device)
        checkpoint = load_checkpoint(args.pearl_checkpoint, agent, device)
        if checkpoint["taskbook_hash"] != taskbook_hash:
            raise SystemExit("PEARL checkpoint belongs to a different frozen taskbook")
        before = agent.parameter_hash()
        result = {task.task_id: _evaluate_no_context(agent, task, cfg, query_cases(task)) for task in evaluation_tasks}
        if agent.parameter_hash() != before:
            raise RuntimeError("no-context baseline changed PEARL parameters")
        checkpoint_hash = json.loads(Path(args.pearl_checkpoint).with_suffix(".manifest.json").read_text(encoding="utf-8"))["checkpoint_hash"]
        artifacts["metrics"] = str(_save_metrics(root / "no_context_metrics.json", {"protocol": "fixed_unit_normal_prior", "tasks": result}))

    else:
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise SystemExit("stable_baselines3 is required for SAC baselines") from exc

        train_books = {task.task_id: casebooks[task.task_id] for task in train_tasks}
        if args.baseline in {"per_task_sac", "cross_task_policy_matrix"}:
            policies: dict[str, Any] = {}
            for index, task in enumerate(train_tasks):
                env = LogicalMergeEnv(task, cfg, train_books[task.task_id]["train_pool"])
                try:
                    model = _new_sac(env, args.seed + index); model.learn(total_timesteps=args.env_steps)
                    target = root / "policies" / f"{task.task_id}.zip"; target.parent.mkdir(parents=True, exist_ok=True); model.save(target)
                    artifacts[f"policy:{task.task_id}"] = str(target)
                    policies[task.task_id] = model
                finally:
                    env.close()
            if args.baseline == "per_task_sac":
                metrics = {task.task_id: _evaluate_sac(policies[task.task_id], task, cfg, query_cases(task)) for task in train_tasks}
                artifacts["metrics"] = str(_save_metrics(root / "per_task_metrics.json", {"tasks": metrics}))
            else:
                matrix = {
                    policy_id: {task.task_id: _evaluate_sac(model, task, cfg, query_cases(task)) for task in evaluation_tasks}
                    for policy_id, model in policies.items()
                }
                artifacts["matrix"] = str(_save_metrics(root / "cross_task_matrix.json", {"policy_tasks": list(policies), "evaluation_tasks": [task.task_id for task in evaluation_tasks], "matrix": matrix}))

        elif args.baseline in {"topology_conditioned_pooled_sac", "oracle_task_conditioned_sac"}:
            pooled = PooledLogicalMergeEnv(train_tasks, cfg, train_books, args.seed)
            geometry_ids = [task.geometry_id for task in all_tasks]
            env = OracleTaskObservation(pooled, geometry_ids) if args.baseline == "oracle_task_conditioned_sac" else pooled
            try:
                model = _new_sac(env, args.seed); model.learn(total_timesteps=args.env_steps)
                target = root / "model.zip"; target.parent.mkdir(parents=True, exist_ok=True); model.save(target); artifacts["model"] = str(target)
            finally:
                env.close()
            transform = None
            if args.baseline == "oracle_task_conditioned_sac":
                positions = {geometry_id: index for index, geometry_id in enumerate(geometry_ids)}
                transform = lambda observation, task: np.concatenate([np.asarray(observation, dtype=np.float32), np.eye(len(geometry_ids), dtype=np.float32)[positions[task.geometry_id]]])
            metrics = {task.task_id: _evaluate_sac(model, task, cfg, query_cases(task), transform) for task in evaluation_tasks}
            artifacts["metrics"] = str(_save_metrics(root / "heldout_metrics.json", {"tasks": metrics}))

        elif args.baseline == "scratch_sac":
            metrics = {}
            for index, task in enumerate(evaluation_tasks):
                env = LogicalMergeEnv(task, cfg, casebooks[task.task_id]["test_support"])
                try:
                    model = _new_sac(env, args.seed + index); model.learn(total_timesteps=args.env_steps)
                    target = root / "policies" / f"{task.task_id}.zip"; target.parent.mkdir(parents=True, exist_ok=True); model.save(target); artifacts[f"policy:{task.task_id}"] = str(target)
                finally:
                    env.close()
                metrics[task.task_id] = _evaluate_sac(model, task, cfg, query_cases(task))
            artifacts["metrics"] = str(_save_metrics(root / "scratch_metrics.json", {"support_steps": args.env_steps, "tasks": metrics}))

        elif args.baseline == "pooled_finetune_sac":
            pretrain_steps = int(args.pretrain_steps or args.env_steps)
            pooled = PooledLogicalMergeEnv(train_tasks, cfg, train_books, args.seed)
            try:
                base = _new_sac(pooled, args.seed); base.learn(total_timesteps=pretrain_steps)
                base_path = root / "pooled_pretrain.zip"; base_path.parent.mkdir(parents=True, exist_ok=True); base.save(base_path); artifacts["pooled_pretrain"] = str(base_path)
            finally:
                pooled.close()
            metrics = {}
            for index, task in enumerate(evaluation_tasks):
                env = LogicalMergeEnv(task, cfg, casebooks[task.task_id]["test_support"])
                try:
                    model = SAC.load(base_path, env=env, device="auto"); model.set_random_seed(args.seed + index); model.learn(total_timesteps=args.env_steps, reset_num_timesteps=False)
                    target = root / "finetuned" / f"{task.task_id}.zip"; target.parent.mkdir(parents=True, exist_ok=True); model.save(target); artifacts[f"finetuned:{task.task_id}"] = str(target)
                finally:
                    env.close()
                metrics[task.task_id] = _evaluate_sac(model, task, cfg, query_cases(task))
            artifacts["metrics"] = str(_save_metrics(root / "pooled_finetune_metrics.json", {"pretrain_steps": pretrain_steps, "support_steps": args.env_steps, "tasks": metrics}))

        else:
            raise RuntimeError(f"unhandled baseline {args.baseline}")

    write_baseline_manifest(
        args.output, name=args.baseline, taskbook_hash=taskbook_hash, seed=args.seed, env_steps=args.env_steps,
        smoke=args.smoke, artifacts=artifacts, config_hash=content_hash(cfg), casebook_hashes=case_hashes,
        checkpoint_hash=checkpoint_hash,
    )


if __name__ == "__main__":
    main()
