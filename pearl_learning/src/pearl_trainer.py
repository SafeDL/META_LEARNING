"""Actual MetaDrive PEARL meta-training loop with task-separated buffers."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from .casebook import build_casebook, save_casebook
from .checkpoint import save_checkpoint
from .collector import collect_episode
from .evaluator import evaluate_fewshot, validation_score
from .io import content_hash, write_json
from .pearl_agent import PEARLAgent
from .replay import TaskReplayBuffers
from .task_env import LogicalMergeEnv


def train(config: Mapping[str, Any], tasks: list[Any], validation_tasks: list[Any], max_env_steps: int, seed: int, run_name: str, smoke: bool = False) -> Path:
    rng = np.random.default_rng(seed); torch.manual_seed(seed); device = torch.device("cuda" if torch.cuda.is_available() and config["experiment"].get("device") != "cpu" else "cpu")
    taskbooks = {task.task_id: build_casebook(task, config) for task in tasks}
    observation_dim = int(config["environment"]["observation_dim"])
    action_dim = int(config["environment"]["action_dim"])
    agent = PEARLAgent(observation_dim, action_dim, config, device); buffers = TaskReplayBuffers([task.task_id for task in tasks]); root = Path(config["project"]["output_root"]) / run_name; root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config_resolved.json", dict(config)); write_json(root / "versions.json", {"torch": torch.__version__, "device": str(device), "metadrive": "0.4.3"})
    all_task_hash = content_hash([task.to_dict() for task in tasks + validation_tasks])
    write_json(root / "tasks.json", {"meta_train": [task.to_dict() for task in tasks], "meta_validation": [task.to_dict() for task in validation_tasks], "taskbook_hash": all_task_hash})
    for task in tasks + validation_tasks:
        save_casebook(task, taskbooks.get(task.task_id, build_casebook(task, config)), str(root))
    steps = 0; progress = []; best_score = None
    validation_interval = int(config["meta_training"]["validation_interval_steps"])
    next_validation = validation_interval
    def collect(task: Any, case: dict[str, Any], z: torch.Tensor, mode: str):
        # MetaDrive 0.4.3 owns a process-global engine.  Creating/closing one
        # task facade per full episode prevents cross-task engine state leaks.
        env = LogicalMergeEnv(task, config, taskbooks[task.task_id]["train_pool"])
        try:
            return collect_episode(env, task, case, agent, z, mode, device)
        finally:
            env.close()
    try:
        bootstrap_episodes = 1 if smoke else int(config["meta_training"]["bootstrap_episodes_per_task"])
        # Bootstrap each task from q(z|empty); a support episode's z is fixed.
        for task in tasks:
            for case in taskbooks[task.task_id]["train_pool"][:bootstrap_episodes]:
                mu, log_var = agent.prior()
                rollout = collect(task, case, agent.sample_latent(mu, log_var), "prior_support")
                buffers.add_episode(task.task_id, rollout.transitions)
                steps += len(rollout.transitions)
        batch_size = min(32, int(config["sac"]["batch_size"])) if smoke else int(config["sac"]["batch_size"])
        context_size = min(16, int(config["pearl"]["context_batch_size"])) if smoke else int(config["pearl"]["context_batch_size"])
        updates_per_iteration = 1 if smoke else int(config["meta_training"]["gradient_updates_per_iteration"])
        while steps < max_env_steps:
            sampled = [tasks[int(rng.integers(len(tasks)))] for _ in range(min(int(config["meta_training"]["meta_batch_size"]), len(tasks)))]
            sampled = list({task.task_id: task for task in sampled}.values())
            for task in sampled:
                buf = buffers.buffers[task.task_id]
                context = [buf.sample_context(min(context_size, len(buf)), rng)]
                mu, log_var = agent.infer_posterior(context); case = taskbooks[task.task_id]["train_pool"][int(rng.integers(len(taskbooks[task.task_id]["train_pool"])))]
                rollout = collect(task, case, agent.sample_latent(mu, log_var), "posterior_support"); buffers.add_episode(task.task_id, rollout.transitions); steps += len(rollout.transitions)
                if steps >= max_env_steps: break
            ready = [task for task in tasks if len(buffers.buffers[task.task_id]) >= max(batch_size, 2)]
            if ready:
                for _ in range(updates_per_iteration):
                    selected = rng.choice(ready, size=min(len(ready), int(config["meta_training"]["meta_batch_size"])), replace=False)
                    update_tasks = list(selected)
                    context = buffers.context_per_task([x.task_id for x in update_tasks], min(context_size, min(len(buffers.buffers[x.task_id]) for x in update_tasks)), rng)
                    rl = buffers.sample_per_task([x.task_id for x in update_tasks], batch_size, rng)
                    metrics = agent.update(context, rl)
                    metrics.update({"environment_steps": steps, "buffer_tasks": len(ready)})
                    progress.append(metrics)
            if validation_tasks and steps >= next_validation:
                validation = evaluate_fewshot(agent, config, validation_tasks, "meta_validation")
                validation_root = root / "validation" / f"step_{steps}"
                write_json(validation_root / "fewshot.json", validation)
                score = validation_score(validation)
                if best_score is None or score > best_score:
                    best_score = score
                    save_checkpoint(root / "best_model.pt", agent, config, all_task_hash, steps)
                next_validation += validation_interval
        save_checkpoint(root / "final_model.pt", agent, config, all_task_hash, steps)
        if best_score is None:
            save_checkpoint(root / "best_model.pt", agent, config, all_task_hash, steps)
        write_json(root / "train_progress.json", progress)
        return root
    finally:
        # No persistent MetaDrive environment is retained across tasks.
        pass
