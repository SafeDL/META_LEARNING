"""PEARL trainer with prior→posterior collection per sampled task."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

from .checkpoint import save_checkpoint
from .collector import collect_episode
from .evaluator import evaluate_fewshot, validation_score
from .gates import verify_formal_gate
from .io import content_hash, write_json
from .pearl_agent import PEARLAgent
from .replay import TaskReplayBuffers
from .task_env import LogicalMergeEnv


def train(config: Mapping[str, Any], tasks: list[Any], validation_tasks: list[Any], casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]],
          taskbook_hash: str, max_env_steps: int, seed: int, run_name: str, smoke: bool = False,
          gate_manifest: str | None = None) -> Path:
    if not smoke:
        verify_formal_gate(gate_manifest, taskbook_hash)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config["experiment"].get("device") != "cpu" else "cpu")
    observation_dim, action_dim = int(config["environment"]["observation_dim"]), int(config["environment"]["action_dim"])
    agent = PEARLAgent(observation_dim, action_dim, config, device)
    buffers = TaskReplayBuffers([task.task_id for task in tasks])
    root = Path(config["project"]["output_root"]) / ("smoke" if smoke else "runs") / run_name
    root.mkdir(parents=True, exist_ok=True)
    casebook_hashes = {task_id: content_hash(book) for task_id, book in casebooks.items()}
    write_json(root / "config_resolved.json", dict(config))
    write_json(root / "provenance.json", {"taskbook_hash": taskbook_hash, "casebook_hashes": casebook_hashes, "training_seed": seed})
    steps, progress, episode_counter, best_score = 0, [], 0, None
    validation_interval, next_validation = int(config["meta_training"]["validation_interval_steps"]), int(config["meta_training"]["validation_interval_steps"])

    def collect(task: Any, case: dict[str, Any], z: torch.Tensor, mode: str, posterior_version: int):
        nonlocal episode_counter
        episode_counter += 1
        env = LogicalMergeEnv(task, config, casebooks[task.task_id]["train_pool"])
        try:
            return collect_episode(env, task, case, agent, z, mode, device, episode_id=f"{task.task_id}:{episode_counter:08d}", posterior_version=posterior_version)
        finally:
            env.close()

    def collect_prior_posterior_pair(task: Any) -> int:
        """Clear per-adaptation context, then collect one prior and posterior rollout."""
        book = casebooks[task.task_id]["train_pool"]
        prior_case = book[int(rng.integers(len(book)))]
        prior_mu, prior_log_var = agent.prior()
        prior = collect(task, prior_case, agent.sample_latent(prior_mu, prior_log_var), "prior_support", 0)
        buffers.add_episode(task.task_id, prior.transitions)
        posterior_mu, posterior_log_var = agent.infer_posterior([[prior.transitions]])
        posterior_case = book[int(rng.integers(len(book)))]
        posterior = collect(task, posterior_case, agent.sample_latent(posterior_mu, posterior_log_var), "posterior_rollout", 1)
        buffers.add_episode(task.task_id, posterior.transitions)
        return len(prior.transitions) + len(posterior.transitions)

    bootstrap = 1 if smoke else int(config["meta_training"]["bootstrap_episodes_per_task"])
    for task in tasks:
        for _ in range(bootstrap):
            steps += collect_prior_posterior_pair(task)
            if steps >= max_env_steps:
                break

    batch_size = min(32, int(config["sac"]["batch_size"])) if smoke else int(config["sac"]["batch_size"])
    context_size = min(64, int(config["pearl"]["context_batch_size"])) if smoke else int(config["pearl"]["context_batch_size"])
    transitions_per_episode = int(config["pearl"]["context_transitions_per_episode"])
    updates_per_iteration = 1 if smoke else int(config["meta_training"]["gradient_updates_per_iteration"])
    while steps < max_env_steps:
        sampled = [tasks[int(rng.integers(len(tasks)))] for _ in range(min(int(config["meta_training"]["meta_batch_size"]), len(tasks)))]
        sampled = list({task.task_id: task for task in sampled}.values())
        for task in sampled:
            steps += collect_prior_posterior_pair(task)
            if steps >= max_env_steps:
                break
        ready = [task for task in tasks if len(buffers.buffers[task.task_id]) >= max(batch_size, 2)]
        for _ in range(updates_per_iteration if ready else 0):
            selected = list(rng.choice(ready, size=min(len(ready), int(config["meta_training"]["meta_batch_size"])), replace=False))
            context = buffers.context_per_task([task.task_id for task in selected], context_size, transitions_per_episode, rng)
            rl = buffers.sample_per_task([task.task_id for task in selected], batch_size, rng)
            metrics = agent.update(context, rl)
            progress.append({**metrics, "environment_steps": steps, "buffer_tasks": len(ready)})
        if validation_tasks and steps >= next_validation:
            validation = evaluate_fewshot(agent, config, validation_tasks, casebooks, "meta_validation")
            validation_root = root / "validation" / f"step_{steps}"
            write_json(validation_root / "fewshot.json", validation)
            score = validation_score(validation)
            if best_score is None or score > best_score:
                best_score = score
                save_checkpoint(root / "best_model.pt", agent, config, taskbook_hash, steps, casebook_hashes=casebook_hashes, training_seed=seed, rng_state={"numpy_generator": rng.bit_generator.state, "torch": torch.get_rng_state()})
            next_validation += validation_interval
    save_checkpoint(root / "final_model.pt", agent, config, taskbook_hash, steps, casebook_hashes=casebook_hashes, training_seed=seed, rng_state={"numpy_generator": rng.bit_generator.state, "torch": torch.get_rng_state()})
    if best_score is None:
        save_checkpoint(root / "best_model.pt", agent, config, taskbook_hash, steps, casebook_hashes=casebook_hashes, training_seed=seed, rng_state={"numpy_generator": rng.bit_generator.state, "torch": torch.get_rng_state()})
    write_json(root / "train_progress.json", progress)
    return root
