"""PEARL trainer with prior→posterior collection per sampled task."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .collector import collect_episode
from .evaluator import evaluate_fewshot, validation_score
from .gates import verify_formal_gate
from .io import content_hash, write_json
from .pearl_agent import PEARLAgent
from .replay import TaskReplayBuffers
from .task_env import LogicalMergeEnv


def train(config: Mapping[str, Any], tasks: list[Any], validation_tasks: list[Any], casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]],
          taskbook_hash: str, max_env_steps: int, seed: int, run_name: str, smoke: bool = False,
          gate_manifest: str | None = None, resume_checkpoint: str | None = None,
          checkpoint_interval_steps: int | None = None) -> Path:
    if not smoke:
        verify_formal_gate(gate_manifest, taskbook_hash)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config["experiment"].get("device") != "cpu" else "cpu")
    observation_dim, action_dim = int(config["environment"]["observation_dim"]), int(config["environment"]["action_dim"])
    agent = PEARLAgent(observation_dim, action_dim, config, device)
    root = Path(config["project"]["output_root"]) / ("smoke" if smoke else "models") / run_name
    root.mkdir(parents=True, exist_ok=True)
    casebook_hashes = {task_id: content_hash(book) for task_id, book in casebooks.items()}
    write_json(root / "config_resolved.json", dict(config))
    steps, progress, episode_counter, best_score = 0, [], 0, None
    validation_interval, next_validation = int(config["meta_training"]["validation_interval_steps"]), int(config["meta_training"]["validation_interval_steps"])
    buffers = TaskReplayBuffers([task.task_id for task in tasks])
    if resume_checkpoint:
        checkpoint = load_checkpoint(resume_checkpoint, agent, device)
        if checkpoint["taskbook_hash"] != taskbook_hash or checkpoint["config_hash"] != content_hash(config):
            raise ValueError("resume checkpoint does not match the frozen taskbook/configuration")
        state = checkpoint.get("trainer_state")
        if not isinstance(state, Mapping):
            raise ValueError("resume checkpoint lacks trainer replay/progress state")
        buffers = state["buffers"]
        if set(buffers.buffers) != {task.task_id for task in tasks}:
            raise ValueError("resume checkpoint replay tasks do not match the requested meta-train tasks")
        steps = int(state["steps"]); progress = list(state["progress"]); episode_counter = int(state["episode_counter"])
        best_score = state.get("best_score"); next_validation = int(state["next_validation"])
        rng.bit_generator.state = checkpoint["rng_state"]["numpy_generator"]
        cpu_rng = torch.as_tensor(checkpoint["rng_state"]["torch"], dtype=torch.uint8, device="cpu").clone()
        torch.set_rng_state(cpu_rng)
        if torch.cuda.is_available() and checkpoint["rng_state"].get("cuda") is not None:
            torch.cuda.set_rng_state_all([torch.as_tensor(state, dtype=torch.uint8, device="cpu").clone() for state in checkpoint["rng_state"]["cuda"]])
    checkpoint_interval = int(checkpoint_interval_steps) if checkpoint_interval_steps else None
    next_checkpoint = ((steps // checkpoint_interval) + 1) * checkpoint_interval if checkpoint_interval else None

    def rng_state() -> dict[str, Any]:
        return {"numpy_generator": rng.bit_generator.state, "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def trainer_state() -> dict[str, Any]:
        return {"buffers": buffers, "steps": steps, "progress": progress, "episode_counter": episode_counter,
                "best_score": best_score, "next_validation": next_validation}

    def save(path: Path) -> None:
        save_checkpoint(path, agent, config, taskbook_hash, steps, casebook_hashes=casebook_hashes,
                        training_seed=seed, rng_state=rng_state(), trainer_state=trainer_state())

    def collect(task: Any, case: dict[str, Any], z: torch.Tensor, mode: str, posterior_version: int):
        nonlocal episode_counter
        episode_counter += 1
        collection_config = config
        regularization = config.get("regularization", {})
        warmup_steps = int(regularization.get("topology_dropout_warmup_steps", 0))
        if warmup_steps and steps >= warmup_steps:
            # Preserve the immutable resolved configuration in checkpoints
            # while ending descriptor masking after the scheduled warm-up.
            collection_config = {
                **config,
                "regularization": {**regularization, "topology_dropout_probability": 0.0},
            }
        env = LogicalMergeEnv(task, collection_config, casebooks[task.task_id]["train_pool"])
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

    if not resume_checkpoint:
        bootstrap = 1 if smoke else int(config["meta_training"]["bootstrap_episodes_per_task"])
        for task in tasks:
            for _ in range(bootstrap):
                steps += collect_prior_posterior_pair(task)
                if steps >= max_env_steps:
                    break

    batch_size = min(32, int(config["sac"]["batch_size"])) if smoke else int(config["sac"]["batch_size"])
    context_size = min(64, int(config["pearl"]["context_batch_size"])) if smoke else int(config["pearl"]["context_batch_size"])
    transitions_per_episode = int(config["pearl"]["context_transitions_per_episode"])
    max_context_episodes = max(1, context_size // transitions_per_episode)
    min_context_episodes = min(max_context_episodes, max(1, int(config["pearl"].get("context_min_episodes", 1))))
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
            # Few-shot evaluation starts at one support episode and grows to
            # the configured context capacity.  Training only at the maximum
            # capacity makes posterior inference brittle for K=1/2, so expose
            # the encoder to the entire evaluation support-count range.
            context_episodes = int(rng.integers(min_context_episodes, max_context_episodes + 1))
            context = buffers.context_per_task(
                [task.task_id for task in selected],
                context_episodes * transitions_per_episode,
                transitions_per_episode,
                rng,
            )
            rl = buffers.sample_per_task([task.task_id for task in selected], batch_size, rng)
            metrics = agent.update(context, rl)
            progress.append({**metrics, "environment_steps": steps, "buffer_tasks": len(ready)})
        if validation_tasks and steps >= next_validation:
            validation = evaluate_fewshot(agent, config, validation_tasks, casebooks, "meta_validation")
            score = validation_score(validation)
            if best_score is None or score > best_score:
                best_score = score
                save(root / "best_model.pt")
            next_validation += validation_interval
        if next_checkpoint is not None and steps >= next_checkpoint:
            save(root / "last_model.pt")
            next_checkpoint = ((steps // checkpoint_interval) + 1) * checkpoint_interval
    if best_score is None:
        save(root / "best_model.pt")
    write_json(root / "training_summary.json", {
        "environment_steps": steps,
        "gradient_updates": len(progress),
        "best_validation_score": best_score,
        "selected_checkpoint": "best_model.pt",
    })
    return root
