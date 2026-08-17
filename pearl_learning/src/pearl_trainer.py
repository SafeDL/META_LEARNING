"""PEARL trainer with prior→posterior collection per sampled task."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .collector import collect_episode
from .evaluator import evaluate_fewshot, validation_score
from .formal_validation import verify_formal_validation
from .io import content_hash, write_json
from .pearl_agent import PEARLAgent
from .replay import TaskReplayBuffers
from .task_env import LogicalMergeEnv
from .task_env import freeze_physical_task_descriptor
from .task_representation import representation_target


def _training_context_episode_count(
    buffers: TaskReplayBuffers,
    task_ids: list[str],
    minimum: int,
    maximum: int,
    rng: np.random.Generator,
) -> int:
    """Choose one shape-safe episode count shared by a meta-batch."""
    if not task_ids:
        raise ValueError("a training context batch requires at least one task")
    # Keep at least one full-replay episode outside the context so the SAC
    # batch is strictly disjoint from the encoder evidence.
    available = min(
        min(
            len(buffers.recent_context_buffers[task_id].episodes),
            len(buffers.buffers[task_id].episodes) - 1,
        )
        for task_id in task_ids
    )
    if available < 1:
        raise RuntimeError("all sampled tasks need at least one complete replay episode")
    upper = min(int(maximum), available)
    lower = min(int(minimum), upper)
    return int(rng.integers(lower, upper + 1))


def _sample_tasks_without_replacement(
    tasks: list[Any],
    count: int,
    rng: np.random.Generator,
) -> list[Any]:
    if not tasks:
        raise ValueError("meta-training requires at least one task")
    if int(count) < 1:
        raise ValueError("meta-batch size must be positive")
    size = min(len(tasks), int(count))
    return list(rng.choice(tasks, size=size, replace=False))


def train(
    config: Mapping[str, Any],
    tasks: list[Any],
    validation_tasks: list[Any],
    casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    taskbook_hash: str,
    max_env_steps: int,
    seed: int,
    run_name: str,
    smoke: bool = False,
    formal_validation: str | None = None,
    resume_checkpoint: str | None = None,
    checkpoint_interval_steps: int | None = None,
    mechanism_gate: bool = False,
) -> Path:
    if not smoke and not mechanism_gate:
        verify_formal_validation(formal_validation, taskbook_hash)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["experiment"].get("device") != "cpu" else "cpu"
    )
    observation_dim = int(config["environment"]["observation_dim"])
    action_dim = int(config["environment"]["action_dim"])
    agent = PEARLAgent(observation_dim, action_dim, config, device)
    semantic_targets = None
    if bool(config.get("task_representation", {}).get("enabled", False)):
        semantic_targets = {task.task_id: representation_target(task) for task in tasks}
    run_directory = "smoke" if smoke else ("mechanism_gate" if mechanism_gate else "models")
    root = Path(config["project"]["output_root"]) / run_directory / run_name
    root.mkdir(parents=True, exist_ok=True)
    router_audit_path = root / "router_audit.jsonl"
    training_update_path = root / "training_updates.jsonl"
    if not resume_checkpoint and router_audit_path.exists():
        router_audit_path.unlink()
    if not resume_checkpoint and training_update_path.exists():
        training_update_path.unlink()
    task_descriptors = {}
    if agent.actor_architecture == "posterior_routed_moe":
        task_descriptors = {
            task.task_id: freeze_physical_task_descriptor(
                task,
                config,
                casebooks[task.task_id]["train_pool"],
            )
            for task in tasks
        }
    scenario_tasks = {task.task_id: task for task in tasks}
    casebook_hashes = {task_id: content_hash(book) for task_id, book in casebooks.items()}
    write_json(root / "config_resolved.json", dict(config))
    steps = 0
    gradient_updates = 0
    episode_counter = 0
    best_score = None
    validation_interval = int(config["meta_training"]["validation_interval_steps"])
    next_validation = validation_interval
    buffers = TaskReplayBuffers(
        [task.task_id for task in tasks],
        capacity=int(config["meta_training"].get("replay_capacity_transitions", 200_000)),
        recent_context_episodes=int(config["meta_training"]["recent_context_episodes_per_task"]),
    )
    context_refresh_interval = int(config["meta_training"]["context_refresh_interval_updates"])
    if context_refresh_interval < 1:
        raise ValueError("context_refresh_interval_updates must be positive")
    next_context_refresh = context_refresh_interval
    if resume_checkpoint:
        checkpoint = load_checkpoint(resume_checkpoint, agent, device)
        if checkpoint["taskbook_hash"] != taskbook_hash or checkpoint["config_hash"] != content_hash(config):
            raise ValueError("resume checkpoint does not match the frozen taskbook/configuration")
        state = checkpoint.get("trainer_state")
        if not isinstance(state, Mapping):
            raise ValueError("resume checkpoint lacks trainer state")
        buffers = state["buffers"]
        if set(buffers.buffers) != {task.task_id for task in tasks}:
            raise ValueError("resume checkpoint replay tasks do not match the requested meta-train tasks")
        steps = int(state["steps"])
        gradient_updates = int(state.get("gradient_updates", len(state.get("progress", ()))))
        episode_counter = int(state["episode_counter"])
        best_score = state.get("best_score")
        next_validation = int(state["next_validation"])
        next_context_refresh = int(state["next_context_refresh"])
        rng.bit_generator.state = checkpoint["rng_state"]["numpy_generator"]
        cpu_rng = torch.as_tensor(checkpoint["rng_state"]["torch"], dtype=torch.uint8, device="cpu").clone()
        torch.set_rng_state(cpu_rng)
        if torch.cuda.is_available() and checkpoint["rng_state"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(
                [
                    torch.as_tensor(state, dtype=torch.uint8, device="cpu").clone()
                    for state in checkpoint["rng_state"]["cuda"]
                ]
            )
    checkpoint_interval = int(checkpoint_interval_steps) if checkpoint_interval_steps else None
    next_checkpoint = ((steps // checkpoint_interval) + 1) * checkpoint_interval if checkpoint_interval else None

    def rng_state() -> dict[str, Any]:
        return {
            "numpy_generator": rng.bit_generator.state,
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def trainer_state() -> dict[str, Any]:
        return {
            "buffers": buffers,
            "steps": steps,
            "gradient_updates": gradient_updates,
            "episode_counter": episode_counter,
            "best_score": best_score,
            "next_validation": next_validation,
            "next_context_refresh": next_context_refresh,
        }

    def save(path: Path) -> None:
        save_checkpoint(
            path,
            agent,
            config,
            taskbook_hash,
            steps,
            casebook_hashes=casebook_hashes,
            training_seed=seed,
            rng_state=rng_state(),
            trainer_state=trainer_state(),
        )

    def log_route(task: Any, phase: str, route: Any, **extra: Any) -> None:
        if route is None:
            return
        row = {
            "schema": "posterior_router_audit_v1",
            "training_seed": int(seed),
            "task_ref": content_hash({"task_id": task.task_id}),
            "phase": phase,
            "environment_steps": int(steps),
            "posthoc_only": False,
            "parameter_hash": agent.parameter_hash(),
            **route.audit_dict(),
            **extra,
        }
        with router_audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def collect(task: Any, case: dict[str, Any], z: torch.Tensor, mode: str,
                posterior_version: int, route: Any = None) -> Any:
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
            return collect_episode(
                env,
                task,
                case,
                agent,
                z,
                mode,
                device,
                episode_id=f"{task.task_id}:{episode_counter:08d}",
                posterior_version=posterior_version,
                route_context=route,
            )
        finally:
            env.close()

    def collect_prior_posterior_pair(task: Any) -> int:
        """Clear per-adaptation context, then collect one prior and posterior rollout."""
        book = casebooks[task.task_id]["train_pool"]
        prior_case = book[int(rng.integers(len(book)))]
        prior_mu, prior_log_var = agent.prior(tasks=[scenario_tasks[task.task_id]])
        prior_route = agent.compute_route(
            task_descriptors[task.task_id], prior_mu, prior_log_var, 0
        ) if task_descriptors else None
        prior = collect(
            task,
            prior_case,
            prior_mu if agent.no_context_training else agent.sample_latent(prior_mu, prior_log_var),
            "prior_support",
            0,
            prior_route,
        )
        log_route(task, "collection", prior_route, collection_mode="prior_support")
        buffers.add_episode(task.task_id, prior.transitions)
        if agent.no_context_training:
            posterior_mu, posterior_log_var = prior_mu, prior_log_var
        else:
            posterior_mu, posterior_log_var = agent.infer_posterior([[prior.transitions]], [scenario_tasks[task.task_id]])
        posterior_route = agent.compute_route(
            task_descriptors[task.task_id], posterior_mu, posterior_log_var, 1
        ) if task_descriptors else None
        posterior_case = book[int(rng.integers(len(book)))]
        posterior = collect(
            task,
            posterior_case,
            posterior_mu if agent.no_context_training else agent.sample_latent(posterior_mu, posterior_log_var),
            "posterior_rollout",
            1,
            posterior_route,
        )
        log_route(task, "collection", posterior_route, collection_mode="posterior_rollout")
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
    configured_context_size = int(config["pearl"]["context_batch_size"])
    context_size = min(64, configured_context_size) if smoke else configured_context_size
    transitions_per_episode = int(config["pearl"]["context_transitions_per_episode"])
    max_context_episodes = max(1, context_size // transitions_per_episode)
    min_context_episodes = min(
        max_context_episodes,
        max(1, int(config["pearl"].get("context_min_episodes", 1))),
    )
    updates_per_iteration = 1 if smoke else int(config["meta_training"]["gradient_updates_per_iteration"])
    while steps < max_env_steps:
        if gradient_updates >= next_context_refresh:
            buffers.clear_recent_context()
            while gradient_updates >= next_context_refresh:
                next_context_refresh += context_refresh_interval
        sampled = _sample_tasks_without_replacement(
            tasks,
            int(config["meta_training"]["meta_batch_size"]),
            rng,
        )
        for task in sampled:
            steps += collect_prior_posterior_pair(task)
            if steps >= max_env_steps:
                break
        ready = [
            task
            for task in tasks
            if len(buffers.buffers[task.task_id]) >= max(batch_size, 2)
            and len(buffers.buffers[task.task_id].episodes) >= 2
            and len(buffers.recent_context_buffers[task.task_id].episodes) >= 1
        ]
        for _ in range(updates_per_iteration if ready else 0):
            selected = _sample_tasks_without_replacement(
                ready,
                int(config["meta_training"]["meta_batch_size"]),
                rng,
            )
            # Few-shot evaluation starts at one support episode and grows to
            # the configured context capacity.  Training only at the maximum
            # capacity makes posterior inference brittle for K=1/2, so expose
            # the encoder to the entire evaluation support-count range.
            context_episodes = _training_context_episode_count(
                buffers,
                [task.task_id for task in selected],
                min_context_episodes,
                max_context_episodes,
                rng,
            )
            context = buffers.context_per_task(
                [task.task_id for task in selected],
                context_episodes * transitions_per_episode,
                transitions_per_episode,
                rng,
            )
            rl = buffers.sample_per_task_excluding_context(
                [task.task_id for task in selected],
                context,
                batch_size,
                rng,
            )
            task_targets = None if semantic_targets is None else [
                semantic_targets[task.task_id] for task in selected
            ]
            metrics = agent.update(
                context,
                rl,
                task_targets,
                None if not task_descriptors else [task_descriptors[task.task_id] for task in selected],
                [context_episodes] * len(selected),
                [scenario_tasks[task.task_id] for task in selected],
            )
            with training_update_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "schema": "pearl_training_update_v1",
                    "training_seed": int(seed),
                    "environment_steps": int(steps),
                    "gradient_update": int(gradient_updates + 1),
                    **metrics,
                }, ensure_ascii=False, sort_keys=True) + "\n")
            for task, audit in zip(selected, agent.last_router_audits):
                row = {
                    "schema": "posterior_router_audit_v1",
                    "training_seed": int(seed),
                    "task_ref": content_hash({"task_id": task.task_id}),
                    "phase": "actor_update",
                    "environment_steps": int(steps),
                    "gradient_update": int(gradient_updates + 1),
                    "posthoc_only": False,
                    **audit,
                }
                with router_audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            gradient_updates += 1
        if validation_tasks and steps >= next_validation:
            validation = evaluate_fewshot(
                agent,
                config,
                validation_tasks,
                casebooks,
                "meta_validation",
                query_execution_mode=str(
                    config["evaluation"]["selection_query_execution_mode"]
                ),
            )
            score = validation_score(
                validation,
                shot=int(config["evaluation"].get("selection_shot", 5)),
            )
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
        "gradient_updates": gradient_updates,
        "best_validation_score": best_score,
        "selected_checkpoint": "best_model.pt",
    })
    return root
