"""Zero-environment-step signal audit for the Gate 3 Critic replay path.

This diagnostic is authorized only after Stage A passes and Stage B_Q fails.
It loads the resumable trainer state from a frozen checkpoint, reproduces the
training context/RL-batch sampler with an audit-local deterministic RNG, and
reports how much terminal and conflict-near evidence reaches the Critic.  It
does not construct environments, update parameters, or write the checkpoint.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from pearl_learning.src.causal_audit import _conflict_near_indexes
from pearl_learning.src.io import content_hash, write_json
from pearl_learning.src.pearl_trainer import _training_context_episode_count
from pearl_learning.src.replay import Transition


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    if not len(data):
        return {"count": 0, "mean": 0.0, "std": 0.0, "p01": 0.0, "p50": 0.0, "p99": 0.0}
    return {
        "count": int(len(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "p01": float(np.quantile(data, 0.01)),
        "p50": float(np.quantile(data, 0.50)),
        "p99": float(np.quantile(data, 0.99)),
    }


def conflict_near_transition_ids(episodes: Iterable[Any], per_episode: int = 3) -> set[int]:
    """Return label-free conflict-near rows using the established Stage-A score."""
    selected: set[int] = set()
    for episode in episodes:
        rows = list(episode.transitions)
        for index in _conflict_near_indexes(rows[:-1], min(int(per_episode), max(0, len(rows) - 1))):
            selected.add(id(rows[index]))
    return selected


def transition_signal_summary(rows: Iterable[Transition], conflict_near_ids: set[int]) -> dict[str, Any]:
    """Summarize Critic-visible terminal/conflict signal without task labels.

    ``task_sensitive_proxy`` is the union of terminal rows and the existing
    public-dynamics conflict-near selection.  It is deliberately called a
    proxy: a single replay transition has no counterfactual task label, so this
    audit must not claim direct task identifiability from the rate alone.
    """
    data = list(rows)
    terminal = [row for row in data if row.terminated or row.truncated]
    near = [row for row in data if id(row) in conflict_near_ids]
    proxy = [row for row in data if row.terminated or row.truncated or id(row) in conflict_near_ids]
    terminal_ids = {id(row) for row in terminal}
    common = [row for row in data if id(row) not in terminal_ids and id(row) not in conflict_near_ids]

    def stratum(values: list[Transition]) -> dict[str, Any]:
        return {
            "transition_count": int(len(values)),
            "reward": _quantiles(float(row.reward) for row in values),
            "abs_arrival_time_difference": _quantiles(abs(float(row.obs[16])) for row in values),
            "min_abs_distance_to_conflict": _quantiles(
                min(abs(float(row.obs[0])), abs(float(row.obs[8]))) for row in values
            ),
            "abs_ttc": _quantiles(abs(float(row.obs[20])) for row in values),
        }

    total = max(1, len(data))
    return {
        "transition_count": int(len(data)),
        "terminal_transition_rate": float(len(terminal) / total),
        "conflict_near_transition_rate": float(len(near) / total),
        "task_sensitive_proxy_rate": float(len(proxy) / total),
        "task_sensitive_proxy_definition": "terminal OR one of up to three public-dynamics conflict-near non-terminal rows per replay episode",
        "termination_reason_counts": dict(sorted(Counter(row.termination_reason for row in terminal).items())),
        "collection_mode_counts": dict(sorted(Counter(row.collection_mode for row in data).items())),
        "signal_strata": {
            "terminal": stratum(terminal),
            "conflict_near": stratum(near),
            "terminal_or_conflict_near": stratum(proxy),
            "common": stratum(common),
        },
    }


def _training_sampler_spec(config: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    batch_size = int(config["sac"]["batch_size"])
    context_size = int(config["pearl"]["context_batch_size"])
    transitions_per_episode = int(config["pearl"]["context_transitions_per_episode"])
    max_context_episodes = max(1, context_size // transitions_per_episode)
    min_context_episodes = min(
        max_context_episodes,
        max(1, int(config["pearl"].get("context_min_episodes", 1))),
    )
    scheme = str(config["pearl"].get("context_transition_sampling", "random"))
    return batch_size, transitions_per_episode, min_context_episodes, max_context_episodes, scheme


def audit(checkpoint_path: Path, output_path: Path, sampled_batches: int) -> dict[str, Any]:
    if sampled_batches < 1:
        raise ValueError("--sampled-batches must be positive")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer_state = payload.get("trainer_state")
    if not isinstance(trainer_state, Mapping) or "buffers" not in trainer_state:
        raise ValueError("checkpoint lacks resumable trainer_state.buffers required for replay audit")
    resolved_path = checkpoint_path.parent / "config_resolved.json"
    if not resolved_path.exists():
        raise ValueError("checkpoint directory lacks config_resolved.json")
    config = json.loads(resolved_path.read_text(encoding="utf-8"))
    if payload.get("config_hash") != content_hash(config):
        raise ValueError("checkpoint config hash does not match saved resolved configuration")
    buffers = trainer_state["buffers"]
    task_ids = sorted(buffers.buffers)
    if not task_ids:
        raise ValueError("checkpoint replay buffers are empty")
    batch_size, transitions_per_episode, min_context, max_context, scheme = _training_sampler_spec(config)
    near_ids = {
        task_id: conflict_near_transition_ids(buffers.buffers[task_id].episodes)
        for task_id in task_ids
    }
    full_replay = {
        task_id: transition_signal_summary(
            (row for episode in buffers.buffers[task_id].episodes for row in episode.transitions),
            near_ids[task_id],
        )
        for task_id in task_ids
    }

    checkpoint_manifest = json.loads(checkpoint_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    audit_seed = int(content_hash({"checkpoint_hash": checkpoint_manifest["checkpoint_hash"], "audit": "gate3_critic_replay_signal_v1"})[:16], 16)
    rng = np.random.default_rng(audit_seed)
    sampled_rows = {task_id: [] for task_id in task_ids}
    context_episode_counts: list[int] = []
    for _ in range(int(sampled_batches)):
        context_episodes = _training_context_episode_count(
            buffers, task_ids, min_context, max_context, rng,
        )
        context = buffers.context_per_task(
            task_ids,
            context_episodes * transitions_per_episode,
            transitions_per_episode,
            rng,
            scheme=scheme,
        )
        rl = buffers.sample_per_task_excluding_context(task_ids, context, batch_size, rng)
        context_episode_counts.append(context_episodes)
        for task_id, rows in zip(task_ids, rl):
            sampled_rows[task_id].extend(rows)
    sampled_rl = {
        task_id: transition_signal_summary(sampled_rows[task_id], near_ids[task_id])
        for task_id in task_ids
    }
    result = {
        "schema": "gate3_critic_replay_signal_audit_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_hash": checkpoint_manifest["checkpoint_hash"],
        "checkpoint_training_seed": payload["training_seed"],
        "config_hash": payload["config_hash"],
        "not_a_training_or_environment_run": True,
        "environment_steps": 0,
        "training_updates": 0,
        "audit_rng_seed": audit_seed,
        "sampler": {
            "sampled_batches": int(sampled_batches),
            "rl_batch_size_per_task": batch_size,
            "context_transitions_per_episode": transitions_per_episode,
            "context_episode_count": _quantiles(context_episode_counts),
            "context_transition_sampling": scheme,
            "rl_batch_excludes_context_episode_ids": True,
        },
        "tasks": {
            task_id: {
                "full_replay": full_replay[task_id],
                "sampled_rl_batches": sampled_rl[task_id],
            }
            for task_id in task_ids
        },
    }
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sampled-batches", type=int, default=128)
    args = parser.parse_args()
    result = audit(Path(args.checkpoint), Path(args.output), args.sampled_batches)
    print(
        "Gate 3 Critic replay signal audit: "
        f"{result['sampler']['sampled_batches']} batches -> {Path(args.output)}"
    )


if __name__ == "__main__":
    main()
