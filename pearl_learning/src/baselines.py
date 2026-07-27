"""Comparable SAC baseline registry and a shared frozen-task environment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import gymnasium as gym

from .io import content_hash, write_json
from .task_env import LogicalMergeEnv


BASELINE_NAMES = (
    "per_task_sac", "cross_task_policy_matrix", "topology_conditioned_pooled_sac", "scratch_sac",
    "pooled_finetune_sac", "oracle_task_conditioned_sac", "pearl_no_context",
)

# ``pearl_no_context`` evaluates a trained PEARL actor and therefore cannot be
# available before a formal PEARL run.  Keeping the two phases explicit avoids
# a circular formal-training gate.
PRETRAIN_BASELINE_NAMES = tuple(name for name in BASELINE_NAMES if name != "pearl_no_context")


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    training_scope: str
    privileged_task_input: bool = False


SPECS = {
    "per_task_sac": BaselineSpec("per_task_sac", "one independent SAC policy per frozen meta-train task"),
    "cross_task_policy_matrix": BaselineSpec("cross_task_policy_matrix", "evaluate every per-task policy on every held-out task"),
    "topology_conditioned_pooled_sac": BaselineSpec("topology_conditioned_pooled_sac", "one SAC policy sampled across frozen train tasks"),
    "scratch_sac": BaselineSpec("scratch_sac", "fresh SAC with exactly the query task support environment-step budget"),
    "pooled_finetune_sac": BaselineSpec("pooled_finetune_sac", "pooled policy fine-tuned with exactly the scratch support budget"),
    "oracle_task_conditioned_sac": BaselineSpec("oracle_task_conditioned_sac", "pooled SAC with a privileged frozen geometry one-hot", True),
    "pearl_no_context": BaselineSpec("pearl_no_context", "same PEARL actor queried only with the unit-normal prior"),
}


class PooledLogicalMergeEnv(gym.Env):
    """One Gym facade that samples whole episodes from frozen task casebooks."""
    metadata = LogicalMergeEnv.metadata

    def __init__(self, tasks: list[Any], config: Mapping[str, Any], casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]], seed: int):
        super().__init__()
        if not tasks:
            raise ValueError("pooled baseline needs at least one task")
        self.tasks, self.config, self.casebooks, self.rng = list(tasks), dict(config), casebooks, np.random.default_rng(seed)
        self.current: LogicalMergeEnv | None = None
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (int(config["environment"]["observation_dim"]),), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (int(config["environment"]["action_dim"]),), np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.close()
        task = self.tasks[int(self.rng.integers(len(self.tasks)))]
        book = self.casebooks[task.task_id]["train_pool"]
        self.current = LogicalMergeEnv(task, self.config, book)
        return self.current.reset(options=options)

    def step(self, action):
        if self.current is None:
            raise RuntimeError("reset before step")
        return self.current.step(action)

    def close(self):
        if self.current is not None:
            self.current.close(); self.current = None


class OracleTaskObservation(gym.ObservationWrapper):
    """Explicitly privileged geometry one-hot used only by the oracle baseline."""
    def __init__(self, env: PooledLogicalMergeEnv, geometry_ids: list[str]):
        super().__init__(env)
        self.geometry_ids = list(geometry_ids)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (env.observation_space.shape[0] + len(self.geometry_ids),), np.float32)

    def observation(self, observation):
        current = self.env.current.task.geometry_id
        one_hot = np.zeros(len(self.geometry_ids), dtype=np.float32)
        one_hot[self.geometry_ids.index(current)] = 1.0
        return np.concatenate([np.asarray(observation, dtype=np.float32), one_hot])


def write_baseline_manifest(output_root: str | Path, *, name: str, taskbook_hash: str, seed: int, env_steps: int,
                            smoke: bool, artifacts: Mapping[str, str] | None = None,
                            config_hash: str | None = None, casebook_hashes: Mapping[str, str] | None = None,
                            checkpoint_hash: str | None = None) -> Path:
    if name not in SPECS:
        raise ValueError(f"unknown baseline {name}")
    root = Path(output_root) / name
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "logical_merge_baseline", "baseline": name, "description": SPECS[name].training_scope,
        "privileged_task_input": SPECS[name].privileged_task_input, "taskbook_hash": taskbook_hash,
        "seed": int(seed), "environment_steps": int(env_steps), "status": "smoke_completed" if smoke else "completed",
        "config_hash": config_hash, "casebook_hashes": dict(casebook_hashes or {}),
        "checkpoint_hash": checkpoint_hash, "artifacts": dict(artifacts or {}),
    }
    target = root / "baseline_manifest.json"; write_json(target, payload)
    return target
