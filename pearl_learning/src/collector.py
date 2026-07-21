"""Episode collector with immutable task/case/posterior provenance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

from .replay import CollectionMode, Transition


@dataclass
class Rollout:
    transitions: list[Transition]
    record: dict[str, object]
    mode: CollectionMode
    episode_id: str


def collect_episode(env: Any, task: Any, case: dict[str, object], agent: Any, z: torch.Tensor, mode: CollectionMode,
                    device: torch.device, *, episode_id: str, posterior_version: int) -> Rollout:
    observation, _ = env.reset(options={"case": case})
    rows: list[Transition] = []
    termination_reason = "running"
    while True:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.act(tensor, z, deterministic=mode == "deterministic_query").squeeze(0).cpu().numpy()
        next_observation, reward, terminated, truncated, info = env.step(action)
        termination_reason = str(info["termination_reason"])
        rows.append(Transition(
            observation.copy(), np.asarray(action, dtype=np.float32).copy(), float(reward), next_observation.copy(),
            bool(terminated), bool(truncated), termination_reason, task.task_id, episode_id, str(case["case_id"]), mode,
            int(posterior_version),
        ))
        observation = next_observation
        if terminated or truncated:
            break
    return Rollout(rows, env.episode_record(), mode, episode_id)
