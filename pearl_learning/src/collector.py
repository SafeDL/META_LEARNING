"""Episode collector; query rollouts are explicitly never returned as context."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
import torch

from .replay import Transition


@dataclass
class Rollout:
    transitions: list[Transition]; record: dict[str, object]; mode: str


def collect_episode(env: Any, task: Any, case: dict[str, object], agent: Any, z: torch.Tensor, mode: Literal["prior_support", "posterior_support", "deterministic_query"], device: torch.device) -> Rollout:
    observation, _ = env.reset(options={"case": case}); rows: list[Transition] = []
    while True:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.act(obs_tensor, z, deterministic=mode == "deterministic_query").squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rows.append(Transition(observation.copy(), np.asarray(action, dtype=np.float32).copy(), float(reward), next_obs.copy(), bool(terminated), bool(truncated), task.task_id))
        observation = next_obs
        if terminated or truncated: break
    return Rollout(rows, env.episode_record(), mode)
