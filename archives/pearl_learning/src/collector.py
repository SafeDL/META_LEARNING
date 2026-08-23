"""Episode collector with immutable task/case/posterior provenance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

from .replay import CollectionMode, Transition
from .moe import RouteContext


@dataclass
class Rollout:
    transitions: list[Transition]
    record: dict[str, object]
    mode: CollectionMode
    episode_id: str


def collect_episode(env: Any, task: Any, case: dict[str, object], agent: Any, z: torch.Tensor, mode: CollectionMode,
                    device: torch.device, *, episode_id: str, posterior_version: int,
                    route_context: RouteContext | None = None) -> Rollout:
    if route_context is not None and route_context.posterior_version != posterior_version:
        raise ValueError("route context and rollout use different posterior versions")
    observation, _ = env.reset(options={"case": case})
    rows: list[Transition] = []
    termination_reason = "running"
    while True:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.act(
                tensor,
                z,
                deterministic=mode in {"deterministic_query", "posterior_sampled_query"},
                route=route_context,
            ).squeeze(0).cpu().numpy()
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
    record = env.episode_record()
    record.update({
        "collection_mode": mode,
        "posterior_version": int(posterior_version),
        "router_schema": None if route_context is None else "posterior_router_v1",
        "route_hash": None if route_context is None else route_context.route_hash,
        "route_query_free": True if route_context is None else route_context.query_free,
    })
    return Rollout(rows, record, mode, episode_id)
