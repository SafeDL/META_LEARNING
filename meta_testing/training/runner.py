"""Hierarchical collection: one simulator rollout updates every consumer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from ..failure.signature import FailureSignature


@dataclass
class Rollout:
    scene_config: Mapping[str, Any]
    option: str
    transitions: list[dict[str, Any]]
    outcome: Mapping[str, Any]
    signature: FailureSignature


class HierarchicalRunner:
    def __init__(self, max_steps: int = 240) -> None:
        self.max_steps = int(max_steps)

    def rollout(self, env: Any, scene_config: Mapping[str, Any], option: str, inner_action: Callable[[Any], np.ndarray], analyze: Callable[[list[dict[str, Any]]], tuple[Mapping[str, Any], FailureSignature]]) -> Rollout:
        transitions: list[dict[str, Any]] = []
        observation = getattr(env, "_meta_testing_observation", None)
        if observation is None:
            raise RuntimeError("runner requires executor reset observation on the environment")
        for _ in range(self.max_steps):
            action = np.asarray(inner_action(observation), dtype=np.float32)
            next_observation, reward, terminated, truncated, info = env.step(action)
            transitions.append({"state": observation, "action": action, "reward_inner": float(reward), "next_state": next_observation, "done": bool(terminated or truncated), "info": dict(info)})
            observation = next_observation
            if terminated or truncated:
                break
        outcome, signature = analyze(transitions)
        return Rollout(dict(scene_config), str(option), transitions, outcome, signature)
