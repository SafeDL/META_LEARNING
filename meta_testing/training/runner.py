"""Hierarchical collection: one simulator rollout updates every consumer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from ..failure.signature import FailureSignature
from ..failure.inner_reward import InnerRiskReward
from ..context.trajectory_features import TrajectoryFeatureExtractor
from ..scenario.applied import ExecutableEpisode
from ..state import CanonicalStateExtractor


@dataclass
class Rollout:
    scene_config: Mapping[str, Any]
    option: str
    transitions: list[dict[str, Any]]
    outcome: Mapping[str, Any]
    signature: FailureSignature
    trajectory: Any


class HierarchicalRunner:
    def __init__(self, max_steps: int = 240) -> None:
        self.max_steps = int(max_steps)

    def rollout(self, episode: ExecutableEpisode, scene_config: Mapping[str, Any], option: str, inner_action: Callable[[np.ndarray], np.ndarray], analyze: Callable[[list[dict[str, Any]]], tuple[Mapping[str, Any], FailureSignature]], *, state_extractor: CanonicalStateExtractor | None = None, trajectory_extractor: TrajectoryFeatureExtractor | None = None, reward_fn: InnerRiskReward | None = None) -> Rollout:
        transitions: list[dict[str, Any]] = []
        env, observation = episode.env, episode.initial_observation
        extractor = trajectory_extractor or TrajectoryFeatureExtractor()
        reward_fn = reward_fn or InnerRiskReward()
        state_extractor = state_extractor or CanonicalStateExtractor(5)
        extractor.reset(env, episode.layout)
        for _ in range(self.max_steps):
            state = state_extractor(observation)
            action = np.asarray(inner_action(state), dtype=np.float32)
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            sut_observation = episode.sut_adapter.observe(env, episode.sut)
            sut_evidence = episode.sut_adapter.step(sut_observation)
            trajectory_row = extractor.step(env, episode.adversary, episode.sut, info)
            transitions.append({"state": state, "action": action, "reward_inner": reward_fn(trajectory_row, info, option, len(transitions), self.max_steps), "reward_env": float(env_reward), "next_state": state_extractor(next_observation), "done": bool(terminated or truncated), "info": dict(info), "sut_observation": sut_observation, "sut_evidence": sut_evidence, "trajectory_features": trajectory_row})
            observation = next_observation
            if terminated or truncated:
                break
        outcome, signature = analyze(transitions)
        return Rollout(dict(scene_config), str(option), transitions, outcome, signature, extractor.finalize())
