"""Hierarchical collection: one simulator rollout updates every consumer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from ..failure.analyzer import analyze_rollout
from ..failure.signature import FailureSignature
from ..failure.inner_reward import InnerRiskReward
from ..context.trajectory_features import TrajectoryFeatureExtractor
from ..scenario.applied import ExecutableEpisode
from ..state import PhysicalStateExtractor


@dataclass
class Rollout:
    transitions: list[dict[str, Any]]
    outcome: Mapping[str, Any]
    signature: FailureSignature
    trajectory: Any


class HierarchicalRunner:
    def __init__(self, max_steps: int = 240) -> None:
        self.max_steps = int(max_steps)

    def rollout(self, episode: ExecutableEpisode, scenario_family: str, option: str, inner_action: Callable[[np.ndarray], np.ndarray], *, trajectory_extractor: TrajectoryFeatureExtractor | None = None, reward_fn: InnerRiskReward | None = None) -> Rollout:
        transitions: list[dict[str, Any]] = []
        env = episode.env
        extractor = trajectory_extractor or TrajectoryFeatureExtractor()
        reward_fn = reward_fn or InnerRiskReward()
        state_extractor = PhysicalStateExtractor()
        extractor.reset(env, episode.layout)
        state_extractor.reset(env, episode.layout)
        for _ in range(self.max_steps):
            state = state_extractor(episode.adversary, episode.sut)
            action = np.asarray(inner_action(state), dtype=np.float32)
            _, env_reward, terminated, truncated, info = env.step(action)
            sut_observation = episode.sut_adapter.observe(env, episode.sut)
            sut_evidence = episode.sut_adapter.step(sut_observation)
            trajectory_row = extractor.step(env, episode.adversary, episode.sut, info)
            transitions.append({"state": state, "action": action, "reward_inner": reward_fn(trajectory_row, info, option, len(transitions), self.max_steps), "reward_env": float(env_reward), "next_state": state_extractor(episode.adversary, episode.sut), "done": bool(terminated or truncated), "info": dict(info), "sut_observation": sut_observation, "sut_evidence": sut_evidence, "trajectory_features": trajectory_row})
            if terminated or truncated:
                break
        applied = episode.applied_scenario
        outcome, signature = analyze_rollout(
            transitions,
            scenario_family,
            applied.conflict_zone_id,
            applied.selected_candidate,
        )
        return Rollout(transitions, outcome, signature, extractor.finalize())
