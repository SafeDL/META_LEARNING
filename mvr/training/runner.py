"""Hierarchical collection: one simulator rollout updates every consumer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from ..failure.analyzer import analyze_rollout
from ..failure.criteria import DEFAULT_FAILURE_CRITERIA, FailureCriteria
from ..failure.inner_reward import InnerRiskReward
from ..failure.signature import FailureSignature
from ..context.trajectory_features import TrajectoryFeatureExtractor
from ..safety import TrafficActionShield
from ..scenario.applied import ExecutableEpisode
from ..state import PhysicalStateExtractor


@dataclass
class Rollout:
    transitions: list[dict[str, Any]]
    outcome: Mapping[str, Any]
    signature: FailureSignature
    trajectory: Any


class HierarchicalRunner:
    def __init__(self, max_steps: int = 240, criteria: FailureCriteria = DEFAULT_FAILURE_CRITERIA) -> None:
        self.max_steps = int(max_steps)
        self.criteria = criteria

    @staticmethod
    def _sut_arrived_destination(episode: ExecutableEpisode) -> bool:
        """Stop once the tested vehicle has completed its prescribed route."""
        final_lane = episode.sut.navigation.final_lane
        longitudinal, lateral = final_lane.local_coordinates(episode.sut.position)
        lane_width = float(episode.sut.navigation.get_current_lane_width())
        lane_count = int(episode.sut.navigation.get_current_lane_num())
        return bool(
            float(final_lane.length) - 5.0 < float(longitudinal) < float(final_lane.length) + 5.0
            and (0.5 - lane_count) * lane_width <= float(lateral) <= 0.5 * lane_width
        )

    def rollout(
        self,
        episode: ExecutableEpisode,
        scenario_family: str,
        option: str,
        inner_action: Callable[[np.ndarray], np.ndarray],
        *,
        trajectory_extractor: TrajectoryFeatureExtractor | None = None,
        reward_fn: InnerRiskReward | None = None,
        step_callback: Callable[[ExecutableEpisode, int, Mapping[str, Any]], None] | None = None,
    ) -> Rollout:
        transitions: list[dict[str, Any]] = []
        env = episode.env
        extractor = trajectory_extractor or TrajectoryFeatureExtractor()
        reward_fn = reward_fn or InnerRiskReward(self.criteria)
        state_extractor = PhysicalStateExtractor()
        shield = TrafficActionShield(episode)
        extractor.reset(env, episode.layout, episode.adversary_route, episode.sut_route)
        state_extractor.reset(env, episode.layout, episode.adversary_route, episode.sut_route)
        for _ in range(self.max_steps):
            state = state_extractor(episode.adversary, episode.sut)
            raw_action = np.asarray(inner_action(state), dtype=np.float32)
            shielded = shield.project(raw_action)
            _, env_reward, terminated, truncated, info = env.step(shielded.action)
            info = {**dict(info), **shield.observe(shielded, info)}
            sut_arrived = self._sut_arrived_destination(episode)
            info["sut_arrived_destination"] = sut_arrived
            info["adversary_out_of_road"] = bool(
                info.get("adversary_out_of_road", info.get("out_of_road", False))
            )
            if step_callback is not None:
                step_callback(episode, len(transitions), info)
            sut_observation = episode.sut_adapter.observe(env, episode.sut)
            sut_evidence = episode.sut_adapter.step(sut_observation)
            trajectory_row = extractor.step(env, episode.adversary, episode.sut, info)
            transitions.append({
                "state": state,
                # Replay is defined in the SAC actor's normalized action space.
                "raw_action": shielded.raw_action,
                "action": shielded.raw_action,
                "executed_action": shielded.action,
                "reward_inner": reward_fn(
                    trajectory_row, info, option, len(transitions), self.max_steps
                ),
                "reward_env": float(env_reward),
                "next_state": state_extractor(episode.adversary, episode.sut),
                "done": bool(terminated or truncated or sut_arrived),
                "info": info,
                "sut_observation": sut_observation,
                "sut_evidence": sut_evidence,
                "trajectory_features": trajectory_row,
            })
            if terminated or truncated or sut_arrived:
                break
        applied = episode.applied_scenario
        outcome, signature = analyze_rollout(
            transitions,
            scenario_family,
            applied.conflict_zone_id,
            applied.selected_candidate,
            self.criteria,
        )
        return Rollout(transitions, outcome, signature, extractor.finalize())
