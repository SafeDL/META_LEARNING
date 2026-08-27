"""Hierarchical collection: one simulator rollout updates every consumer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from ..control import NativeAdversaryBaseController
from ..failure.analyzer import analyze_rollout
from ..failure.criteria import DEFAULT_FAILURE_CRITERIA, FailureCriteria
from ..failure.inner_reward import InnerRiskReward
from ..failure.signature import FailureSignature
from ..context.trajectory_features import TrajectoryFeatureExtractor
from ..safety import TrafficActionShield
from ..scenario.applied import ExecutableEpisode
from ..scenario.semantics import ScenarioActionAdapter, ScenarioSemanticMonitor
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
        schedule = ScenarioActionAdapter(episode, scenario_family)
        controller = NativeAdversaryBaseController(episode, scenario_family, schedule)
        shield = TrafficActionShield(episode, schedule)
        monitor = ScenarioSemanticMonitor(episode, scenario_family, schedule)
        extractor.reset(env, episode.layout, episode.adversary_route, episode.sut_route)
        state_extractor.reset(env, episode.layout, episode.adversary_route, episode.sut_route)
        try:
            for step in range(self.max_steps):
                state = state_extractor(episode.adversary, episode.sut, schedule.state)
                raw_action = np.asarray(inner_action(state), dtype=np.float32).reshape(-1)
                if raw_action.shape != (3,) or not np.isfinite(raw_action).all():
                    raise ValueError("Inner SAC must emit one finite 3-D interaction residual")
                raw_action = np.clip(raw_action, -1.0, 1.0)
                schedule.update(float(raw_action[1]), step)
                effective_action = np.asarray(
                    (raw_action[0], schedule.state.timing_reference, raw_action[2]),
                    dtype=np.float32,
                )
                base_action, candidate_action = controller.action(effective_action)
                shielded = shield.project(base_action, candidate_action)
                _, env_reward, terminated, truncated, info = env.step(shielded.action)
                info = {**dict(info), **shield.observe(shielded, info)}
                info["adversary_out_of_road"] = bool(
                    info.get("adversary_out_of_road", info.get("out_of_road", False))
                )
                monitor.update()
                trajectory_row = extractor.step(env, episode.adversary, episode.sut, info)
                target_collision = bool(
                    info.get("target_collision", info.get("crash_vehicle", False))
                )
                distance = float(trajectory_row[10]) * 100.0
                ttc = float(trajectory_row[8]) * 15.0
                near_miss = (
                    not target_collision
                    and 0.0 < distance < self.criteria.distance_m
                    and ttc < self.criteria.ttc_s
                )
                monitor.capture_event(
                    "collision" if target_collision else "near_miss" if near_miss else None,
                    info,
                )
                info = {**info, **monitor.info()}
                info["valid_target_collision"] = bool(
                    target_collision
                    and info["event_kind"] == "collision"
                    and info["event_semantic_valid"]
                    and info["event_traffic_valid"]
                )
                info["valid_critical_near_miss"] = bool(
                    near_miss
                    and info["event_kind"] == "near_miss"
                    and info["event_semantic_valid"]
                    and info["event_traffic_valid"]
                )
                sut_arrived = self._sut_arrived_destination(episode)
                info["sut_arrived_destination"] = sut_arrived
                valid_event = bool(
                    info["event_kind"] is not None
                    and info["event_semantic_valid"]
                    and info["event_traffic_valid"]
                )
                hard_violation = bool(info["adversary_traffic_violation"])
                if step_callback is not None:
                    step_callback(episode, len(transitions), info)
                sut_observation = episode.sut_adapter.observe(env, episode.sut)
                sut_evidence = episode.sut_adapter.step(sut_observation)
                done = bool(
                    terminated or truncated or sut_arrived or valid_event or hard_violation
                )
                transitions.append({
                    "state": state,
                    "raw_action": raw_action,
                    # Replay stores the command that actually changed the
                    # schedule, rather than a latched raw timing sample.
                    "action": effective_action,
                    "executed_action": shielded.action,
                    "base_action": shielded.base_action,
                    "candidate_action": shielded.candidate_action,
                    "maneuver_update_mask": schedule.state.maneuver_update_mask,
                    "reward_inner": reward_fn(
                        trajectory_row, info, option, len(transitions), self.max_steps
                    ),
                    "reward_env": float(env_reward),
                    "next_state": state_extractor(episode.adversary, episode.sut, schedule.state),
                    "done": done,
                    "info": info,
                    "sut_observation": sut_observation,
                    "sut_evidence": sut_evidence,
                    "trajectory_features": trajectory_row,
                })
                if done:
                    break
        finally:
            controller.destroy()
        applied = episode.applied_scenario
        outcome, signature = analyze_rollout(
            transitions,
            scenario_family,
            applied.conflict_zone_id,
            applied.selected_candidate,
            self.criteria,
        )
        return Rollout(transitions, outcome, signature, extractor.finalize())
