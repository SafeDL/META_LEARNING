"""Hierarchical collection: one simulator rollout updates every consumer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from ..control import FrenetSACAdversaryController
from ..failure.analyzer import analyze_rollout
from ..failure.criteria import DEFAULT_FAILURE_CRITERIA, FailureCriteria
from ..failure.inner_reward import InnerRiskReward
from ..failure.signature import FailureSignature
from ..context.trajectory_features import TrajectoryFeatureExtractor
from ..safety import TrafficActionShield
from ..scenario.applied import ExecutableEpisode
from ..scenario.executor import ScenarioExecutor
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
        reward_fn.reset()
        state_extractor = PhysicalStateExtractor()
        schedule = ScenarioActionAdapter(episode, scenario_family)
        controller = FrenetSACAdversaryController(
            episode, scenario_family, schedule
        )
        shield = TrafficActionShield(episode, schedule)
        monitor = ScenarioSemanticMonitor(episode, scenario_family, schedule)
        max_steps = max(
            self.max_steps,
            int(episode.layout.traffic_contract.min_completion_steps),
        )
        held_policy_action: np.ndarray | None = None
        extractor.reset(env, episode.layout, episode.adversary_route, episode.sut_route)
        state_extractor.reset(env, episode.layout, episode.adversary_route, episode.sut_route)
        try:
            for step in range(max_steps):
                schedule.update()
                state = state_extractor(episode.adversary, episode.sut, schedule)
                reference_before = schedule.maneuver_reference()
                planner_active = bool(
                    schedule.state.maneuver_latched
                    and not schedule.state.maneuver_completed
                    and reference_before.start_remaining_m <= 0.0
                )
                if planner_active:
                    policy_decision = (
                        held_policy_action is None or schedule.planner.replan_due
                    )
                else:
                    policy_decision = (
                        held_policy_action is None
                        or step % schedule.planner.replan_interval_steps == 0
                    )
                if policy_decision:
                    held_policy_action = np.asarray(
                        inner_action(state), dtype=np.float32
                    ).reshape(-1)
                    if (
                        held_policy_action.shape != (4,)
                        or not np.isfinite(held_policy_action).all()
                    ):
                        raise ValueError(
                            "Inner SAC must emit one finite 4-D planner action"
                        )
                    held_policy_action = np.clip(
                        held_policy_action, -1.0, 1.0
                    )
                raw_action = held_policy_action.copy()
                planner_action, requested_action = controller.action(raw_action)
                shielded = shield.project(requested_action)
                _, env_reward, terminated, truncated, info = env.step(shielded.action)
                controller.observe_environment(info)
                info = {**dict(info), **shield.observe(shielded)}
                reference = schedule.maneuver_reference()
                info.update({
                    "maneuver_reference_progress": reference.progress,
                    "maneuver_reference_lateral_error_m": reference.lateral_error_m,
                    "maneuver_reference_heading_error_rad": reference.heading_error_rad,
                    "maneuver_reference_desired_lateral_m": reference.desired_lateral_m,
                    "maneuver_reference_length_m": reference.length_m,
                    "maneuver_reference_curvature_m_inv": reference.curvature_m_inv,
                    "maneuver_reference_speed_limit_mps": reference.speed_limit_mps,
                    "maneuver_start_remaining_m": reference.start_remaining_m,
                    "maneuver_active_lambda_length": reference.active_lambda_length,
                    "maneuver_active_beta_early": reference.active_beta_early,
                    "maneuver_active_beta_late": reference.active_beta_late,
                    "maneuver_reference_blend_progress": reference.blend_progress,
                    "maneuver_replan_due": reference.replan_due,
                    "maneuver_reference_start_s_m": schedule.contract.start_s_m,
                    "maneuver_reference_points_xy": schedule.planner.reference_points(),
                })
                sut_policy = env.engine.get_policy(episode.sut.id)
                sut_action = np.asarray(
                    getattr(sut_policy, "action_info", {}).get("action", (0.0, 0.0)),
                    dtype=float,
                )
                sut_projection = episode.sut_route.projection(
                    episode.sut.position, episode.sut.heading_theta
                )
                info.update(ScenarioExecutor.sut_lane_status(episode, require_routing_target=True))
                info.update({
                    "adversary_speed_mps": float(episode.adversary.speed_km_h) / 3.6,
                    "speed_limit_mps": float(
                        episode.layout.traffic_contract.speed_limit_mps
                    ),
                    "adversary_route_progress": float(
                        episode.adversary_route.projection(
                            episode.adversary.position, episode.adversary.heading_theta
                        ).s_m
                    ) / max(float(episode.adversary_route.length_m), 1e-6),
                    "sut_steering": float(sut_action[0]),
                    "sut_acceleration": float(sut_action[1]),
                    "sut_lateral_error_m": float(sut_projection.lateral_m),
                    "sut_heading_error_rad": float(sut_projection.heading_error),
                    "sut_speed_mps": float(episode.sut.speed_km_h) / 3.6,
                    "sut_route_progress_m": float(sut_projection.s_m),
                    "sut_target_speed_mps": float(getattr(sut_policy, "target_speed", 0.0)) / 3.6,
                    "sut_nominal_target_speed_mps": float(
                        getattr(sut_policy, "nominal_target_speed_mps", 0.0)
                    ),
                    "sut_curve_safe_speed_mps": float(
                        getattr(sut_policy, "curve_safe_speed_mps", 0.0)
                    ),
                })
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
                info["inner_raw_policy_action_l2"] = float(
                    np.square(raw_action).sum()
                )
                info["inner_planner_action_l2"] = float(
                    np.square(planner_action).sum()
                )
                info["inner_executed_vehicle_action_l2"] = float(
                    np.square(shielded.action).sum()
                )
                info["raw_policy_action"] = raw_action.tolist()
                info["planner_action"] = planner_action.tolist()
                info["inner_policy_decision"] = bool(policy_decision)
                info["requested_vehicle_action"] = (
                    shielded.requested_action.tolist()
                )
                info["executed_vehicle_action"] = shielded.action.tolist()
                info["valid_target_collision"] = bool(
                    target_collision
                    and info["event_kind"] == "collision"
                    and info["event_semantic_valid"]
                    and info["event_execution_valid"]
                )
                info["valid_critical_near_miss"] = bool(
                    near_miss
                    and info["event_kind"] == "near_miss"
                    and info["event_semantic_valid"]
                    and info["event_execution_valid"]
                )
                sut_arrived = self._sut_arrived_destination(episode)
                info["sut_arrived_destination"] = sut_arrived
                if target_collision:
                    termination_reason = "target_collision"
                elif sut_arrived:
                    termination_reason = "sut_route_completed"
                # MetaDrive's sole controllable agent is the adversary.  Its
                # native arrival sets ``terminated`` permanently, but does
                # not make the SUT route complete; continue this test until
                # the SUT completes or a real terminal event occurs.
                elif terminated and not bool(info.get("arrive_dest", False)):
                    termination_reason = "simulator_terminated"
                elif truncated:
                    termination_reason = "simulator_truncated"
                elif step + 1 >= max_steps:
                    termination_reason = "runner_step_budget"
                else:
                    termination_reason = None
                info["test_completion_condition"] = (
                    episode.layout.traffic_contract.completion_condition
                )
                info["termination_reason"] = termination_reason
                if step_callback is not None:
                    step_callback(episode, len(transitions), info)
                sut_observation = episode.sut_adapter.observe(env, episode.sut)
                sut_evidence = episode.sut_adapter.step(sut_observation)
                # A near-miss is evidence, not a route terminator: unless a
                # collision or hard violation occurs, the SUT must finish its
                # declared route for the scenario to be a complete test.
                done = termination_reason is not None
                transitions.append({
                    "state": state,
                    "raw_policy_action": raw_action,
                    "planner_action": planner_action,
                    "requested_vehicle_action": shielded.requested_action,
                    "executed_vehicle_action": shielded.action,
                    "reward_inner": reward_fn(trajectory_row, info),
                    "reward_env": float(env_reward),
                    "next_state": state_extractor(episode.adversary, episode.sut, schedule),
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
