"""Direct SAC action interface for the controllable adversary."""
from __future__ import annotations

from typing import Any

import numpy as np

from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy

from ..safety.dynamics import VehicleActionProjector
from ..scenario.semantics import ScenarioActionAdapter


class DirectSACAdversaryController:
    """Track Cut-in reference geometry with SAC longitudinal risk control."""

    def __init__(
        self,
        episode: Any,
        family: str,
        schedule: ScenarioActionAdapter,
    ) -> None:
        self.episode = episode
        self.family = str(family)
        self.schedule = schedule
        self.projector = VehicleActionProjector(
            episode.adversary,
            float(episode.layout.traffic_contract.speed_limit_mps),
            max_jerk_mps3=1.5,
        )
        self._arrived_destination = False

    def observe_environment(self, info: Any) -> None:
        """Keep the controller interface symmetric with rollout consumers."""
        self._arrived_destination = self._arrived_destination or bool(
            dict(info).get("arrive_dest", False)
        )

    def action(self, sac_action: np.ndarray) -> np.ndarray:
        action = np.asarray(sac_action, dtype=np.float32).reshape(-1)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("direct SAC controller requires one finite 2-D action")
        action = np.clip(action, -1.0, 1.0)
        if self._arrived_destination:
            return self.projector.project((0.0, -1.0))
        steering = 0.0
        longitudinal = float(action[1])
        reference = self.schedule.cutin_reference()
        if self.schedule.state.maneuver_latched:
            vehicle = self.episode.adversary
            max_steering = np.deg2rad(float(vehicle.config["max_steering"]))
            heading_term = reference.heading_error_rad / max(max_steering, 1e-6)
            # The feasibility probe establishes 0.25 * lateral error as the
            # stable reference feedback gain for this actuator.  The former
            # weaker term let a saturated learned residual leave the target
            # lane after q reached one.  Heading only damps that feedback;
            # it must not dominate the geometric lane-centering term.
            tracking = 0.25 * reference.lateral_error_m - 0.05 * heading_term
            # The planned quintic supplies the complete lane transition.
            # SAC may only make a small correction while the curve is still
            # developing; freeze it before the exit so a risk-seeking action
            # cannot turn a completed legal lane change into road departure.
            residual = 0.0 if reference.progress >= 0.85 else 0.005 * action[0]
            steering = float(tracking + residual)
        speed = float(self.episode.adversary.speed_km_h) / 3.6
        if speed > reference.speed_limit_mps:
            # The cap is geometry-derived, never a learned reward cue.  It
            # applies before temporal onset too: lane motion remains locked,
            # but braking must begin early enough to enter a short spatial
            # curve within its lateral-acceleration envelope.
            longitudinal = min(longitudinal, -min(
                1.0, (speed - reference.speed_limit_mps) / 2.0
            ))
        # The spatial reference supplies legal lateral geometry. SAC retains
        # a bounded lateral correction and directly selects full bounded
        # longitudinal acceleration or braking through this projector.
        return self.projector.project((steering, longitudinal))

    def destroy(self) -> None:
        """Direct actions allocate no native policy resources."""


class NativeAdversaryBaseController:
    """Legacy nominal controller retained for non-Cut-in families.

    The direct SAC contract is intentionally used only by Cut-in.  Merge and
    roundabout still use their established IDM base so this change does not
    alter their simulator contracts or historical tests.
    """

    acceleration_residual_scale = 0.25
    steering_residual_scale = 0.15

    def __init__(
        self,
        episode: Any,
        family: str,
        schedule: ScenarioActionAdapter,
    ) -> None:
        self.episode = episode
        self.family = str(family)
        self.schedule = schedule
        self.policy = IDMPolicy(episode.adversary, int(episode.episode_seed or 0))
        self.policy.enable_lane_change = False
        self.policy.NORMAL_SPEED = float(episode.layout.traffic_contract.sut_nominal_speed_mps) * 3.6
        self.policy.target_speed = self.policy.NORMAL_SPEED
        self._arrived_destination = False

    def observe_environment(self, info: Any) -> None:
        self._arrived_destination = self._arrived_destination or bool(
            dict(info).get("arrive_dest", False)
        )

    def _set_target_speed(self) -> None:
        speed_limit = float(self.episode.layout.traffic_contract.speed_limit_mps) * 3.6
        nominal = float(self.episode.layout.traffic_contract.sut_nominal_speed_mps) * 3.6
        self.policy.NORMAL_SPEED = float(np.clip(nominal, 1.0, speed_limit))
        self.policy.target_speed = self.policy.NORMAL_SPEED

    def _route_follow_action(self) -> np.ndarray:
        self.policy.move_to_next_road()
        lane = self.episode.adversary.navigation.current_lane
        objects = self.episode.adversary.lidar.get_surrounding_objects(
            self.episode.adversary
        )
        surrounding = FrontBackObjects.get_find_front_back_objs(
            objects, lane, self.episode.adversary.position, self.policy.MAX_LONG_DIST
        )
        action = np.asarray(
            (
                self.policy.steering_control(lane),
                self.policy.acceleration(
                    surrounding.front_object(), surrounding.front_min_distance()
                ),
            ),
            dtype=np.float32,
        )
        self.policy.action_info["action"] = action.tolist()
        return action

    def action(self, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residual = np.asarray(residual, dtype=np.float32).reshape(-1)
        if residual.shape != (2,) or not np.isfinite(residual).all():
            raise ValueError("native adversary controller requires a finite 2-D residual")
        if self._arrived_destination:
            stopped = np.asarray((0.0, -1.0), dtype=np.float32)
            return stopped, stopped.copy()
        self._set_target_speed()
        base = np.clip(self._route_follow_action(), -1.0, 1.0)
        candidate = base + np.asarray(
            (
                self.steering_residual_scale * float(residual[0]),
                self.acceleration_residual_scale * float(residual[1]),
            ),
            dtype=np.float32,
        )
        return base, np.clip(candidate, -1.0, 1.0)

    def destroy(self) -> None:
        self.policy.destroy()
