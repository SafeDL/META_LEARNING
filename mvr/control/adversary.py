"""Direct SAC action interface for the controllable adversary."""
from __future__ import annotations

from typing import Any

import numpy as np
from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy
from ..scenario.semantics import ScenarioActionAdapter


class DirectSACAdversaryController:
    """Expose SAC's two commands without an IDM nominal-action path.

    The Logical Scenario fixes when a Cut-in may begin. It does not generate
    steering or longitudinal commands: both values originate with the Inner
    SAC and are subsequently projected only by the physical traffic shield.
    """

    def __init__(
        self,
        episode: Any,
        family: str,
        schedule: ScenarioActionAdapter,
    ) -> None:
        self.episode = episode
        self.family = str(family)
        self.schedule = schedule
    def observe_environment(self, _info: Any) -> None:
        """Keep the controller interface symmetric with rollout consumers."""

    def action(self, sac_action: np.ndarray) -> np.ndarray:
        action = np.asarray(sac_action, dtype=np.float32).reshape(-1)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("direct SAC controller requires one finite 2-D action")
        return np.clip(action, -1.0, 1.0)

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
