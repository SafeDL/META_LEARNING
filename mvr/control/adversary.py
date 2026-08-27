"""Thin MetaDrive IDM wrapper for nominal adversary actions."""
from __future__ import annotations

from typing import Any

import numpy as np
from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy

from ..scenario.semantics import ScenarioActionAdapter


class NativeAdversaryBaseController:
    """Compute nominal actions without registering a second engine policy."""

    longitudinal_residual_scale = 0.25
    lateral_residual_scale = 0.15
    timing_speed_scale_kmh = 9.0

    _option_speed_bias_kmh = {
        "approach_conflict": 0.0,
        "yield_then_press": -3.0,
        "gap_close": 3.0,
    }

    def __init__(
        self,
        episode: Any,
        family: str,
        schedule: ScenarioActionAdapter,
        option: str = "approach_conflict",
    ) -> None:
        self.episode = episode
        self.family = str(family)
        self.schedule = schedule
        self.option = str(option)
        if self.option not in self._option_speed_bias_kmh:
            raise ValueError(f"unsupported adversarial option {option!r}")
        self.policy = IDMPolicy(episode.adversary, int(episode.episode_seed or 0))
        self.policy.enable_lane_change = False
        self.policy.NORMAL_SPEED = float(episode.layout.traffic_contract.sut_nominal_speed_mps) * 3.6
        self.policy.target_speed = self.policy.NORMAL_SPEED
        self._arrived_destination = False

    def observe_environment(self, info: Any) -> None:
        """Hold the completed adversary route while the SUT finishes its test."""
        self._arrived_destination = self._arrived_destination or bool(dict(info).get("arrive_dest", False))

    def _set_target_speed(self) -> None:
        speed_limit = float(self.episode.layout.traffic_contract.speed_limit_mps) * 3.6
        # Zero residual is ordinary, matched-flow traffic.  The timing and
        # longitudinal residuals then create bounded interaction pressure;
        # a fixed 75%-of-limit base would make the adversary leave short
        # scenario routes long before the SUT completed its test route.
        nominal = float(self.episode.layout.traffic_contract.sut_nominal_speed_mps) * 3.6
        option_bias = self._option_speed_bias_kmh[self.option]
        if self.option == "yield_then_press" and self.schedule.state.challenge_phase_active:
            option_bias = abs(option_bias)
        target = (
            nominal
            + option_bias
            + self.timing_speed_scale_kmh * self.schedule.state.timing_reference
        )
        self.policy.NORMAL_SPEED = float(np.clip(target, 1.0, speed_limit))
        self.policy.target_speed = self.policy.NORMAL_SPEED

    def _cutin_action(self) -> np.ndarray:
        lane = self.schedule.target_lane()
        self.policy.routing_target_lane = lane
        objects = self.episode.adversary.lidar.get_surrounding_objects(self.episode.adversary)
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

    def _route_follow_action(self) -> np.ndarray:
        """Use native road progression, but follow its current physical lane."""
        self.policy.move_to_next_road()
        lane = self.episode.adversary.navigation.current_lane
        objects = self.episode.adversary.lidar.get_surrounding_objects(self.episode.adversary)
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
        if residual.shape != (3,) or not np.isfinite(residual).all():
            raise ValueError("native adversary controller requires a finite 3-D residual")
        if self._arrived_destination:
            stopped = np.asarray((0.0, -1.0), dtype=np.float32)
            return stopped, stopped.copy()
        self._set_target_speed()
        base = self._cutin_action() if self.family == "cutin" else self._route_follow_action()
        base = np.clip(base, -1.0, 1.0)
        candidate = base + np.asarray(
            (
                self.lateral_residual_scale * float(residual[2]),
                self.longitudinal_residual_scale * float(residual[0]),
            ),
            dtype=np.float32,
        )
        return base, np.clip(candidate, -1.0, 1.0)

    def destroy(self) -> None:
        self.policy.destroy()
