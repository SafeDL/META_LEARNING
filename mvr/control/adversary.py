"""Frenet-planning SAC interface and deterministic vehicle tracking."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..safety.dynamics import VehicleActionProjector
from ..scenario.semantics import ScenarioActionAdapter


class FrenetSACAdversaryController:
    """Convert a four-dimensional planner action into physical controls."""

    stanley_cross_track_gain = 0.5
    heading_error_gain = 1.0
    stanley_soft_speed_mps = 2.0

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
    def observe_environment(self, info: Any) -> None:
        """MetaDrive arrival does not end the SUT-completion test."""

    def action(self, sac_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        planner_action = self.schedule.apply_planner_action(sac_action)
        reference = self.schedule.maneuver_reference()
        steering = 0.0
        active = bool(
            self.family != "cutin"
            or (
                self.schedule.state.maneuver_latched
                and reference.start_remaining_m <= 0.0
            )
        )
        vehicle = self.episode.adversary
        speed = float(vehicle.speed_km_h) / 3.6
        if active:
            wheelbase = max(0.6 * float(vehicle.LENGTH), 1.0)
            feedforward = float(np.arctan(wheelbase * reference.curvature_m_inv))
            cross_track = float(np.arctan2(
                self.stanley_cross_track_gain * reference.lateral_error_m,
                self.stanley_soft_speed_mps + speed,
            ))
            steering_angle = (
                feedforward
                - cross_track
                - self.heading_error_gain * reference.heading_error_rad
            )
            max_steering = np.deg2rad(float(vehicle.config["max_steering"]))
            steering = float(steering_angle / max(max_steering, 1e-6))
        longitudinal = float(planner_action[3])
        if speed > reference.speed_limit_mps:
            longitudinal = min(
                longitudinal,
                -min(1.0, (speed - reference.speed_limit_mps) / 2.0),
            )
        return planner_action, self.projector.project((steering, longitudinal))

    def destroy(self) -> None:
        """The deterministic tracker owns no native policy resources."""
