"""CutIn environment with an externally controlled ego vehicle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle

from highway_env_benchmark.envs.cutin_env import CutInEnv, CutInScenario
from sut_algorithms.highway_env.idm_profiles import SUTProfile


@dataclass(frozen=True)
class ExternalEpisode:
    ego_collision: bool
    background_collision: bool
    near_miss: bool
    completed: bool
    min_ttc: float
    min_distance: float
    distance: float
    mean_speed: float
    steps: int


class ExternalCutInEnv(CutInEnv):
    """Run the shared scenario with a five-action external ego contract."""

    def __init__(
        self,
        scenario: CutInScenario,
        ego_kind: str = "mdp",
        render_mode: str | None = None,
    ) -> None:
        self.ego_kind = ego_kind
        self._distance = 0.0
        self._speed_samples: list[float] = []
        placeholder_profile = SUTProfile("external", "IDM")
        super().__init__(placeholder_profile, scenario, render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": True,
                },
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": ["x", "y", "vx", "vy", "sin_h", "cos_h"],
                    "features_range": {
                        "x": [-50, 50],
                        "y": [-50, 50],
                        "vx": [-40, 40],
                        "vy": [-40, 40],
                    },
                    "absolute": False,
                    "order": "sorted",
                    "normalize": True,
                    "see_behind": True,
                },
                "collision_reward": -1.0,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.4,
                "lane_change_reward": -0.05,
                "duration": 7.0,
                "simulation_frequency": 20,
                "policy_frequency": 5,
            }
        )
        return config

    def _reset(self) -> None:
        self._distance = 0.0
        self._speed_samples = []
        super()._reset()

    def _create_ego_vehicle(self, ego_lane, ego_road_lane):
        position = ego_road_lane.position(60.0, 0.0)
        common = {
            "heading": ego_road_lane.heading_at(60.0),
            "speed": self.EGO_SPEED,
            "target_lane_index": ego_lane,
        }
        if self.ego_kind == "idm_mobil":
            return IDMVehicle(
                self.road,
                position,
                target_speed=30.0,
                **common,
            )
        if self.ego_kind == "mdp":
            return MDPVehicle(
                self.road,
                position,
                target_speeds=np.asarray(self.action_type.target_speeds),
                **common,
            )
        raise ValueError(f"Unknown external ego kind: {self.ego_kind}")

    def _simulate(self, action: int | None = None) -> None:
        frames = int(
            self.config["simulation_frequency"]
            // self.config["policy_frequency"]
        )
        simulation_dt = 1 / self.config["simulation_frequency"]
        if self.ego_kind == "mdp":
            self.action_type.act(1 if action is None else int(action))

        for _ in range(frames):
            if self.ego_kind == "idm_mobil":
                self.vehicle.act()
            for vehicle in self.road.vehicles:
                if vehicle is not self.vehicle:
                    vehicle.act()
            previous_position = self.vehicle.position.copy()
            self.road.step(simulation_dt)
            self._distance += float(
                np.linalg.norm(self.vehicle.position - previous_position)
            )
            self._speed_samples.append(float(self.vehicle.speed))
            self.steps += 1
            self._record_lead_trace()
            self._update_safety_metrics()

    def _is_terminated(self) -> bool:
        return bool(any(vehicle.crashed for vehicle in self.road.vehicles))

    @property
    def scheduled_vehicle(self):
        if self._cutin_vehicle is None:
            raise RuntimeError("The environment must be reset before planning")
        return self._cutin_vehicle

    def external_result(self) -> ExternalEpisode:
        ego_collision = bool(self.vehicle.crashed)
        background_collision = any(
            vehicle.crashed
            for vehicle in self.road.vehicles
            if vehicle is not self.vehicle
        )
        near_miss = not ego_collision and (
            self._min_ttc < 1.5 or self._min_distance < self.NEAR_MISS_CLEARANCE
        )
        mean_speed = (
            float(np.mean(self._speed_samples)) if self._speed_samples else 0.0
        )
        return ExternalEpisode(
            ego_collision=ego_collision,
            background_collision=bool(background_collision),
            near_miss=near_miss,
            completed=(
                not self._is_terminated()
                and self.time >= self.config["duration"]
            ),
            min_ttc=float(self._min_ttc),
            min_distance=float(self._min_distance),
            distance=self._distance,
            mean_speed=mean_speed,
            steps=self.steps,
        )

    def ttc_observation(self) -> np.ndarray:
        """Return the Highway-env finite-MDP TTC grid."""
        return compute_ttc_grid(self, time_quantization=1.0, horizon=6.0)
