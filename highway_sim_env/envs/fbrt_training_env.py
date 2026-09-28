"""On-policy PPO environment using the unified evaluator's physical step.

The training policy owns each 5 Hz ego action.  One Gym step advances four
20 Hz physics frames, matching the external-policy evaluation contract.
Scenario sampling is restricted to a caller-supplied training pool; final
confirmation manifests must never be supplied here.
"""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np

from highway_sim_env.envs.fbrt_unified_env import (
    EXECUTION_CONTRACT, FBRTUnifiedEnv, PHYSICS_HZ,
)
from sut_algorithms.highway_env.registry import build_spec_factory


TRAINING_CONTRACT = EXECUTION_CONTRACT + ";external-action-5Hz;reward-progress-v1"


class FBRTTrainingEnv(gym.Env):
    """A scenario-resampling Gym interface for a single frozen PPO build."""

    metadata = {"render_modes": []}

    def __init__(self, cases: list[dict], build_id: str = "ppo_ref_v2") -> None:
        if not cases:
            raise ValueError("the PPO training scenario pool is empty")
        self.spec = build_spec_factory(build_id)
        if self.spec.adapter_kind != "external_meta_policy" or self.spec.control_hz != 5:
            raise ValueError("PPO training requires a 5 Hz external-meta-policy build")
        if PHYSICS_HZ % int(self.spec.control_hz):
            raise ValueError("policy frequency does not divide physics frequency")
        self.cases = tuple(deepcopy(case) for case in cases)
        self.core = FBRTUnifiedEnv(self.spec, self.cases[0])
        self.action_space = self.core.action_space
        self.observation_space = self.core.observation_space
        self.decision_steps = 0
        self.physics_steps = 0
        self.total_decision_steps = 0
        self.total_physics_steps = 0
        self.scenario_id: str | None = None
        self.physics_seed: int | None = None
        self._done = True

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = options or {}
        index = (int(options["scenario_index"]) if "scenario_index" in options else
                 int(self.np_random.integers(len(self.cases))))
        if not 0 <= index < len(self.cases):
            raise IndexError("scenario index is outside the training pool")
        self.physics_seed = (int(options["physics_seed"]) if "physics_seed" in options
                             else int(self.np_random.integers(0, 2**31 - 1)))
        case = self.cases[index]
        self.core.close()
        self.core = FBRTUnifiedEnv(self.spec, case)
        if (self.core.action_space != self.action_space or
                self.core.observation_space != self.observation_space):
            raise ValueError("a training scenario changes the PPO space contract")
        observation, _ = self.core.reset(seed=self.physics_seed)
        self.decision_steps = 0
        self.physics_steps = 0
        self.scenario_id = case["scenario_id"]
        self._done = False
        return np.asarray(observation, dtype=np.float32), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("reset before taking another PPO step")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid PPO action: {action}")
        start_x = float(self.core.vehicle.position[0])
        frames = PHYSICS_HZ // int(self.spec.control_hz)
        limit = int(round(float(self.core.config["duration"]) * PHYSICS_HZ))
        advanced = 0
        for frame in range(frames):
            if self.core.steps >= limit or any(actor.crashed for actor in self.core.actors.values()):
                break
            self.core._advance(action_override=action if frame == 0 else None)
            advanced += 1
            if any(actor.crashed for actor in self.core.actors.values()):
                break
        if not advanced:
            raise AssertionError("PPO step did not advance the physical runner")
        self.decision_steps += 1
        self.physics_steps += advanced
        self.total_decision_steps += 1
        self.total_physics_steps += advanced
        ego_collision = bool(self.core.vehicle.crashed)
        any_collision = any(actor.crashed for actor in self.core.actors.values())
        terminated = bool(any_collision)
        truncated = bool(self.core.steps >= limit and not terminated)
        self._done = terminated or truncated
        progress = float(self.core.vehicle.position[0]) - start_x
        reward = progress / (40.0 * frames / PHYSICS_HZ) - 5.0 * ego_collision
        observation = np.asarray(self.core.observation_type.observe(), dtype=np.float32)
        return observation, float(reward), terminated, truncated, self._info()

    def _info(self) -> dict:
        return {"scenario_id": self.scenario_id,
                "physics_seed": self.physics_seed,
                "decision_steps": self.decision_steps,
                "physics_steps": self.physics_steps,
                "total_decision_steps": self.total_decision_steps,
                "total_physics_steps": self.total_physics_steps,
                "ego_collision": bool(self.core.vehicle.crashed),
                "background_collision": any(actor.crashed for role, actor in
                                            self.core.actors.items() if role != "ego"),
                "training_contract": TRAINING_CONTRACT}

    def close(self) -> None:
        self.core.close()
