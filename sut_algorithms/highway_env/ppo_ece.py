"""Frozen PPO policy reproduced from ece-rl-highway-driving."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from highway_env_benchmark.envs.external_cutin import ExternalCutInEnv


class PPOPolicy:
    ego_kind = "mdp"
    name = "ppo_ece"

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = Path(checkpoint)
        self.model = None

    def reset(self) -> None:
        return None

    def load(self) -> None:
        if self.model is None:
            from stable_baselines3 import PPO

            self.model = PPO.load(self.checkpoint, device="cpu")

    def act(self, env: ExternalCutInEnv) -> int:
        self.load()
        observation = env.observation_type.observe()
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

