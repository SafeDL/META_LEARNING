"""Frozen PPO policy reproduced from ece-rl-highway-driving."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import TYPE_CHECKING

PPO_CHECKPOINT = Path("sut_algorithms/highway_env/checkpoints/ppo_ece/vd_1_5_trial_1.zip")

if TYPE_CHECKING:
    from highway_sim_env.envs.external_cutin import ExternalCutInEnv


@lru_cache(maxsize=8)
def _frozen_model(path: str, modified_ns: int, size: int):
    """Reuse an immutable feed-forward PPO policy across physical episodes."""
    from stable_baselines3 import PPO

    return PPO.load(path, device="cpu")


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
            path = self.checkpoint.resolve()
            stat = path.stat()
            self.model = _frozen_model(str(path), stat.st_mtime_ns, stat.st_size)

    def act(self, env: ExternalCutInEnv) -> int:
        self.load()
        observation = env.observation_type.observe()
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
