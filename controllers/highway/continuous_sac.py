"""Frozen continuous SAC policy on the common visible-world interface."""

from functools import lru_cache
from io import BytesIO

import numpy as np

from .control import Control, steering_limit
from .decision_interface import DecisionInterface
from .learning import ACTION_INTERFACE, DiscreteDriver
from .public_agents import CHECKPOINTS, sha256

SUT_ID = "continuous_sac"
CHECKPOINT = CHECKPOINTS / SUT_ID / "policy.pt"


def continuous_control(action, ego):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("Continuous SAC requires two finite control values")
    acceleration, steering = np.clip(action, -1, 1)
    return Control(
        -2.5 + 5.5 * float(acceleration),
        float(steering) * steering_limit(ego.speed, ego.length),
    ).limited(ego.speed, ego.length)


class ContinuousSAC:
    def __init__(self, model):
        self.model = model
        self.driver = DiscreteDriver()
        self.last_decision = None

    def act(self, world):
        action, _ = self.model.predict(self.driver.observation(world), deterministic=True)
        command = continuous_control(action, world.ego)
        self.last_decision = {"action": action.tolist()}
        return command


@lru_cache(maxsize=1)
def _load(path, expected_digest):
    from stable_baselines3 import SAC

    content = path.read_bytes()
    if sha256(content) != expected_digest:
        raise ValueError("Continuous SAC checkpoint changed before loading")
    model = SAC.load(BytesIO(content), device="cpu")
    policy = model.policy
    if (
        policy.observation_space.shape != (ACTION_INTERFACE["observation_dimension"],)
        or policy.action_space.shape != (2,)
        or not np.all(policy.action_space.low == -1)
        or not np.all(policy.action_space.high == 1)
        or any(not parameter.isfinite().all() for parameter in policy.parameters())
    ):
        raise ValueError("Invalid continuous SAC checkpoint")
    return policy.eval()


def create(path=CHECKPOINT, *, digest=None):
    digest = sha256(path.read_bytes()) if digest is None else digest
    return DecisionInterface(ContinuousSAC(_load(path, digest)))
