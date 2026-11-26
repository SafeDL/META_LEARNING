"""Episode adapters for the three locally trained public architectures."""

import math
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import one_hot

from .learning import DiscreteDriver
from .public_agents import CHECKPOINTS, THREE_LANE_RANGES, kinematics, sha256

CONTEXT = 30
MAX_EPISODE_STEPS = 90


def attention_observation(world, count=15):
    return kinematics(world, count=count, ranges=THREE_LANE_RANGES, extra="orientation")


def attention_network():
    from rl_agents.agents.common.models import EgoAttentionNetwork

    embedding = {
        "type": "MultiLayerPerceptron", "layers": [64, 64], "in": 7, "reshape": False,
    }
    return EgoAttentionNetwork({
        "in": 7, "out": 5,
        "embedding_layer": dict(embedding), "others_embedding_layer": dict(embedding),
        "self_attention_layer": None,
        "attention_layer": {"type": "EgoAttention", "feature_size": 64, "heads": 2},
        "output_layer": {"type": "MultiLayerPerceptron", "layers": [64, 64], "reshape": False},
    })


def transformer_network():
    from modules.decision_transformer import DecisionTransformer

    return DecisionTransformer(
        state_dim=70, act_dim=5, max_length=CONTEXT, max_ep_len=MAX_EPISODE_STEPS,
        hidden_size=128, n_layer=3, n_head=4, n_inner=512, activation_function="relu",
        n_positions=1024, n_ctx=1024, resid_pdrop=0.1, attn_pdrop=0.1, action_tanh=False,
    )


NETWORKS = {
    "social_attention_ddqn": attention_network,
    "decision_transformer": transformer_network,
}


@lru_cache(maxsize=None)
def load_adapted(sut_id):
    checkpoint = torch.load(CHECKPOINTS / sut_id / "policy.pt", weights_only=True, map_location="cpu")
    network = NETWORKS[sut_id]()
    network.load_state_dict(checkpoint["model"], strict=True)
    if any(not torch.isfinite(p).all() for p in network.parameters()):
        raise ValueError(f"Nonfinite adapted checkpoint: {sut_id}")
    return network.eval(), checkpoint


class AttentionPolicy:
    def __init__(self, network):
        self.network = network
        self.driver = DiscreteDriver()
        self.last_decision = None

    @torch.inference_mode()
    def act(self, world):
        obs = torch.from_numpy(attention_observation(world)).unsqueeze(0)
        action = int(self.network(obs).argmax(dim=1).item())
        self.last_decision = {"action": action}
        return self.driver.command(action, world)


def driving_reward(progress, command, ego, world):
    """The continuing, visible part of DrivingEnv's reward for an executed transition."""
    from simulator.highway.timing import DECISION_DT
    from simulator.highway.training_env import safety_penalty

    reward = 0.3 * progress / (30 * DECISION_DT) + 0.01
    reward -= 0.002 * command.acceleration**2 / 64
    applied = command.limited(ego.speed, ego.length)
    lateral_acceleration = ego.speed**2 * math.tan(applied.steering) / ego.length
    reward -= 0.002 * (lateral_acceleration / 6) ** 2
    reward -= 0.03 * (max(0, ego.speed - 32) / 5) ** 2
    return reward - safety_penalty(world)


class TransformerPolicy:
    def __init__(self, network, checkpoint):
        self.network = network
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.remaining = float(checkpoint["initial_return"])
        self.scale = float(checkpoint["scale"])
        self.driver = DiscreteDriver()
        self.history = deque(maxlen=CONTEXT - 1)
        self.last_decision = None
        self.previous = None
        self.step = 0

    @torch.inference_mode()
    def act(self, world):
        if self.previous is not None:
            ego_x, command = self.previous
            self.remaining -= driving_reward(world.ego.x - ego_x, command, world.ego, world)
        obs = torch.from_numpy(attention_observation(world, count=10).ravel())
        current = ((obs - self.mean) / self.std, 0, self.remaining / self.scale, self.step)
        states, acts, returns, steps = zip(*self.history, current)
        actions = one_hot(torch.tensor(acts), num_classes=5).float()
        actions[-1].zero_()
        logits = self.network.get_action(
            torch.stack(states), actions, None, torch.tensor(returns), torch.tensor(steps)
        )
        action = int(logits.argmax().item())
        self.history.append((current[0], action, current[2], self.step))
        self.step += 1
        command = self.driver.command(action, world)
        self.previous = world.ego.x, command
        self.last_decision = {"action": action, "return_to_go": self.remaining}
        return command


def create_adapted(sut_id):
    network, checkpoint = load_adapted(sut_id)
    if sut_id == "decision_transformer":
        return TransformerPolicy(network, checkpoint)
    return AttentionPolicy(network)


def adapted_assets(sut_id):
    if sut_id == "decision_transformer":
        from modules import decision_transformer, model, trajectory_gpt2

        modules = [decision_transformer, model, trajectory_gpt2]
    else:
        from rl_agents.agents.common import models

        modules = [models]
    assets = {module.__name__: sha256(Path(module.__file__).read_bytes()) for module in modules}
    assets["training.json"] = sha256((CHECKPOINTS / sut_id / "training.json").read_bytes())
    return assets
