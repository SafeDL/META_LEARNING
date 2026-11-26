"""Frozen external driving policies behind the visible-world interface."""

import hashlib
import io
import math
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.road.lane import StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle

from simulator.highway.timing import timing_contract

from .control import Control
from .decision_interface import DecisionInterface

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache/highway_agents"
CHECKPOINTS = ROOT / "controllers/highway/checkpoints/public"
RUNS = ROOT / "results/public_ads/qualification"
MC_REV = "0fc3bc33213f8efd1168daa4f25bebb84cc7678b"
RL_AGENTS_REV = "84df15ea977271e6a4d015f10f9f355f7e866890"
DT_REV = "96a1dd3fd0ec8fb489f857eeafd07df4f212e654"
WEIGHTS = {
    "online_mc_ppo": (
        "online_mc/highway_ppo/model.zip",
        "217343510d1bf947a2c32035a95005c111a83d438499e33c034789ccfd2c6ba3",
        MC_REV,
    ),
}
LOCAL_SOURCES = {
    "social_attention_ddqn": RL_AGENTS_REV,
    "decision_transformer": DT_REV,
}
THREE_LANE_RANGES = ((-200, 200), (-12, 12), (-80, 80), (-80, 80))
META_SPEEDS = tuple(range(0, 41, 5))


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def weight_bytes(sut_id):
    filename, expected, _ = WEIGHTS[sut_id]
    path = CACHE / filename
    content = path.read_bytes()
    if sha256(content) != expected:
        raise ValueError(f"Public checkpoint differs from registered SHA-256: {sut_id}")
    return content


def kinematics(world, *, count, ranges, absolute=False, extra=None, see_behind=False):
    """HighwayEnv kinematics applied to actors that passed the common visibility boundary."""
    ego = world.ego
    neighbors = [
        v for v in world.vehicles
        if math.hypot(v.x - ego.x, v.y - ego.y) < 200
        and (see_behind or v.x - ego.x > -2 * ego.length)
    ]
    neighbors.sort(key=lambda v: (abs(v.x - ego.x), v.actor_id))
    width = {None: 5, "heading": 6, "orientation": 7, "lane_offsets": 8}[extra]
    output = np.zeros((count, width), dtype=np.float32)
    low, high = np.asarray(ranges).T
    origin = np.array([ego.x, ego.y, ego.vx, ego.vy])
    for i, actor in enumerate((ego, *neighbors[:count - 1])):
        values = np.array([actor.x, actor.y, actor.vx, actor.vy])
        if i and not absolute:
            values -= origin
        output[i, 0] = 1
        output[i, 1:5] = np.clip(2 * (values - low) / (high - low) - 1, -1, 1)
        if extra == "heading":
            output[i, 5] = actor.heading
        elif extra == "orientation":
            output[i, 5:] = math.cos(actor.heading), math.sin(actor.heading)
        elif extra == "lane_offsets":
            lane_y = min(world.lane_centers, key=lambda y: abs(actor.y - y))
            output[i, 5:] = actor.x, actor.y - lane_y, math.remainder(actor.heading, 2 * math.pi)
    return output


def public_observation(sut_id, world, *, shield=False):
    if sut_id != "online_mc_ppo":
        raise ValueError(f"Unregistered public checkpoint: {sut_id}")
    return kinematics(
        world, count=5, ranges=THREE_LANE_RANGES, extra="heading" if shield else None
    )


class NativeMetaDriver:
    """Reuse the source tactical controller on a road containing only the visible ego."""

    def __init__(self, speed_grid):
        self.speed_grid = speed_grid
        self.vehicle = None

    def update(self, world):
        ego = world.ego
        if self.vehicle is None:
            network = RoadNetwork()
            for y in world.lane_centers:
                network.add_lane("0", "1", StraightLane([0, y], [2000, y]))
            self.vehicle = MDPVehicle(
                Road(network=network), [ego.x, ego.y], speed=ego.speed,
                heading=ego.heading, target_speeds=self.speed_grid,
            )
            self.action_type = DiscreteMetaAction(
                SimpleNamespace(vehicle=self.vehicle), target_speeds=self.speed_grid
            )
        vehicle = self.vehicle
        vehicle.position[:] = ego.x, ego.y
        vehicle.speed, vehicle.heading = ego.speed, ego.heading
        vehicle.on_state_update()

    def get_available_actions(self):
        return self.action_type.get_available_actions()

    def command(self, action):
        self.action_type.act(action)
        return Control(**self.vehicle.action)


@lru_cache(maxsize=None)
def load_public_model(sut_id):
    content = io.BytesIO(weight_bytes(sut_id))
    from stable_baselines3 import PPO

    # Historic training schedules are pickled Python functions and unused in inference.
    replace_schedules = dict(lr_schedule=None, exploration_schedule=None)
    model = PPO.load(content, device="cpu", custom_objects=replace_schedules)
    if model.observation_space.shape != (5, 5):
        raise ValueError(f"Incorrect observation space: {sut_id}")
    network = model.policy
    if any(not p.isfinite().all() for p in network.parameters()):
        raise ValueError(f"Nonfinite public checkpoint: {sut_id}")
    return network.eval()


class PublicPolicy:
    def __init__(self, sut_id):
        self.sut_id = sut_id
        self.model = load_public_model(sut_id)
        self.driver = NativeMetaDriver(META_SPEEDS)
        from .public_shield import OnlineShield

        self.shield = OnlineShield()
        self.last_decision = None

    @torch.inference_mode()
    def scores(self, observation):
        tensor = torch.as_tensor(observation[None])
        return self.model.get_distribution(tensor).distribution.probs[0].numpy()

    def act(self, world):
        observation = public_observation(self.sut_id, world, shield=True)
        scores = self.scores(observation[:, :5])
        self.driver.update(world)
        action, rejected = self.shield.choose(
            np.argsort(scores)[::-1], observation, world, self.driver.get_available_actions()
        )
        self.last_decision = {"action": action, "shield_rejected": rejected}
        command = self.driver.command(action)
        self.last_decision.update(
            target_lane=int(self.driver.vehicle.target_lane_index[2]),
            target_speed_mps=float(self.driver.vehicle.target_speed),
        )
        return command


def create(sut_id):
    """Instantiate one new episode, including fresh recurrent or budget state."""
    if sut_id in WEIGHTS:
        return DecisionInterface(PublicPolicy(sut_id))
    from .adapted_agents import create_adapted

    return DecisionInterface(create_adapted(sut_id))


def runtime_contract(sut_id):
    files = [
        "controllers/highway/public_agents.py",
        "controllers/highway/decision_interface.py",
        "controllers/highway/control.py",
        "simulator/highway/collision_env.py",
        "simulator/highway/observation.py",
        "simulator/highway/timing.py",
        "experiments/scenarios.py",
    ]
    libraries = ["highway-env", "torch", "numpy"]
    assets = {}
    if sut_id in WEIGHTS:
        checkpoint_sha256 = sha256(weight_bytes(sut_id))
        revision = WEIGHTS[sut_id][2]
        libraries.append("stable-baselines3")
        from .public_shield import shield_assets

        files.append("controllers/highway/public_shield.py")
        libraries.append("maude")
        assets.update(shield_assets())
    else:
        from .adapted_agents import adapted_assets

        revision = LOCAL_SOURCES[sut_id]
        checkpoint_sha256 = sha256((CHECKPOINTS / sut_id / "policy.pt").read_bytes())
        files.extend(("controllers/highway/adapted_agents.py", "controllers/highway/learning.py"))
        assets.update(adapted_assets(sut_id))
        if sut_id == "decision_transformer":
            files.append("simulator/highway/training_env.py")
            libraries.append("transformers")
    return {
        "sut_id": sut_id,
        "upstream_revision": revision,
        "checkpoint_sha256": checkpoint_sha256,
        "timing": timing_contract(),
        "dependencies": {package: version(package) for package in libraries},
        "source_files": {name: sha256((ROOT / name).read_bytes()) for name in files},
        "upstream_assets": assets,
    }
