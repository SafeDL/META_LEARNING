"""Train registered public architectures on development-only native driving courses."""

import argparse
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from controllers.highway.adapted_agents import (
    CONTEXT,
    MAX_EPISODE_STEPS,
    attention_network,
    attention_observation,
    transformer_network,
)
from controllers.highway.decision_interface import public_world
from controllers.highway.learning import DiscreteDriver
from controllers.highway.public_agents import CHECKPOINTS, LOCAL_SOURCES, sha256
from simulator.highway.collision_env import CollisionWorld
from simulator.highway.training_env import DrivingEnv

from experiments.regression.dqn_learning import ordinary_courses
from experiments.paper_results import create_controller, published_release
from experiments.qualification import COURSES, DEVELOPMENT_COURSES
from experiments.storage import write_json

SEED = 1301

class TrainingWorld(CollisionWorld):
    """Omit per-frame metric/trajectory logging while retaining native termination."""

    def _record(self):
        ego = self.ego
        half_width = (
            abs(math.sin(ego.heading)) * ego.LENGTH + abs(math.cos(ego.heading)) * ego.WIDTH
        ) / 2
        if (
            ego.position[1] - half_width < self.lane_centers[0] - 2
            or ego.position[1] + half_width > self.lane_centers[-1] + 2
        ):
            self.violations.add("road_departure")


class PublicDevelopmentEnv(DrivingEnv):
    def __init__(self, *, world_class=TrainingWorld):
        super().__init__()
        self.world_class = world_class
        self.curriculum = ordinary_courses("training")
        # These physical scenario kinds and fractional coordinates are disjoint from
        # the four integer formal and animation pools.
        def coordinate(course):
            return course.scenario.kind, tuple(sorted(course.scenario.parameters.items()))

        if {coordinate(c) for c in self.curriculum} & {
            coordinate(c) for c in (*COURSES, *DEVELOPMENT_COURSES)
        }:
            raise ValueError("Public-agent development intersects an acceptance coordinate")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        if self.world is not None:
            self.world.close()
        course = self.curriculum[int(self.np_random.integers(len(self.curriculum)))]
        self.world = self.world_class(
            course.scenario, lane_centers=course.lane_centers, duration_s=course.duration_s,
            environment_seed=int(self.np_random.integers(1_000_000)),
        )
        self.driver = DiscreteDriver()
        return self.driver.observation(self.world.visible_world()), {}

    def entities(self, count=15):
        return attention_observation(public_world(self.world.visible_world()), count=count)


def log_progress(curve, update, **measurements):
    row = {"step": update, **measurements}
    curve.append(row)
    print(row, flush=True)


def train_attention(env, rng, device):
    limit, capacity, batch_size = 100_000, 15_000, 64
    gamma = 0.99
    network = attention_network().to(device)
    network.reset()
    target = deepcopy(network).eval()
    opt = torch.optim.Adam(network.parameters(), lr=5e-4)
    states = np.empty((capacity, 15, 7), dtype=np.float32)
    successors = np.empty_like(states)
    actions = np.empty(capacity, dtype=np.int64)
    rewards = np.empty(capacity, dtype=np.float32)
    terminals = np.empty(capacity, dtype=np.float32)
    env.reset(seed=SEED)
    observation = env.entities()
    curve, episode_return, episodes, collisions = [], 0.0, 0, 0
    losses = []
    for step in range(limit):
        epsilon = 0.05 + 0.95 * math.exp(-step / 6000)
        if rng.random() < epsilon:
            action = int(rng.integers(5))
        else:
            with torch.no_grad():
                action = int(network(torch.as_tensor(observation[None], device=device)).argmax())
        _, reward, terminated, truncated, info = env.step(action)
        successor = env.entities()
        done = terminated or truncated
        slot = step % capacity
        states[slot], successors[slot], actions[slot] = observation, successor, action
        rewards[slot], terminals[slot] = reward, done
        episode_return += reward
        observation = successor
        if done:
            collisions += info["collision"]
            episodes += 1
            env.reset()
            observation = env.entities()
        if step + 1 >= batch_size:
            batch = rng.choice(min(capacity, step + 1), batch_size, replace=False)
            obs, ns, acts, r, d = [
                torch.as_tensor(array[batch], device=device)
                for array in (states, successors, actions, rewards, terminals)
            ]
            with torch.no_grad():
                next_actions = network(ns).argmax(dim=1, keepdim=True)
                values = r + gamma * (1 - d) * target(ns).gather(1, next_actions).squeeze(1)
            q = network(obs).gather(1, acts[:, None]).squeeze(1)
            loss = F.mse_loss(q, values)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            opt.step()
            losses.append(loss.detach())
        if (step + 1) % 512 == 0:
            target.load_state_dict(network.state_dict())
        if (step + 1) % 1000 == 0:
            log_progress(
                curve, step + 1, loss=float(torch.stack(losses).mean()),
                episodes=episodes, collisions=collisions, reward_sum=episode_return,
            )
            losses.clear()
            episode_return = episodes = collisions = 0
    return {"model": network.state_dict()}, curve, {
        "control_steps": limit, "gamma": gamma, "batch_size": batch_size,
        "memory_capacity": capacity, "target_update": 512, "epsilon_final": 0.05,
        "epsilon_decay_steps": 6000, "learning_rate": 5e-4,
    }


def build_demonstrations(env, directory):
    driver = create_controller("target_dqn").policy
    teacher = published_release()
    env.reset(seed=SEED)
    shape = (500, MAX_EPISODE_STEPS)
    states = np.zeros((*shape, 70), np.float32)
    actions = np.zeros(shape, np.int64)
    rewards = np.zeros(shape, np.float32)
    lengths = np.zeros(500, np.int64)
    records = []
    for episode in range(500):
        scene = asdict(env.world.scenario)
        observation = env.driver.observation(env.world.visible_world())
        for step in range(MAX_EPISODE_STEPS):
            states[episode, step] = env.entities(count=10).ravel()
            action, _ = driver.model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            actions[episode, step], rewards[episode, step] = int(action), reward
            if terminated or truncated:
                lengths[episode] = step + 1
                records.append({
                    "scenario": scene, "seed": env.world.environment_seed,
                    "collision": info["collision"], "violations": list(info["violations"]),
                    "steps": step + 1,
                })
                break
        if not lengths[episode]:
            raise ValueError("Demonstration exceeded its registered duration")
        env.reset()
    archive = directory / "demonstrations.npz"
    np.savez_compressed(archive, states=states, actions=actions, rewards=rewards, lengths=lengths)
    write_json(directory / "demonstrations.json", {
        "teacher_sha256": teacher["checkpoint_sha256"], "data_sha256": sha256(archive.read_bytes()),
        "episodes": records,
    })
    return states, actions, rewards, lengths


def train_transformer(env, rng, device, directory):
    network = transformer_network().to(device)
    states, actions, rewards, lengths = build_demonstrations(env, directory)
    times = np.arange(MAX_EPISODE_STEPS)
    active = times < lengths[:, None]
    mean = states[active].mean(axis=0)
    std = states[active].std(axis=0) + 1e-6
    scale = float(lengths.mean())
    returns = np.flip(np.cumsum(np.flip(rewards, axis=1), axis=1), axis=1).copy()
    initial_return = float(np.quantile(returns[:, 0], 0.75))
    opt = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min((step + 1) / 10000, 1))
    curve, recent_loss, recent_accuracy = [], [], []
    offsets = np.arange(CONTEXT)
    for step in range(10_000):
        ep = rng.choice(len(lengths), size=32, p=lengths / lengths.sum())
        start = rng.integers(lengths[ep])
        length = np.minimum(CONTEXT, lengths[ep] - start)
        mask = offsets >= CONTEXT - length[:, None]
        ti = np.maximum(0, start[:, None] + offsets - (CONTEXT - length[:, None]))
        valid = torch.as_tensor(mask, device=device)
        state_inputs = np.where(mask[..., None], (states[ep[:, None], ti] - mean) / std, 0)
        labels = torch.as_tensor(actions[ep[:, None], ti], device=device)
        action_input = F.one_hot(labels, num_classes=5) * valid.unsqueeze(2)
        rtg = np.where(mask, returns[ep[:, None], ti] / scale, 0).astype(np.float32)
        _, logits, _ = network(
            torch.as_tensor(state_inputs, device=device), action_input.float(), None,
            torch.as_tensor(rtg[:, :, None], device=device),
            torch.as_tensor(np.where(mask, ti, 0), device=device), attention_mask=valid.long(),
        )
        loss = F.cross_entropy(logits[valid], labels[valid])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 0.25)
        opt.step()
        schedule.step()
        recent_loss.append(loss.detach())
        recent_accuracy.append((logits.detach().argmax(dim=-1)[valid] == labels[valid]).float().mean())
        if (step + 1) % 500 == 0:
            log_progress(curve, step + 1, loss=float(torch.stack(recent_loss).mean()),
                         action_accuracy=float(torch.stack(recent_accuracy).mean()))
            recent_loss.clear()
            recent_accuracy.clear()
    return {
        "model": network.state_dict(), "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
        "scale": scale, "initial_return": initial_return,
    }, curve, {
        "optimizer_updates": 10_000, "demonstration_episodes": 500, "context_length": CONTEXT,
        "initial_return": initial_return, "return_scale": scale, "batch_size": 32,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sut", choices=LOCAL_SOURCES)
    parser.add_argument("--output-dir", type=Path, help="Isolated output pending validation")
    args = parser.parse_args()
    directory = args.output_dir if args.output_dir is not None else CHECKPOINTS / args.sut
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "policy.pt"
    training_source_sha256 = sha256(Path(__file__).read_bytes())
    if checkpoint_path.exists():
        raise FileExistsError(f"Frozen training output exists: {checkpoint_path}")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = PublicDevelopmentEnv()
    start = time.perf_counter()
    try:
        if args.sut == "decision_transformer":
            checkpoint, curve, settings = train_transformer(env, rng, device, directory)
        else:
            checkpoint, curve, settings = train_attention(env, rng, device)
    finally:
        env.close()
    if any(not torch.isfinite(p).all() for p in checkpoint["model"].values()):
        raise ValueError("Nonfinite training result")
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    write_json(directory / "training.json", {
        "sut_id": args.sut, "seed": SEED, "device": str(device),
        "elapsed_s": time.perf_counter() - start, "upstream_revision": LOCAL_SOURCES[args.sut],
        "checkpoint_sha256": sha256(temporary.read_bytes()), "settings": settings,
        "training_source_sha256": training_source_sha256,
        "curriculum": [asdict(course) for course in env.curriculum], "curve": curve,
    })
    temporary.replace(checkpoint_path)


if __name__ == "__main__":
    main()
