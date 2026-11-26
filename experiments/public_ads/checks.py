"""Recompute public model interfaces on independent fixed states before driving."""

import contextlib
import io
import math
import runpy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from highway_env.envs.common.observation import KinematicObservation
from highway_env.road.lane import StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from controllers.highway.adapted_agents import CONTEXT, attention_observation, load_adapted
from controllers.highway.public_agents import (
    CACHE,
    THREE_LANE_RANGES,
    create,
    public_observation,
    weight_bytes,
)
from controllers.highway.public_shield import shield_states
from simulator.highway.observation import VehicleState, VisibleWorld

TOLERANCE = 1e-5
FEATURES = ("presence", "x", "y", "vx", "vy")
SCHEDULES = dict(lr_schedule=None, exploration_schedule=None)


def fixture(gap):
    return VisibleWorld(
        0, VehicleState("ego", 150, 4, 27, 0),
        (
            VehicleState("vehicle_00", 150 + gap, 4, 8, 0),
            VehicleState("vehicle_01", 162, 0, 26, -0.04),
            VehicleState("vehicle_02", 180, 8, 26, 0.04),
            VehicleState("vehicle_03", 145, 8, 29, 0),
            VehicleState("vehicle_04", 133, 0, 30, 0),
        ),
    )


def reference_road(world):
    network = RoadNetwork()
    for y in world.lane_centers:
        network.add_lane("0", "1", StraightLane([0, y], [2000, y]))
    road = Road(network=network)
    road.vehicles = [
        Vehicle(road, [s.x, s.y], heading=s.heading, speed=s.speed)
        for s in (world.ego, *world.vehicles)
    ]
    return SimpleNamespace(vehicle=road.vehicles[0], road=road, PERCEPTION_DISTANCE=200)


def reference_observation(sut, world):
    env = reference_road(world)
    extra = {
        "online_mc_ppo": ("heading",),
        "social_attention_ddqn": ("cos_h", "sin_h"),
        "decision_transformer": ("cos_h", "sin_h"),
    }.get(sut, ())
    rows = {"social_attention_ddqn": 15, "decision_transformer": 10}.get(sut, 5)
    observation = KinematicObservation(
        env, features=list(FEATURES + extra), vehicles_count=rows,
        features_range=dict(zip(FEATURES[1:], THREE_LANE_RANGES, strict=True)), absolute=False,
    ).observe()
    return observation


def maximum_error(actual, reference):
    error = float(np.max(np.abs(actual - reference)))
    if not math.isfinite(error) or error > TOLERANCE:
        raise ValueError(f"Interface differs from the reference by {error}")
    return error


def reference_model(sut):
    content = io.BytesIO(weight_bytes(sut))
    from stable_baselines3 import PPO

    return PPO.load(content, device="cpu", custom_objects=SCHEDULES)


def compare_shield(policy):
    upstream = runpy.run_path(str(CACHE / "online_mc/SafetyShield.py"))
    rows = []
    for gap in (110, 12):
        world = fixture(gap)
        observation = public_observation("online_mc_ppo", world, shield=True)
        if not observation[:, 0].all():
            raise ValueError("Shield comparison requires all four NPC slots populated")
        policy.driver.update(world)
        ego, npcs = shield_states(world, observation)
        env = reference_road(world)
        heading = env.vehicle.heading
        e = env.vehicle
        ego_ref = np.round([e.position[0], -e.position[1], e.velocity[0], -e.velocity[1],
                            heading * 180 / np.pi], 3)
        ref = reference_observation("online_mc_ppo", world)
        npc_ref = np.round(
            ref[1:, 1:] * np.array([200, -12, 80, -80, 180 / np.pi])
            + np.r_[ego_ref[:4], 0], 3,
        )
        maximum_error(np.asarray(ego), ego_ref)
        maximum_error(np.asarray(npcs), npc_ref)
        keys = ("x", "y", "vx", "vy", "heading")
        state_info = {
            "timeStamp": 0, "ego": dict(zip(keys, ego, strict=True)),
            "npcs": [dict(zip(keys, npc, strict=True)) for npc in npcs],
        }
        available = policy.driver.get_available_actions()
        ranking = np.array([1, 3, 2, 0, 4])
        with contextlib.redirect_stdout(io.StringIO()):
            action, rejected = upstream["choose_action"](policy.driver, ranking, state_info)
            allowed = [
                upstream["validate"](policy.driver, i, state_info) for i in range(5)
            ]
        actual = policy.shield.choose(ranking, observation, world, available)
        actual_allowed = [
            i == 4 or (i in available and policy.shield.allows(i, ego, npcs))
            for i in range(5)
        ]
        if actual != (action, rejected) or actual_allowed != allowed:
            raise ValueError("Maude action rejection differs from the author's implementation")
        rows.append({"gap_m": gap, "action": action, "rejected": rejected,
                     "allowed": allowed, "test_ranking": ranking.tolist(),
                     "ppo_ranking": np.argsort(policy.scores(observation[:, :5]))[::-1].tolist()})
    if not any(row["rejected"] for row in rows) or not any(not row["rejected"] for row in rows):
        raise ValueError("Shield fixtures need both an intervention and a pass-through")
    return rows


@torch.inference_mode()
def external_interface(sut):
    policy = create(sut).policy
    reference = reference_model(sut)
    observation_error, scores_error = [], []
    for world in (fixture(12), fixture(110),
                  replace(fixture(110), ego=replace(fixture(110).ego, y=8), vehicles=())):
        shield = sut == "online_mc_ppo"
        observation = public_observation(sut, world, shield=shield)
        expected = reference_observation(sut, world)
        observation_error.append(maximum_error(observation, expected))
        values = expected[:, :5] if shield else expected
        tensor = torch.as_tensor(values[None])
        scores = policy.scores(observation[:, :5] if shield else observation)
        logits = reference.policy.get_distribution(tensor).distribution.logits
        ref_scores = torch.softmax(logits, dim=-1)[0].numpy()
        action = int(np.argsort(ref_scores)[::-1][0])
        scores_error.append(maximum_error(scores, ref_scores))
        predicted = int(np.argsort(scores)[::-1][0])
        if predicted != action:
            raise ValueError("Discrete action disagrees with the source policy")
    result = {"observation_max_abs_error": max(observation_error),
              "network_max_abs_error": max(scores_error), "fixtures": 3}
    if sut == "online_mc_ppo":
        result["shield_cases"] = compare_shield(policy)
    return result


@torch.inference_mode()
def adapted_interface(sut):
    network, _ = load_adapted(sut)
    world = fixture(12)
    count = 15 if sut == "social_attention_ddqn" else 10
    obs = attention_observation(world, count=count)
    encoding_error = maximum_error(obs, reference_observation(sut, world))
    if sut == "social_attention_ddqn":
        before, after = network(torch.from_numpy(np.stack((obs, obs[[0, *range(14, 0, -1)]]))))
        changed_padding = obs.copy()
        changed_padding[changed_padding[:, 0] == 0, 1:] = 99
        empty = network(torch.from_numpy(changed_padding[None]))[0]
        return {"permutation_error": maximum_error(before.numpy(), after.numpy()),
                "empty_row_error": maximum_error(before.numpy(), empty.numpy()),
                "observation_max_abs_error": encoding_error}
    rng = torch.Generator().manual_seed(1301)
    states = torch.randn(1, CONTEXT, 70, generator=rng)
    actions = torch.randn(1, CONTEXT, 5, generator=rng)
    returns = torch.randn(1, CONTEXT, 1, generator=rng)
    times = torch.arange(CONTEXT).reshape(1, -1)
    _, first, _ = network(states, actions, None, returns, times)
    states[:, 9:] *= -1
    actions[:, 8:] *= -1
    returns[:, 9:] *= -1
    _, other, _ = network(states, actions, None, returns, times)
    error = maximum_error(first[:, :9].numpy(), other[:, :9].numpy())
    return {"future_prefix_error": error, "observation_max_abs_error": encoding_error}
