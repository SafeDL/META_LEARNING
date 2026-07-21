from __future__ import annotations

import copy
import unittest
import numpy as np
import torch

from pearl_learning.src.adapters.base import MetaDriveAdapterBase
from pearl_learning.src.context_encoder import ContextEncoder
from pearl_learning.src.io import read_config
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.replay import TaskReplayBuffer, Transition
from pearl_learning.src.reward import compute_reward
from pearl_learning.src.observation import OBS_FIELDS, build_observation
from pearl_learning.src.routes import RoutePolyline, wrap_to_pi
from pearl_learning.src.task_spec import LogicalScenarioTaskSpec
from pearl_learning.src.taskbook import build_taskbook


def transition(episode: str, terminated: bool = False, truncated: bool = False) -> Transition:
    return Transition(np.zeros(37, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.0, np.ones(37, dtype=np.float32), terminated, truncated, "horizon" if truncated else "running", "task", episode, "case", "prior_support", 0)


class DummyVehicle:
    LENGTH = 5.0
    WIDTH = 2.0
    def __init__(self, position, heading=0.0, crash=False):
        self.position = np.asarray(position, dtype=float); self.heading_theta = heading; self.crash_vehicle = crash
        self.velocity = np.asarray([5.0, 0.0]); self.acceleration = np.asarray([0.0, 0.0])
        self.lane = type("Lane", (), {"width": 3.8})()


class DummyAdapter(MetaDriveAdapterBase):
    def build_env(self, task, case, config):
        raise NotImplementedError


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_config("pearl_learning/configs/merge_family_pearl.yaml")

    def test_task_schema_rejects_non_current_and_split_hashes_are_disjoint(self):
        taskbook = build_taskbook(self.config)
        task = taskbook["meta_train"][0]
        invalid = task.to_dict(); invalid["schema"] = "unsupported"
        with self.assertRaises(ValueError):
            LogicalScenarioTaskSpec.from_dict(invalid)
        train_hashes = {task.map_hash for task in taskbook["meta_train"]}
        self.assertFalse(train_hashes & {task.map_hash for task in taskbook["meta_validation"]})
        self.assertFalse(train_hashes & {task.map_hash for task in taskbook["meta_test_logical"]})

    def test_route_projection_wraps_heading_and_uses_arc_length(self):
        route = RoutePolyline((("a", "b", 0),), np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]), np.asarray([0.0, 10.0, 20.0]), (20.0,))
        projection = route.projection([5.0, 1.0], 2 * np.pi - 0.2)
        self.assertAlmostEqual(projection.s_m, 5.0, places=5)
        self.assertAlmostEqual(projection.lateral_m, 1.0, places=5)
        self.assertAlmostEqual(projection.heading_error, -0.2, places=5)
        self.assertAlmostEqual(wrap_to_pi(3 * np.pi), -np.pi, places=5)

    def test_marking_violation_is_separate_from_wrong_route_penalty(self):
        events = {"lane_marking_violation": True, "wrong_route": False}
        reward = compute_reward(5.0, 50.0, np.zeros(2), np.zeros(2), events, self.config["reward"])
        self.assertEqual(reward.wrong_route, 0.0)
        self.assertEqual(reward.lane_marking_violation, 0.0)

    def test_observation_priority_features_are_complementary(self):
        route = RoutePolyline((("a", "b", 0),), np.asarray([[0.0, 0.0], [20.0, 0.0]]), np.asarray([0.0, 20.0]), (20.0,))
        frame = {"adversary_route": route, "sut_route": route, "adversary_conflict_s_m": 10.0, "sut_conflict_s_m": 10.0, "priority_spec": {"sut_has_priority": True}}
        topology = {"num_incoming_branches": 2.0, "num_outgoing_branches": 1.0, "adversary_lane_count": 1.0, "sut_lane_count": 1.0, "merge_length_m": 20.0, "conflict_radius_m": 3.0, "adversary_route_curvature": 0.0, "sut_route_curvature": 0.0, "adversary_speed_limit_mps": 15.0, "sut_speed_limit_mps": 15.0, "num_conflict_zones": 1.0}
        observation = build_observation(DummyVehicle([0.0, 0.0]), DummyVehicle([1.0, 0.0]), frame, topology, self.config)
        self.assertEqual(float(observation[OBS_FIELDS.index("adversary_priority")]), 0.0)
        self.assertEqual(float(observation[OBS_FIELDS.index("sut_priority")]), 1.0)

    def test_pairwise_collision_uses_obb_without_crash_flags(self):
        adapter = DummyAdapter()
        first, second = DummyVehicle([0.0, 0.0]), DummyVehicle([1.0, 0.0])
        hit, method = adapter.target_contact(object(), first, second)
        self.assertTrue(hit); self.assertEqual(method, "obb_overlap")
        hit, method = adapter.target_contact(object(), DummyVehicle([0.0, 0.0], crash=True), DummyVehicle([100.0, 0.0], crash=True))
        self.assertFalse(hit); self.assertEqual(method, "no_pairwise_contact")

    def test_episode_balanced_context_preserves_episode_provenance(self):
        buffer = TaskReplayBuffer()
        first = [transition("episode-a") for _ in range(2)]; first[-1] = transition("episode-a", terminated=True)
        buffer.add_episode(first, "task")
        second = [transition("episode-b") for _ in range(2)]; second.append(transition("episode-b", truncated=True))
        buffer.add_episode(second, "task")
        groups = buffer.sample_episode_balanced(64, 4, np.random.default_rng(7))
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(group) == 4 for group in groups))
        self.assertEqual({group[0].episode_id for group in groups}, {"episode-a", "episode-b"})
        episodes = {episode.episode_id: episode for episode in buffer.episodes}
        self.assertTrue(episodes["episode-a"].terminated)
        self.assertTrue(episodes["episode-b"].truncated)
        self.assertEqual(episodes["episode-b"].termination_reason, "horizon")

    def test_context_encoder_pools_episode_before_gaussian_product(self):
        encoder = ContextEncoder(4, 2, [8])
        mu, log_var = encoder(torch.zeros((2, 3, 4, 4)))
        self.assertEqual(tuple(mu.shape), (2, 2)); self.assertEqual(tuple(log_var.shape), (2, 2))
        with self.assertRaises(ValueError):
            encoder(torch.zeros((2, 3, 4)))

    def test_parameter_hash_includes_target_critics_and_alpha(self):
        agent = PEARLAgent(37, 2, self.config, torch.device("cpu"))
        before = agent.parameter_hash()
        with torch.no_grad():
            next(agent.target_q1.parameters()).add_(1.0)
        self.assertNotEqual(before, agent.parameter_hash())
        before = agent.parameter_hash()
        with torch.no_grad():
            agent.log_alpha.add_(0.5)
        self.assertNotEqual(before, agent.parameter_hash())

    def test_truncated_transition_is_not_bootstrap_terminal(self):
        row = transition("episode", terminated=False, truncated=True)
        self.assertFalse(row.terminated)
        self.assertTrue(row.truncated)


if __name__ == "__main__":
    unittest.main()
