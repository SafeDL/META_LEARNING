from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pearl_learning.src.benchmark_calibration import calibrate_thresholds, longitudinal_policy
from pearl_learning.src.casebook_v2 import solve_adversary_spawn
from pearl_learning.src.causal_audit import _actor_means
from pearl_learning.src.critical import CRITICAL_METRIC_SCHEMA, critical_measurements
from pearl_learning.src.io import read_config
from pearl_learning.src.metrics import EpisodeMetrics
from pearl_learning.src.observation import (
    DYNAMIC_OBSERVATION_DIM,
    DYNAMIC_OBSERVATION_SCHEMA,
    DYNAMIC_OBS_FIELDS,
)
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.replay import Transition
from pearl_learning.src.reward import compute_reward
from pearl_learning.src.taskbook import build_taskbook


class MethodFlowV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = read_config("pearl_learning/configs/merge_method_flow_pilot.yaml")

    def test_dynamic_observation_contract_is_24d_and_static_free(self) -> None:
        self.assertEqual(DYNAMIC_OBSERVATION_SCHEMA, "logical_merge_dynamic_obs_v1")
        self.assertEqual(len(DYNAMIC_OBS_FIELDS), DYNAMIC_OBSERVATION_DIM)
        self.assertEqual(DYNAMIC_OBSERVATION_DIM, 24)
        forbidden = {
            "adversary_priority", "sut_priority", "num_incoming_branches", "num_outgoing_branches",
            "adversary_lane_count", "sut_lane_count", "merge_length", "conflict_radius",
            "adversary_route_curvature", "sut_route_curvature", "adversary_speed_limit",
            "sut_speed_limit", "num_conflict_zones",
        }
        self.assertFalse(forbidden & set(DYNAMIC_OBS_FIELDS))
        self.assertEqual(DYNAMIC_OBS_FIELDS[-2:], ("adversary_route_remaining", "sut_route_remaining"))

    def test_heuristic_action_tapers_with_current_gap_relative_to_initial_gap(self) -> None:
        policy = longitudinal_policy(
            "heuristic", case_id="case", initial_arrival_gap_s=2.0,
            observation_time_scale_s=10.0, initial_gap_normalizer_s=4.0,
        )
        observation = np.zeros(DYNAMIC_OBSERVATION_DIM, dtype=np.float32)
        observation[16] = 0.2
        self.assertAlmostEqual(float(policy(0, observation)[1]), 1.0)
        observation[16] = 0.1
        self.assertAlmostEqual(float(policy(1, observation)[1]), 0.5)
        observation[16] = -0.05
        self.assertAlmostEqual(float(policy(2, observation)[1]), -0.25)

    def test_joint_critical_condition_requires_every_threshold(self) -> None:
        thresholds = {
            "arrival_gap_threshold_s": 1.0,
            "joint_conflict_distance_threshold_m": 10.0,
            "pair_distance_threshold_m": 8.0,
        }
        arrival = {
            "adversary_time_s": 2.0, "sut_time_s": 2.5,
            "adversary_signed_distance_m": 5.0, "sut_signed_distance_m": 7.0,
        }
        valid = critical_measurements(arrival, pair_distance_m=6.0, ttc_s=1.0, closing_speed_mps=2.0, thresholds=thresholds)
        self.assertTrue(valid["spatiotemporal_near_miss_candidate"])
        for key, value in (
            ("arrival_gap_threshold_s", 0.4),
            ("joint_conflict_distance_threshold_m", 6.0),
            ("pair_distance_threshold_m", 5.0),
        ):
            changed = dict(thresholds); changed[key] = value
            result = critical_measurements(arrival, pair_distance_m=6.0, ttc_s=1.0, closing_speed_mps=2.0, thresholds=changed)
            self.assertFalse(result["spatiotemporal_near_miss_candidate"])

    def test_signed_arrival_time_prevents_post_conflict_zero_gap_alias(self) -> None:
        arrival = {
            "adversary_time_s": 0.0, "sut_time_s": 0.0,
            "adversary_signed_time_s": -0.2, "sut_signed_time_s": -1.0,
            "adversary_signed_distance_m": -2.0, "sut_signed_distance_m": -8.0,
        }
        result = critical_measurements(
            arrival, pair_distance_m=4.0, ttc_s=1.0, closing_speed_mps=2.0,
            thresholds={
                "arrival_gap_threshold_s": 0.5,
                "joint_conflict_distance_threshold_m": 10.0,
                "pair_distance_threshold_m": 8.0,
            },
        )
        self.assertAlmostEqual(result["arrival_gap_abs_s"], 0.8)
        self.assertFalse(result["spatiotemporal_near_miss_candidate"])

    def test_route_diagnostic_has_no_success_bonus_and_collision_is_separate(self) -> None:
        cfg = self.cfg["reward"]
        route_only = compute_reward(
            10.0, 100.0, np.zeros(2), np.zeros(2),
            {"route_conflict_proximity": True, "valid_critical_near_miss": False}, cfg,
        )
        self.assertEqual(route_only.target_collision, 0.0)
        self.assertEqual(route_only.valid_critical, 0.0)
        near = compute_reward(
            10.0, 100.0, np.zeros(2), np.zeros(2),
            {"valid_critical_near_miss": True, "target_collision": False}, cfg,
        )
        collision = compute_reward(
            10.0, 100.0, np.zeros(2), np.zeros(2),
            {"valid_critical_near_miss": False, "target_collision": True}, cfg,
        )
        self.assertEqual(near.valid_critical, cfg["valid_critical_bonus"])
        self.assertEqual(near.target_collision, 0.0)
        self.assertEqual(collision.target_collision, cfg["target_collision_bonus"])
        self.assertEqual(collision.valid_critical, 0.0)

    def test_v2_metrics_reject_collision_near_miss_overlap(self) -> None:
        metrics = EpisodeMetrics("task", "case", metric_schema=CRITICAL_METRIC_SCHEMA)
        with self.assertRaises(ValueError):
            metrics.update(0.0, 1.0, 1.0, {
                "target_collision": True, "valid_critical_near_miss": True,
            }, "pairwise")
        diagnostic = EpisodeMetrics("task", "case", metric_schema=CRITICAL_METRIC_SCHEMA)
        diagnostic.update(0.0, 1.0, 1.0, {"route_conflict_proximity": True}, "none")
        self.assertFalse(diagnostic.record(1.5)["valid_critical_strict"])

    def test_calibration_is_deterministic_and_declares_validation_only(self) -> None:
        def row(policy: str, gap: float, collision: bool = False):
            return {
                "policy": policy, "collision": collision, "invalid": False,
                "trace": [{
                    "arrival_gap_abs_s": gap, "joint_conflict_distance_m": gap,
                    "pair_distance_m": gap, "physical_target_contact": collision,
                }],
            }
        rollouts = []
        for policy, gaps in (
            ("zero", [9.0] * 10), ("random", [8.0] * 10),
            ("heuristic", [0.2, 0.3, 0.4, 0.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
        ):
            rollouts.extend(row(policy, gap) for gap in gaps)
        first = calibrate_thresholds(rollouts)
        second = calibrate_thresholds(copy.deepcopy(rollouts))
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "pass")
        self.assertFalse(first["uses_test_or_ood"])
        self.assertEqual(first["uses_splits"], ["meta_validation"])

    def test_case_spawn_solver_realizes_target_with_mock_geometry(self) -> None:
        task = type("Task", (), {"spawn_regions": {"adversary": [0.0, 50.0]}})()
        base = {
            "case_id": "task_train_pool_000", "case_seed": 1,
            "adversary_initial_speed_mps": 10.0, "sut_initial_speed_mps": 10.0,
            "sut_spawn_m": 5.0,
        }
        def measure(case):
            adv_distance = 60.0 - float(case["adversary_spawn_m"])
            return {
                "adversary_time_s": adv_distance / 10.0, "sut_time_s": 3.0,
                "adversary_distance_m": adv_distance, "sut_distance_m": 30.0,
                "adversary_signed_distance_m": adv_distance, "sut_signed_distance_m": 30.0,
                "initial_pair_distance_m": 20.0, "initial_relative_speed_mps": 0.0,
            }
        case, measured = solve_adversary_spawn(task, base, 2.0, measure)
        self.assertAlmostEqual(case["adversary_spawn_m"], 10.0)
        self.assertAlmostEqual(float(measured["adversary_time_s"]) - float(measured["sut_time_s"]), 2.0)

    def test_wrong_evidence_inference_uses_target_task_prior_and_action_audit_is_read_only(self) -> None:
        cfg = self.cfg
        agent = PEARLAgent(24, 2, cfg, torch.device("cpu"))
        tasks = build_taskbook(cfg)
        task_a, task_b = tasks["meta_train"][:2]
        transition = Transition(
            np.zeros(24, np.float32), np.zeros(2, np.float32), 0.0, np.zeros(24, np.float32),
            False, True, "horizon", task_a.task_id, "episode", "case", "prior_support", 0,
        )
        original = agent._scenario_prior
        captured = []
        def wrapped(prior_tasks, count):
            captured.append(prior_tasks)
            return original(prior_tasks, count)
        with patch.object(agent, "_scenario_prior", side_effect=wrapped):
            agent.infer_posterior([[[transition]]], [task_b])
        self.assertIs(captured[-1][0], task_b)
        before = agent.parameter_hash()
        actions = _actor_means(agent, np.zeros((3, 24), np.float32), torch.zeros((1, agent.latent_dim)))
        self.assertEqual(actions.shape, (3, 2))
        self.assertEqual(before, agent.parameter_hash())


if __name__ == "__main__":
    unittest.main()
