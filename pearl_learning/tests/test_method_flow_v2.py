from __future__ import annotations

import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pearl_learning.src.benchmark_calibration import apply_calibration_manifest, calibrate_thresholds, longitudinal_policy
from pearl_learning.src.casebook_v2 import solve_adversary_spawn
from pearl_learning.src.causal_audit import _actor_means
from pearl_learning.src.critical import (
    CRITICAL_METRIC_SCHEMA,
    LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA,
    collision_risk_barrier,
    conflict_entry_order_satisfied,
    critical_measurements,
    strict_near_miss_potential,
)
from pearl_learning.src.io import (
    assert_method_variant_contract,
    prepare_run_manifest,
    read_config,
)
from pearl_learning.src.mechanism_audit import (
    SCRIPTED_POLICIES,
    policy_conflict_report,
    single_task_sac_transfer_report,
    scripted_longitudinal_action,
)
from pearl_learning.src.mechanism_casebook import (
    MATCHED_PHYSICAL_FIELDS,
    matched_case_seed,
    matched_conditions,
    validate_matched_mechanism_cases,
)
from pearl_learning.src.metrics import EpisodeMetrics
from pearl_learning.src.observation import (
    DYNAMIC_OBSERVATION_DIM,
    DYNAMIC_OBSERVATION_SCHEMA,
    DYNAMIC_OBS_FIELDS,
)
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.replay import Transition
from pearl_learning.src.reward import compute_reward
from pearl_learning.src.task_env import interpolated_conflict_entry_role
from pearl_learning.src.taskbook import build_taskbook
from pearl_learning.scripts.build_logical_order_mechanism_taskbook import derive_logical_order_taskbook


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

    def test_collision_free_metric_can_penalize_target_collision_without_changing_legacy_reward(self) -> None:
        cfg = {**self.cfg["reward"], "target_collision_penalty": 200.0}
        reward = compute_reward(
            10.0, 100.0, np.zeros(2), np.zeros(2),
            {"target_collision": True, "target_collision_disqualifying": True}, cfg,
        )
        self.assertEqual(reward.target_collision, -200.0)

    def test_strict_near_miss_potential_requires_order_and_all_three_components(self) -> None:
        thresholds = {
            "arrival_gap_threshold_s": 1.0,
            "joint_conflict_distance_threshold_m": 2.0,
            "pair_distance_threshold_m": 3.0,
        }
        close = {"arrival_gap_abs_s": 0.1, "joint_conflict_distance_m": 0.2, "pair_distance_m": 0.3}
        far = {"arrival_gap_abs_s": 0.1, "joint_conflict_distance_m": 20.0, "pair_distance_m": 0.3}
        self.assertGreater(strict_near_miss_potential(close, thresholds, prospective_order_satisfied=True), 0.5)
        self.assertLess(strict_near_miss_potential(far, thresholds, prospective_order_satisfied=True), 0.5)
        self.assertEqual(strict_near_miss_potential(close, thresholds, prospective_order_satisfied=False), 0.0)

    def test_strict_near_miss_potential_is_shaping_not_a_terminal_bonus(self) -> None:
        cfg = {**self.cfg["reward"], "strict_near_miss_potential_weight": 5.0}
        reward = compute_reward(
            10.0, 100.0, np.zeros(2), np.zeros(2),
            {"valid_critical_near_miss": False}, cfg,
            {"strict_near_miss_potential": 0.5},
        )
        self.assertEqual(reward.strict_near_miss_potential, 2.5)
        self.assertEqual(reward.valid_critical, 0.0)

    def test_collision_risk_barrier_penalizes_only_inner_calibrated_band(self) -> None:
        thresholds = {"pair_distance_threshold_m": 10.0}
        outer = collision_risk_barrier(
            {"pair_distance_m": 9.0}, thresholds, safe_pair_distance_ratio=0.85,
        )
        inner = collision_risk_barrier(
            {"pair_distance_m": 4.25}, thresholds, safe_pair_distance_ratio=0.85,
        )
        self.assertEqual(outer, 0.0)
        self.assertAlmostEqual(inner, 0.5)
        cfg = {**self.cfg["reward"], "collision_risk_barrier_weight": 40.0}
        reward = compute_reward(
            10.0, 100.0, np.zeros(2), np.zeros(2), {}, cfg,
            {"collision_risk_barrier": inner},
        )
        self.assertEqual(reward.collision_risk_barrier, -20.0)

    def test_v2_metrics_reject_collision_near_miss_overlap(self) -> None:
        metrics = EpisodeMetrics("task", "case", metric_schema=CRITICAL_METRIC_SCHEMA)
        with self.assertRaises(ValueError):
            metrics.update(0.0, 1.0, 1.0, {
                "target_collision": True, "valid_critical_near_miss": True,
            }, "pairwise")
        diagnostic = EpisodeMetrics("task", "case", metric_schema=CRITICAL_METRIC_SCHEMA)
        diagnostic.update(0.0, 1.0, 1.0, {"route_conflict_proximity": True}, "none")
        self.assertFalse(diagnostic.record(1.5)["valid_critical_strict"])

    def test_v3_entry_order_is_a_required_semantic_condition(self) -> None:
        self.assertTrue(conflict_entry_order_satisfied("any", None))
        self.assertTrue(conflict_entry_order_satisfied("adversary_first", "adversary"))
        self.assertFalse(conflict_entry_order_satisfied("adversary_first", "sut"))
        self.assertFalse(conflict_entry_order_satisfied("sut_first", None))
        with self.assertRaises(ValueError):
            conflict_entry_order_satisfied("ambiguous", "adversary")

    def test_v3_entry_order_interpolates_within_a_simulator_step(self) -> None:
        # Adversary is farther downstream after this step, but the SUT crossed
        # its entrance earlier; using post-step distance alone would be wrong.
        self.assertEqual(
            interpolated_conflict_entry_role(
                {"adversary": -0.9, "sut": -0.1}, {"adversary": 0.9, "sut": 0.2},
            ),
            "sut",
        )
        self.assertIsNone(interpolated_conflict_entry_role(
            {"adversary": -0.5, "sut": -0.5}, {"adversary": 0.5, "sut": 0.5},
        ))

    def test_v3_reuses_only_v2_validation_thresholds_with_explicit_provenance(self) -> None:
        manifest = calibrate_thresholds([
            {"policy": policy, "collision": False, "invalid": False, "trace": [{
                "arrival_gap_abs_s": gap, "joint_conflict_distance_m": gap,
                "pair_distance_m": gap, "physical_target_contact": False,
            }]}
            for policy, gaps in (("zero", [9.0] * 10), ("random", [8.0] * 10),
                                 ("heuristic", [0.2, 0.3, 0.4, 0.5] + [5.0] * 6))
            for gap in gaps
        ])
        resolved = apply_calibration_manifest({"critical_metric": {
            "schema": LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA,
        }}, manifest)
        self.assertEqual(resolved["critical_metric"]["schema"], LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA)
        self.assertEqual(resolved["critical_metric"]["threshold_source_metric_schema"], CRITICAL_METRIC_SCHEMA)

    def test_logical_order_pair_is_physically_identical_but_rule_distinct(self) -> None:
        parent = build_taskbook(self.cfg)
        parent_task = next(task for task in parent["meta_train"] if task.geometry_id == "lane_drop_24")
        derived, variants = derive_logical_order_taskbook(parent, "lane_drop_24")
        self.assertEqual(len(variants), 2)
        self.assertEqual(len(derived["meta_train"]), len(parent["meta_train"]) + 1)
        self.assertEqual({task.priority_spec["target_contact_entry_order"] for task in variants}, {"adversary_first", "sut_first"})
        for task in variants:
            self.assertEqual(task.map_hash, parent_task.map_hash)
            self.assertEqual(task.adversary_route_hash, parent_task.adversary_route_hash)
            self.assertEqual(task.sut_route_hash, parent_task.sut_route_hash)
            self.assertEqual(task.conflict_hash, parent_task.conflict_hash)

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

    def test_mechanism_case_conditions_are_absolute_and_do_not_use_calibration(self) -> None:
        conditions = matched_conditions(8)
        self.assertEqual(len(conditions), 8)
        self.assertEqual({row["target_initial_arrival_gap_s"] for row in conditions}, {-0.8, -0.4, 0.0, 0.4, 0.8})
        self.assertTrue(all("calibration" not in key for row in conditions for key in row))
        self.assertTrue(all("matched_condition_id" in row for row in conditions))

    def test_order_boundary_profile_is_near_simultaneous_without_calibration(self) -> None:
        conditions = matched_conditions(8, profile="order_boundary")
        self.assertEqual(len(conditions), 8)
        self.assertTrue(all(abs(float(row["target_initial_arrival_gap_s"])) <= 0.10 for row in conditions))
        self.assertTrue(all(float(row["target_initial_relative_speed_mps"]) == 0.0 for row in conditions))
        self.assertTrue(all(row["mechanism_case_profile"] == "order_boundary" for row in conditions))

    def test_screened_order_boundary_profile_preserves_source_condition_provenance(self) -> None:
        conditions = matched_conditions(6, profile="order_boundary_screened_v1")
        self.assertEqual(
            [row["matched_condition_id"] for row in conditions],
            [f"mechanism_grid_{index:02d}" for index in range(2, 8)],
        )
        self.assertTrue(all(row["mechanism_case_profile"] == "order_boundary_screened_v1" for row in conditions))

    def test_mechanism_matched_cases_reuse_exogenous_seed_and_require_equal_physics(self) -> None:
        condition = "mechanism_grid_00"
        row = {
            "matched_condition_id": condition,
            "case_seed": matched_case_seed(condition),
            "sut_spawn_m": 2.0,
            "adversary_spawn_m": 3.0,
            "sut_initial_speed_mps": 12.0,
            "adversary_initial_speed_mps": 14.0,
            "actual_initial_arrival_gap_s": -0.8,
            "initial_relative_speed_mps": 2.0,
            "adversary_initial_conflict_distance_m": 50.0,
            "sut_initial_conflict_distance_m": 48.0,
        }
        self.assertEqual(set(MATCHED_PHYSICAL_FIELDS), set(row) - {"matched_condition_id"})
        validate_matched_mechanism_cases({"logical_a": [row], "logical_b": [dict(row)]})
        mismatched = dict(row); mismatched["case_seed"] += 1
        with self.assertRaisesRegex(ValueError, "case_seed"):
            validate_matched_mechanism_cases({"logical_a": [row], "logical_b": [mismatched]})

    def test_policy_conflict_requires_observed_strict_objective(self) -> None:
        task_ids = ["task_a", "task_b"]
        rows = []
        for task_id in task_ids:
            for policy in SCRIPTED_POLICIES:
                rows.append({
                    "task_id": task_id, "policy": policy, "matched_condition_id": "c0",
                    "mean_longitudinal_action": 0.5 if policy == "P1_moderate_accelerate" else 0.0,
                    "record": {
                        "valid_critical_strict": False, "target_collision": False, "invalid": False,
                        "episode_return": 1.0 if (task_id == "task_a" and policy == "P0_coast") or (task_id == "task_b" and policy == "P1_moderate_accelerate") else 0.0,
                        "min_ttc": 1.0,
                    },
                })
        _, _, gate = policy_conflict_report(rows, task_ids)
        self.assertFalse(gate["objective_evidence"]["strict_objective_observed"])
        self.assertEqual(gate["status"], "fail")

    def test_order_feedback_probes_take_opposite_actions_at_the_same_dynamic_state(self) -> None:
        observation = np.zeros(DYNAMIC_OBSERVATION_DIM, dtype=np.float32)
        self.assertGreater(
            float(scripted_longitudinal_action("P7_adversary_first_feedback", 0, observation, 180)[0]),
            0.0,
        )
        self.assertLess(
            float(scripted_longitudinal_action("P8_sut_first_feedback", 0, observation, 180)[0]),
            0.0,
        )
        self.assertLess(
            float(scripted_longitudinal_action("P7_adversary_first_feedback", 0, observation, 180, action_mode="target_arrival_gap")[0]),
            0.0,
        )
        self.assertGreater(
            float(scripted_longitudinal_action("P8_sut_first_feedback", 0, observation, 180, action_mode="target_arrival_gap")[0]),
            0.0,
        )

    def test_single_task_sac_gate_requires_a_diagonal_advantage_for_both_tasks(self) -> None:
        matrix = {
            "a": {
                "a": {"episodes": 8, "valid_critical_strict_rate": 0.50},
                "b": {"episodes": 8, "valid_critical_strict_rate": 0.00},
            },
            "b": {
                "a": {"episodes": 8, "valid_critical_strict_rate": 0.00},
                "b": {"episodes": 8, "valid_critical_strict_rate": 0.25},
            },
        }
        self.assertEqual(single_task_sac_transfer_report(matrix, ["a", "b"])["status"], "pass")
        matrix["b"]["b"]["valid_critical_strict_rate"] = 0.0
        self.assertEqual(single_task_sac_transfer_report(matrix, ["a", "b"])["status"], "fail")

    def test_variant_assertions_and_manifest_resume_guard(self) -> None:
        vanilla = read_config("pearl_learning/configs/merge_method_flow_vanilla_pilot.yaml")
        structure = read_config("pearl_learning/configs/merge_method_flow_pilot.yaml")
        self.assertEqual(assert_method_variant_contract(vanilla, "vanilla", "smoke"), "vanilla")
        self.assertEqual(assert_method_variant_contract(structure, "structure", "smoke"), "structure_aware")
        broken = copy.deepcopy(vanilla); broken["scenario_prior"]["mode"] = "task_conditioned"
        with self.assertRaises(ValueError):
            assert_method_variant_contract(broken, "vanilla", "smoke")
        manifest = {
            "resolved_config_sha256": "cfg", "taskbook_hash": "tasks",
            "casebook_hashes": {"task": "cases"}, "critical_threshold_hash": "threshold",
        }
        with tempfile.TemporaryDirectory() as temp:
            prepare_run_manifest(temp, manifest, resume=False)
            with self.assertRaises(FileExistsError):
                prepare_run_manifest(temp, manifest, resume=False)
            prepare_run_manifest(temp, manifest, resume=True)
            changed = {**manifest, "taskbook_hash": "other"}
            with self.assertRaises(ValueError):
                prepare_run_manifest(temp, changed, resume=True)


if __name__ == "__main__":
    unittest.main()
