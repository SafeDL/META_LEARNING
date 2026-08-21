from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pearl_learning.src.benchmark_calibration import (
    apply_calibration_manifest,
    calibrate_thresholds,
    longitudinal_policy,
    resolve_calibration,
)
from pearl_learning.src.casebook_v2 import solve_adversary_spawn
from pearl_learning.src.causal_audit import (
    _actor_means,
    _transition_context_row,
    context_rows,
    logistic_probe_accuracy,
    posterior_separation,
    sample_context_scheme,
    stage_b_actor_critic_diagnostics,
)
from pearl_learning.src.checkpoint import save_checkpoint
from pearl_learning.scripts.audit_gate3_vanilla_pearl_mechanism import (
    GATE3_DECISION_SHOT,
    gate3_causal_chain_verdict,
    gate3_causal_chain_verdict_v4,
    gate3_causal_chain_verdict_v5,
    policy_separation_ratio,
    verify_casebook_split_provenance,
)
from pearl_learning.scripts.audit_gate3_critic_replay_signal import transition_signal_summary
from pearl_learning.scripts.recompute_gate3_verdict_v4 import recompute as recompute_gate3_v4
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
    content_hash,
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
    matched_split_conditions,
    validate_matched_mechanism_cases,
    validate_mechanism_split_disjointness,
)
from pearl_learning.src.metrics import EpisodeMetrics
from pearl_learning.src.observation import (
    DYNAMIC_OBSERVATION_DIM,
    DYNAMIC_OBSERVATION_SCHEMA,
    DYNAMIC_OBS_FIELDS,
)
from pearl_learning.src.networks import LatentFiLMCritic, LatentGammaOnlyFiLMCritic
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.replay import Transition, select_context_rows
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

    @staticmethod
    def _valid_manifest() -> dict:
        return calibrate_thresholds([
            {"policy": policy, "collision": False, "invalid": False, "trace": [{
                "arrival_gap_abs_s": gap, "joint_conflict_distance_m": gap,
                "pair_distance_m": gap, "physical_target_contact": False,
            }]}
            for policy, gaps in (("zero", [9.0] * 10), ("random", [8.0] * 10),
                                 ("heuristic", [0.2, 0.3, 0.4, 0.5] + [5.0] * 6))
            for gap in gaps
        ])

    def test_strict_calibration_entry_requires_manifest_for_v2_and_v3(self) -> None:
        for schema in (CRITICAL_METRIC_SCHEMA, LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA):
            cfg = {"critical_metric": {"schema": schema}}
            with self.assertRaisesRegex(ValueError, "requires --critical-thresholds"):
                resolve_calibration(cfg, None)
        # A legacy or absent schema keeps the previous manifest-free behavior.
        legacy = resolve_calibration({"critical_metric": {"schema": "legacy_logical_merge_critical"}}, None)
        self.assertEqual(legacy["critical_metric"]["schema"], "legacy_logical_merge_critical")
        absent = resolve_calibration({}, None)
        self.assertEqual(absent, {})

    def test_strict_calibration_entry_applies_the_same_manifest_to_v2_and_v3(self) -> None:
        manifest = self._valid_manifest()
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "critical_thresholds.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            for schema in (CRITICAL_METRIC_SCHEMA, LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA):
                resolved = resolve_calibration({"critical_metric": {"schema": schema}}, manifest_path)
                self.assertEqual(resolved["critical_metric"]["schema"], schema)
                self.assertEqual(resolved["critical_metric"]["calibration_hash"], manifest["calibration_hash"])
                self.assertEqual(
                    resolved["critical_metric"]["arrival_gap_threshold_s"],
                    manifest["thresholds"]["arrival_gap_threshold_s"],
                )
            self.assertEqual(
                resolve_calibration({"critical_metric": {"schema": LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA}}, manifest_path)
                ["critical_metric"]["threshold_source_metric_schema"],
                CRITICAL_METRIC_SCHEMA,
            )

    def test_strict_calibration_entry_rejects_a_tampered_manifest(self) -> None:
        manifest = self._valid_manifest()
        tampered = copy.deepcopy(manifest)
        tampered["thresholds"]["arrival_gap_threshold_s"] += 0.5
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "critical_thresholds.json"
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resolve_calibration({"critical_metric": {"schema": CRITICAL_METRIC_SCHEMA}}, manifest_path)
            # A failed or schema-mismatched manifest is rejected too.
            manifest_path.write_text(json.dumps({"schema": "merge_benchmark_calibration_v1", "status": "fail"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_calibration({"critical_metric": {"schema": LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA}}, manifest_path)

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

    def test_stage_b_diagnostics_report_raw_actions_and_q_grid_without_gradients(self) -> None:
        cfg = self.cfg
        torch.manual_seed(0)
        agent = PEARLAgent(24, 2, cfg, torch.device("cpu"))
        state_bank = np.zeros((2, 24), dtype=np.float32)
        z_correct = torch.ones((1, agent.latent_dim))
        z_wrong = torch.ones((1, agent.latent_dim)) * 0.5
        before = agent.parameter_hash()
        result = stage_b_actor_critic_diagnostics(agent, state_bank, z_correct, z_wrong)
        self.assertEqual(before, agent.parameter_hash())
        geometry = result["latent_geometry"]
        self.assertAlmostEqual(geometry["correct_wrong_l2"], geometry["wrong_norm_l2"])
        self.assertAlmostEqual(geometry["correct_wrong_cosine"], 1.0, places=5)
        # Collinear vectors: separation ratio = 0.5 / 0.75 = 2/3.
        self.assertAlmostEqual(geometry["separation_ratio"], 2.0 / 3.0, places=4)
        pre_tanh = result["actor_pre_tanh"]
        for key in ("raw_mean_l2", "tanh_action_l2", "symmetric_kl_pre_squash"):
            self.assertTrue(np.isfinite(float(pre_tanh[key])))
        self.assertEqual(len(pre_tanh["raw_mean_correct"]), 2)
        interpolation = result["latent_interpolation"]
        self.assertEqual(interpolation["alphas"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(len(interpolation["per_state_actions"]["alpha_0.00"]), 2)
        grid = result["critic_q_grid"]
        self.assertEqual(grid["action_grid_points"], 41)
        self.assertTrue(np.isfinite(grid["argmax_action_distance_mean"]))
        self.assertIn("correct", grid["actor_regret_mean"])

    @staticmethod
    def _synthetic_suite(
        latent_l2: float,
        action_l2: float,
        vcsr: dict,
        separation: float = 0.15,
        critic_distance: float = 0.0,
    ) -> dict:
        def row(vcsr_correct: float, vcsr_wrong: float) -> dict:
            return {
                "latent_l2": {"correct_wrong_l2": latent_l2},
                "action_adaptation": {"correct_wrong": {"action_l2": {"mean": action_l2}}},
                "trajectory_summaries": {
                    "correct": {"valid_critical_strict_rate": vcsr_correct},
                    "wrong": {"valid_critical_strict_rate": vcsr_wrong},
                },
                "paired_gain_means": {"correct_minus_wrong_return": vcsr_correct - vcsr_wrong},
                "stage_b_diagnostics": {
                    "latent_geometry": {"separation_ratio": separation, "correct_wrong_cosine": 0.997},
                    "critic_q_grid": {
                        "argmax_action_distance_mean": critic_distance, "actor_regret_mean": {"correct": 0.0},
                    },
                },
            }
        return {
            "tasks": {
                "task_a": {"shots": {str(GATE3_DECISION_SHOT): row(vcsr["a"][0], vcsr["a"][1])}},
                "task_b": {"shots": {str(GATE3_DECISION_SHOT): row(vcsr["b"][0], vcsr["b"][1])}},
            },
        }

    def test_gate3_verdict_gates_sequentially_at_the_decision_shot(self) -> None:
        oracle_pass = {
            "tasks": {"task_a": {"single_task_action_l2_mean": 0.5},
                      "task_b": {"single_task_action_l2_mean": 0.5}},
            "feasibility": {"status": "pass"},
        }
        oracle_fail = {
            "tasks": oracle_pass["tasks"],
            "feasibility": {"status": "fail"},
        }
        # Stage A fails on the absolute L2 floor: B and C are blocked.
        gate = gate3_causal_chain_verdict(self._synthetic_suite(0.1, 5.0, {"a": (1.0, 0.0), "b": (1.0, 0.0)}), oracle_pass)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["stages"]["stage_b_posterior_to_action"]["status"], "blocked_by_stage_a")
        self.assertEqual(gate["stages"]["stage_c_action_to_outcome"]["status"], "blocked_by_stage_a")
        # v3 re-interpretation of Round 1/2: L2 clears 0.5 but R_sep stays
        # below 0.25, so the posterior is not task-discriminative and Stage A
        # fails before any action-level judgement.
        gate = gate3_causal_chain_verdict(self._synthetic_suite(7.7, 0.03, {"a": (0.0, 0.0), "b": (0.0, 0.0)}), oracle_pass)
        self.assertEqual(gate["schema"], "gate3_vanilla_pearl_causal_chain_gate_v3")
        self.assertEqual(gate["stages"]["stage_a_context_to_posterior"]["status"], "fail")
        self.assertEqual(gate["stages"]["stage_b_posterior_to_action"]["status"], "blocked_by_stage_a")
        self.assertEqual(gate["stages"]["stage_c_action_to_outcome"]["status"], "blocked_by_stage_a")
        # Stage A passes (L2 + R_sep) but B fails: C is blocked by B.
        gate = gate3_causal_chain_verdict(
            self._synthetic_suite(7.7, 0.03, {"a": (0.0, 0.0), "b": (0.0, 0.0)}, separation=0.4), oracle_pass
        )
        self.assertEqual(gate["stages"]["stage_a_context_to_posterior"]["status"], "pass")
        self.assertEqual(gate["stages"]["stage_b_posterior_to_action"]["status"], "fail")
        self.assertEqual(gate["stages"]["stage_c_action_to_outcome"]["status"], "blocked_by_stage_b")
        # Stage B passes but query feasibility fails: C is blocked by the oracle.
        gate = gate3_causal_chain_verdict(
            self._synthetic_suite(7.7, 0.3, {"a": (1.0, 0.0), "b": (1.0, 0.0)}, separation=0.4), oracle_fail
        )
        self.assertEqual(gate["stages"]["stage_b_posterior_to_action"]["status"], "pass")
        self.assertEqual(gate["stages"]["stage_c_action_to_outcome"]["status"], "blocked_by_query_feasibility")
        # Full chain with feasible queries and a strict VCSR advantage passes.
        gate = gate3_causal_chain_verdict(
            self._synthetic_suite(7.7, 0.3, {"a": (0.5, 0.0), "b": (0.0, 0.0)}, separation=0.4), oracle_pass
        )
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["passed_stages"], ["stage_a", "stage_b", "stage_c"])
        self.assertIn("policy_separation_ratio", gate)
        # Stage A requires BOTH tasks to clear the R_sep floor.
        gate = gate3_causal_chain_verdict(
            self._synthetic_suite(7.7, 0.3, {"a": (0.5, 0.0), "b": (0.0, 0.0)}), oracle_pass
        )
        self.assertEqual(gate["stages"]["stage_a_context_to_posterior"]["status"], "fail")
        # The decision shot must be present in the suite.
        missing = {"tasks": {"task_a": {"shots": {"1": {}}}}}
        with self.assertRaises(ValueError):
            gate3_causal_chain_verdict(missing)

    def test_gate3_v4_verdict_separates_critic_and_actor_sequentially(self) -> None:
        oracle_pass = {
            "tasks": {"task_a": {"single_task_action_l2_mean": 0.5},
                      "task_b": {"single_task_action_l2_mean": 0.5}},
            "feasibility": {"status": "pass"},
        }
        oracle_fail = {"tasks": oracle_pass["tasks"], "feasibility": {"status": "fail"}}

        # A fail blocks every downstream link, regardless of Critic/Actor values.
        gate = gate3_causal_chain_verdict_v4(
            self._synthetic_suite(0.1, 0.5, {"a": (1.0, 0.0), "b": (1.0, 0.0)}, separation=0.4, critic_distance=0.5),
            oracle_pass,
        )
        stages = gate["stages"]
        self.assertEqual(stages["stage_b_q_posterior_to_critic_action_preference"]["status"], "blocked_by_stage_a")
        self.assertEqual(stages["stage_b_pi_critic_to_actor_action"]["status"], "blocked_by_stage_a")
        self.assertEqual(stages["stage_c_actor_to_outcome"]["status"], "blocked_by_stage_a")

        # A pass but B_Q below 0.10 blocks Actor and outcome judgement.
        gate = gate3_causal_chain_verdict_v4(
            self._synthetic_suite(7.7, 0.5, {"a": (1.0, 0.0), "b": (1.0, 0.0)}, separation=0.4, critic_distance=0.05),
            oracle_pass,
        )
        stages = gate["stages"]
        self.assertEqual(stages["stage_a_context_to_posterior"]["status"], "pass")
        self.assertEqual(stages["stage_b_q_posterior_to_critic_action_preference"]["status"], "fail")
        self.assertEqual(stages["stage_b_pi_critic_to_actor_action"]["status"], "blocked_by_stage_b_q")
        self.assertEqual(stages["stage_c_actor_to_outcome"]["status"], "blocked_by_stage_b_q")

        # The B_Q floor is inclusive, but both tasks must independently clear it.
        gate = gate3_causal_chain_verdict_v4(
            self._synthetic_suite(7.7, 0.03, {"a": (1.0, 0.0), "b": (1.0, 0.0)}, separation=0.4, critic_distance=0.10),
            oracle_pass,
        )
        stages = gate["stages"]
        self.assertEqual(stages["stage_b_q_posterior_to_critic_action_preference"]["status"], "pass")
        self.assertEqual(stages["stage_b_pi_critic_to_actor_action"]["status"], "fail")
        self.assertEqual(stages["stage_c_actor_to_outcome"]["status"], "blocked_by_stage_b_pi")
        suite = self._synthetic_suite(7.7, 0.5, {"a": (1.0, 0.0), "b": (1.0, 0.0)}, separation=0.4, critic_distance=0.10)
        suite["tasks"]["task_b"]["shots"][str(GATE3_DECISION_SHOT)]["stage_b_diagnostics"]["critic_q_grid"]["argmax_action_distance_mean"] = 0.099
        gate = gate3_causal_chain_verdict_v4(suite, oracle_pass)
        self.assertEqual(gate["stages"]["stage_b_q_posterior_to_critic_action_preference"]["status"], "fail")

        # B_Q/B_pi pass, but an infeasible query set remains an independent blocker.
        gate = gate3_causal_chain_verdict_v4(
            self._synthetic_suite(7.7, 0.10, {"a": (1.0, 0.0), "b": (1.0, 0.0)}, separation=0.4, critic_distance=0.10),
            oracle_fail,
        )
        self.assertEqual(gate["stages"]["stage_c_actor_to_outcome"]["status"], "blocked_by_query_feasibility")

        # The full v4 chain preserves the Stage-C strict VCSR requirement.
        gate = gate3_causal_chain_verdict_v4(
            self._synthetic_suite(7.7, 0.10, {"a": (0.5, 0.0), "b": (0.0, 0.0)}, separation=0.4, critic_distance=0.10),
            oracle_pass,
        )
        self.assertEqual(gate["schema"], "gate3_vanilla_pearl_causal_chain_gate_v4")
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["passed_stages"], ["stage_a", "stage_b_q", "stage_b_pi", "stage_c"])

    def test_gate3_v5_keeps_failed_critic_as_nonblocking_diagnostic(self) -> None:
        oracle = {
            "tasks": {"task_a": {"single_task_action_l2_mean": 0.5},
                      "task_b": {"single_task_action_l2_mean": 0.5}},
            "feasibility": {"status": "pass"},
        }
        gate = gate3_causal_chain_verdict_v5(
            self._synthetic_suite(
                7.7, 0.5, {"a": (0.5, 0.0), "b": (0.0, 0.0)},
                separation=0.4, critic_distance=0.0,
            ),
            oracle,
        )
        self.assertEqual(gate["schema"], "gate3_vanilla_pearl_causal_chain_gate_v5")
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["hard_gate_chain"], ["stage_a", "stage_b_pi", "stage_c"])
        critic = gate["stages"]["stage_b_q_diagnostic_only"]
        self.assertEqual(critic["status"], "diagnostic_only")
        self.assertEqual(critic["observed_threshold_status"], "fail")
        self.assertFalse(critic["blocks_actor_or_outcome"])
        self.assertEqual(
            gate["stages"]["stage_b_pi_posterior_to_actor_action"]["status"], "pass",
        )
        self.assertEqual(gate["stages"]["stage_c_actor_to_outcome"]["status"], "pass")

    def test_gate3_context_sampling_film_critic_config_changes_only_critic(self) -> None:
        round3 = read_config("pearl_learning/configs/merge_method_flow_gate3_context_sampling.yaml")
        combined = read_config("pearl_learning/configs/merge_method_flow_gate3_context_sampling_film_critic.yaml")
        self.assertEqual(combined["project"]["output_root"], "results/pearl_learning/merge_method_flow_gate3_context_sampling_film_critic")
        self.assertEqual(combined["experiment"]["method_variant"], "vanilla_gate3_context_sampling_film_critic")
        self.assertEqual(combined["networks"]["critic_architecture"], "latent_film_dense")
        for section in (
            "environment", "reward", "pearl", "scenario_prior", "scenario_representation",
            "posterior_routed_moe", "sac", "meta_training", "method_flow_pilot", "cases",
        ):
            self.assertEqual(combined[section], round3[section])
        combined_networks = dict(combined["networks"])
        combined_networks.pop("critic_architecture")
        self.assertEqual(combined_networks, round3["networks"])

    def test_gate3_gamma_only_config_changes_only_critic(self) -> None:
        prior = read_config("pearl_learning/configs/merge_method_flow_gate3_context_sampling_film_critic.yaml")
        combined = read_config(
            "pearl_learning/configs/merge_method_flow_gate3_context_sampling_gamma_only_film_critic.yaml"
        )
        self.assertEqual(
            combined["project"]["output_root"],
            "results/pearl_learning/merge_method_flow_gate3_context_sampling_gamma_only_film_critic",
        )
        self.assertEqual(
            combined["experiment"]["method_variant"],
            "vanilla_gate3_context_sampling_gamma_only_film_critic",
        )
        self.assertEqual(combined["networks"]["critic_architecture"], "latent_film_gamma_only")
        for section in (
            "environment", "reward", "pearl", "scenario_prior", "scenario_representation",
            "posterior_routed_moe", "sac", "meta_training", "method_flow_pilot", "cases",
        ):
            self.assertEqual(combined[section], prior[section])
        combined_networks = dict(combined["networks"])
        prior_networks = dict(prior["networks"])
        combined_networks.pop("critic_architecture")
        prior_networks.pop("critic_architecture")
        self.assertEqual(combined_networks, prior_networks)

    def test_gate3_v4_recompute_writes_only_the_additive_verdict(self) -> None:
        suite = self._synthetic_suite(
            7.7, 0.10, {"a": (0.5, 0.0), "b": (0.0, 0.0)}, separation=0.4, critic_distance=0.10,
        )
        suite["schema"] = "gate3_vanilla_pearl_mechanism_causal_audit_suite_v1"
        oracle = {
            "tasks": {"task_a": {"single_task_action_l2_mean": 0.5},
                      "task_b": {"single_task_action_l2_mean": 0.5}},
            "feasibility": {"status": "pass"},
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "gate3_causal_audit.json"
            source.write_text(json.dumps(suite), encoding="utf-8")
            gate = recompute_gate3_v4(source, oracle, None)
            output = root / "gate3_causal_chain_gate_v4.json"
            self.assertTrue(output.exists())
            self.assertEqual(gate["schema"], "gate3_vanilla_pearl_causal_chain_gate_v4")
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), suite)
            self.assertEqual(gate["recomputed_from"]["environment_steps"], 0)
            self.assertEqual(gate["recomputed_from"]["training_updates"], 0)

    def test_policy_separation_ratio_normalizes_by_single_task_sac_distance(self) -> None:
        suite = self._synthetic_suite(7.7, 0.04, {"a": (0.0, 0.0), "b": (0.0, 0.0)})
        oracle = {
            "tasks": {"task_a": {"single_task_action_l2_mean": 0.8},
                      "task_b": {"single_task_action_l2_mean": 0.2}},
            "feasibility": {"status": "fail"},
        }
        ratio = policy_separation_ratio(suite, oracle)
        self.assertAlmostEqual(ratio["task_a"], 0.04 / 0.8)
        self.assertAlmostEqual(ratio["task_b"], 0.04 / 0.2)

    def test_film_critic_conditions_on_latent_and_old_checkpoints_stay_loadable(self) -> None:
        cfg = self.cfg
        torch.manual_seed(0)
        film_cfg = copy.deepcopy(cfg)
        film_cfg["networks"] = {**film_cfg["networks"], "critic_architecture": "latent_film_dense"}
        film_agent = PEARLAgent(24, 2, film_cfg, torch.device("cpu"))
        self.assertEqual(film_agent.architecture_metadata()["critic_architecture"], "latent_film_dense")
        observation = torch.zeros((3, 24))
        action = torch.zeros((3, 2))
        with torch.no_grad():
            q_zero = film_agent.q1(observation, action, torch.zeros((3, film_agent.latent_dim)))
            q_one = film_agent.q1(observation, action, torch.ones((3, film_agent.latent_dim)))
        self.assertTrue(bool((q_zero != q_one).any()))
        dense_agent = PEARLAgent(24, 2, cfg, torch.device("cpu"))
        state = dense_agent.state_dict()
        metadata = dict(state["architecture_metadata"])
        metadata.pop("critic_architecture")
        state["architecture_metadata"] = metadata
        dense_agent.load_state_dict(state)  # pre-FiLM checkpoint metadata defaults to dense
        with self.assertRaises(ValueError):
            film_agent.load_state_dict(state)

    def test_film_critic_update_logs_critic_latent_gradient_norm(self) -> None:
        cfg = self.cfg
        torch.manual_seed(0)
        film_cfg = copy.deepcopy(cfg)
        film_cfg["networks"] = {**film_cfg["networks"], "critic_architecture": "latent_film_dense"}
        agent = PEARLAgent(24, 2, film_cfg, torch.device("cpu"))
        task = next(task for task in build_taskbook(cfg)["meta_train"] if task.geometry_id == "lane_drop_24")
        transition = Transition(
            np.zeros(24, np.float32), np.zeros(2, np.float32), 0.0, np.zeros(24, np.float32),
            False, True, "horizon", task.task_id, "episode", "case", "prior_support", 0,
        )
        metrics = agent.update([[[transition] * 4]], [[transition] * 8], None, None, [0], [task])
        self.assertIn("critic_latent_gradient_norm", metrics)
        self.assertTrue(np.isfinite(float(metrics["critic_latent_gradient_norm"])))
        self.assertGreater(float(metrics["critic_latent_gradient_norm"]), 0.0)

    def test_gamma_only_film_critic_structure_zero_initialization_and_emergent_gradient(self) -> None:
        torch.manual_seed(17)
        observation_dim, action_dim, latent_dim, feature_dim = 4, 2, 3, 8
        critic = LatentGammaOnlyFiLMCritic(
            observation_dim, action_dim, latent_dim, [12, feature_dim]
        )
        historical = LatentFiLMCritic(
            observation_dim, action_dim, latent_dim, [12, feature_dim]
        )

        new_linears = [module for module in critic.modulator if isinstance(module, torch.nn.Linear)]
        old_linears = [module for module in historical.modulator if isinstance(module, torch.nn.Linear)]
        self.assertEqual(len(new_linears), len(old_linears))
        self.assertEqual(
            [layer.out_features for layer in new_linears[:-1]],
            [layer.out_features for layer in old_linears[:-1]],
        )
        self.assertEqual(new_linears[-1].in_features, old_linears[-1].in_features)
        self.assertEqual(new_linears[-1].out_features, feature_dim)
        self.assertEqual(old_linears[-1].out_features, 2 * feature_dim)
        self.assertTrue(torch.equal(new_linears[-1].weight, torch.zeros_like(new_linears[-1].weight)))
        self.assertTrue(torch.equal(new_linears[-1].bias, torch.zeros_like(new_linears[-1].bias)))
        self.assertFalse(any("beta" in key for key in critic.state_dict()))

        observation = torch.randn(16, observation_dim)
        action = torch.randn(16, action_dim)
        zero_latent = torch.zeros(16, latent_dim)
        one_latent = torch.ones(16, latent_dim)
        with torch.no_grad():
            torch.testing.assert_close(
                critic(observation, action, zero_latent),
                critic(observation, action, one_latent),
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(
                int(torch.count_nonzero(torch.tanh(critic.modulator(one_latent)))),
                0,
            )

        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)
        training_latent = torch.cat((zero_latent[:8], one_latent[:8]), dim=0)
        target = torch.linspace(-1.0, 1.0, 16).unsqueeze(-1)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(critic(observation, action, training_latent), target)
        loss.backward()
        final_gradient = new_linears[-1].weight.grad
        self.assertIsNotNone(final_gradient)
        self.assertTrue(torch.isfinite(final_gradient).all())
        self.assertGreater(float(final_gradient.norm()), 0.0)
        optimizer.step()
        self.assertGreater(float(new_linears[-1].weight.detach().norm()), 0.0)
        with torch.no_grad():
            gamma_difference = torch.tanh(critic.modulator(one_latent)) - torch.tanh(critic.modulator(zero_latent))
        self.assertGreater(float(gamma_difference.abs().max()), 0.0)
        self.assertLessEqual(float(torch.tanh(critic.modulator(one_latent)).abs().max()), 1.0)

    def test_gamma_only_film_critic_gradient_starts_after_zero_initialized_gate_opens(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["networks"] = {**cfg["networks"], "critic_architecture": "latent_film_gamma_only"}
        torch.manual_seed(19)
        agent = PEARLAgent(24, 2, cfg, torch.device("cpu"))
        task = next(task for task in build_taskbook(cfg)["meta_train"] if task.geometry_id == "lane_drop_24")
        transition = Transition(
            np.zeros(24, np.float32), np.zeros(2, np.float32), 0.0, np.zeros(24, np.float32),
            False, True, "horizon", task.task_id, "episode", "case", "prior_support", 0,
        )
        first = agent.update([[[transition] * 4]], [[transition] * 8], None, None, [0], [task])
        second = agent.update([[[transition] * 4]], [[transition] * 8], None, None, [0], [task])
        self.assertEqual(float(first["critic_latent_gradient_norm"]), 0.0)
        self.assertTrue(np.isfinite(float(second["critic_latent_gradient_norm"])))
        self.assertGreater(float(second["critic_latent_gradient_norm"]), 0.0)

    def test_gamma_only_film_critic_checkpoint_isolation(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        old_cfg = copy.deepcopy(cfg)
        old_cfg["networks"] = {**old_cfg["networks"], "critic_architecture": "latent_film_dense"}
        gamma_cfg = copy.deepcopy(cfg)
        gamma_cfg["networks"] = {**gamma_cfg["networks"], "critic_architecture": "latent_film_gamma_only"}
        old_agent = PEARLAgent(24, 2, old_cfg, torch.device("cpu"))
        gamma_agent = PEARLAgent(24, 2, gamma_cfg, torch.device("cpu"))

        # The frozen historical implementation still has its gamma+beta state shape.
        self.assertEqual(old_agent.q1.modulator[-1].out_features, 2 * old_agent.q1.feature_dim)
        self.assertEqual(gamma_agent.q1.modulator[-1].out_features, gamma_agent.q1.feature_dim)
        old_state = old_agent.state_dict()
        gamma_state = gamma_agent.state_dict()
        PEARLAgent(24, 2, old_cfg, torch.device("cpu")).load_state_dict(old_state)
        PEARLAgent(24, 2, gamma_cfg, torch.device("cpu")).load_state_dict(gamma_state)
        with self.assertRaisesRegex(ValueError, "architecture metadata"):
            gamma_agent.load_state_dict(old_state)
        with self.assertRaisesRegex(ValueError, "architecture metadata"):
            old_agent.load_state_dict(gamma_state)

    def test_update_logs_gate3_causal_chain_diagnostics_without_changing_losses(self) -> None:
        cfg = self.cfg
        torch.manual_seed(0)
        agent = PEARLAgent(24, 2, cfg, torch.device("cpu"))
        task = next(task for task in build_taskbook(cfg)["meta_train"] if task.geometry_id == "lane_drop_24")
        transition = Transition(
            np.zeros(24, np.float32), np.zeros(2, np.float32), 0.0, np.zeros(24, np.float32),
            False, True, "horizon", task.task_id, "episode", "case", "prior_support", 0,
        )
        batch = [transition] * 8
        context = [[transition] * 4]
        hashes_before = agent.module_hashes()
        metrics = agent.update([context], [batch], None, None, [0], [task])
        hashes_after = agent.module_hashes()
        for key in (
            "context_encoder_critic_gradient_norm",
            "posterior_prior_mean_l2",
            "evidence_to_prior_precision_ratio",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(float(metrics[key])))
        # A fresh encoder receives a non-zero critic gradient, and the update
        # itself changed only the modules the prescribed boundaries allow.
        self.assertGreater(float(metrics["context_encoder_critic_gradient_norm"]), 0.0)
        self.assertNotEqual(hashes_before["context_encoder"], hashes_after["context_encoder"])

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

    def test_fewshot_mechanism_profile_declares_disjoint_train_support_query(self) -> None:
        plan = matched_split_conditions("order_boundary_fewshot_v1")
        self.assertEqual({split: len(rows) for split, rows in plan.items()},
                         {"train_pool": 6, "validation_support": 4, "validation_query": 4})
        ids = [row["matched_condition_id"] for rows in plan.values() for row in rows]
        self.assertEqual(len(set(ids)), 14)
        # The train split is exactly the Gate-1-screened feasible subset.
        self.assertEqual(
            [row["matched_condition_id"] for row in plan["train_pool"]],
            [f"mechanism_grid_{index:02d}" for index in range(2, 8)],
        )
        self.assertTrue(all(row["mechanism_case_profile"] == "order_boundary_fewshot_v1"
                            for rows in plan.values() for row in rows))
        with self.assertRaises(ValueError):
            matched_split_conditions("absolute_grid")

    def test_screened_fewshot_profile_keeps_oracle_selected_query_conditions(self) -> None:
        plan = matched_split_conditions("order_boundary_fewshot_screened_v1")
        self.assertEqual(
            [row["matched_condition_id"] for row in plan["validation_query"]],
            [
                "mechanism_grid_query_candidate_04",
                "mechanism_grid_query_candidate_05",
                "mechanism_grid_query_candidate_02",
                "mechanism_grid_query_candidate_06",
            ],
        )
        self.assertEqual(
            [row["matched_condition_id"] for row in plan["train_pool"]],
            [f"mechanism_grid_{index:02d}" for index in range(2, 8)],
        )
        self.assertEqual(len(plan["validation_support"]), 4)
        ids = [row["matched_condition_id"] for rows in plan.values() for row in rows]
        self.assertEqual(len(set(ids)), 14)

    def test_mechanism_split_disjointness_accepts_matched_seeds_and_rejects_split_reuse(self) -> None:
        def row(condition: str, gap: float = -0.05):
            return {
                "matched_condition_id": condition,
                "case_seed": matched_case_seed(condition),
                "sut_spawn_m": 2.0,
                "adversary_spawn_m": 3.0,
                "sut_initial_speed_mps": 12.0,
                "adversary_initial_speed_mps": 12.0,
                "actual_initial_arrival_gap_s": gap,
                "initial_relative_speed_mps": 0.0,
                "adversary_initial_conflict_distance_m": 50.0,
                "sut_initial_conflict_distance_m": 48.0,
            }
        matched = {
            "logical_a": {
                "train_pool": [row("mechanism_grid_train_00")],
                "validation_support": [row("mechanism_grid_support_00")],
                "validation_query": [row("mechanism_grid_query_00")],
            },
            "logical_b": {
                "train_pool": [row("mechanism_grid_train_00")],
                "validation_support": [row("mechanism_grid_support_00")],
                "validation_query": [row("mechanism_grid_query_00")],
            },
        }
        validate_mechanism_split_disjointness(matched)
        leaked = copy.deepcopy(matched)
        for task_id in leaked:
            leaked[task_id]["train_pool"].append(row("mechanism_grid_query_00"))
        with self.assertRaisesRegex(ValueError, "reused across splits"):
            validate_mechanism_split_disjointness(leaked)
        mismatched = copy.deepcopy(matched)
        mismatched["logical_b"]["validation_query"] = [row("mechanism_grid_query_00", gap=0.05)]
        with self.assertRaisesRegex(ValueError, "actual_initial_arrival_gap_s"):
            validate_mechanism_split_disjointness(mismatched)
        with self.assertRaises(ValueError):
            validate_mechanism_split_disjointness({"logical_a": matched["logical_a"]})

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

    @staticmethod
    def _transition(row_id: int, *, terminated: bool = False) -> Transition:
        return Transition(
            obs=np.full(DYNAMIC_OBSERVATION_DIM, float(row_id), dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            reward=float(row_id),
            next_obs=np.full(DYNAMIC_OBSERVATION_DIM, float(row_id), dtype=np.float32),
            terminated=terminated,
            truncated=False,
            termination_reason="running",
            task_id="task",
            episode_id="episode",
            case_id="case",
            collection_mode="prior_support",
            posterior_version=0,
        )

    def test_critic_replay_signal_summary_reports_terminal_and_conflict_proxy(self) -> None:
        rows = [self._transition(index, terminated=index == 2) for index in range(3)]
        summary = transition_signal_summary(rows, {id(rows[1])})
        self.assertEqual(summary["transition_count"], 3)
        self.assertAlmostEqual(summary["terminal_transition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["conflict_near_transition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["task_sensitive_proxy_rate"], 2.0 / 3.0)
        self.assertEqual(summary["signal_strata"]["terminal"]["transition_count"], 1)
        self.assertEqual(summary["signal_strata"]["conflict_near"]["transition_count"], 1)
        self.assertEqual(summary["signal_strata"]["common"]["transition_count"], 1)

    def test_canonical_context_selector_matches_legacy_random_and_stratifies_terminal(self) -> None:
        rows = [self._transition(index) for index in range(10)]
        # The default scheme must reproduce the historical rng.choice sampler
        # bit-for-bit under identical RNG states.
        rng_legacy = np.random.default_rng(0)
        chosen = rng_legacy.choice(len(rows), size=8, replace=False)
        legacy = [rows[int(item)] for item in np.asarray(chosen).reshape(-1)]
        canonical = select_context_rows(rows, 8, "random", np.random.default_rng(0))
        self.assertEqual([row.reward for row in legacy], [row.reward for row in canonical])
        # terminal_stratified_v1 always carries the terminal transition
        # exactly once and keeps the requested block size; every returned
        # element must be a transition, never an index.
        stratified = select_context_rows(rows, 8, "terminal_stratified_v1", np.random.default_rng(1))
        self.assertEqual(len(stratified), 8)
        self.assertTrue(all(isinstance(row, Transition) for row in stratified))
        self.assertEqual(sum(1 for row in stratified if row is rows[-1]), 1)
        self.assertEqual(stratified[0], rows[-1])
        self.assertTrue({row.reward for row in stratified} <= {row.reward for row in rows})
        # A one-transition episode degrades to repetition, never drops rows.
        tiny = select_context_rows(rows[:1], 8, "terminal_stratified_v1", np.random.default_rng(2))
        self.assertEqual(len(tiny), 8)
        self.assertTrue(all(row is rows[0] for row in tiny))
        with self.assertRaises(ValueError):
            select_context_rows(rows, 8, "unsupported", np.random.default_rng(3))

    def test_sample_context_scheme_adds_conflict_window_and_delegates_canonical(self) -> None:
        rows = [self._transition(index) for index in range(10)]
        # Zero every conflict-proximity channel, then mark rows 2-4 as
        # conflict-near via |arrival_time_difference| (index 16).
        for row in rows:
            row.obs[:] = 0.0
        for index in range(10):
            rows[index].obs[16] = 0.0 if index in (2, 3, 4) else 0.9
        groups = sample_context_scheme([rows], 8, "conflict_window", np.random.default_rng(0))
        selected = groups[0]
        self.assertEqual(len(selected), 8)
        self.assertTrue(any(row is rows[-1] for row in selected))
        for index in (2, 3, 4):
            self.assertTrue(any(row is rows[index] for row in selected))
        delegated = sample_context_scheme([rows], 8, "terminal_stratified_v1", np.random.default_rng(0))
        self.assertEqual(delegated[0][0], rows[-1])

    def test_transition_context_row_matches_agent_context_tensor(self) -> None:
        cfg = self.cfg
        torch.manual_seed(0)
        agent = PEARLAgent(24, 2, cfg, torch.device("cpu"))
        rows = [self._transition(index) for index in range(4)]
        context_by_task = [[rows[:2]], [rows[2:]]]
        tensor_rows = agent.context_tensor(context_by_task).detach().cpu().numpy()
        manual = context_rows([rows], float(agent.reward_scale))
        flat = tensor_rows.reshape(-1, tensor_rows.shape[-1])
        self.assertTrue(np.array_equal(flat, manual))
        row = _transition_context_row(rows[0], float(agent.reward_scale))
        self.assertEqual(row[24], rows[0].reward / float(agent.reward_scale))

    def test_logistic_probe_recovers_separable_tasks_and_drops_noise(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.standard_normal((40, 5)).astype(np.float32)
        labels = np.repeat([0, 1], 20)
        groups = np.repeat(np.arange(8), 5)
        features[labels == 1, 0] += 4.0
        separable = logistic_probe_accuracy(features, labels, groups, seed=0)
        self.assertGreater(separable["transition_accuracy"], 0.8)
        self.assertGreater(separable["episode_majority_accuracy"], 0.8)
        noise = logistic_probe_accuracy(rng.standard_normal((40, 5)).astype(np.float32), labels, groups, seed=1)
        self.assertTrue(0.2 < noise["transition_accuracy"] < 0.8)
        with self.assertRaises(ValueError):
            logistic_probe_accuracy(features[:3], labels[:3], groups[:3], seed=0)

    def test_posterior_separation_uses_prior_relative_definition(self) -> None:
        class StubAgent:
            def prior(self, tasks=None):
                return torch.zeros(1, 5), torch.zeros(1, 5)

            def infer_posterior(self, context_by_task, tasks=None):
                marker = context_by_task[0][0][0]
                value = 2.0 if marker == "correct" else 0.5
                return torch.ones(1, 5) * value, torch.zeros(1, 5)

        separation = posterior_separation(StubAgent(), [["correct"]], [["wrong"]], task=None)
        # D_cw = 1.5 * sqrt(5); c_prior = 2 * sqrt(5); w_prior = 0.5 * sqrt(5)
        # R_sep = 1.5 / (0.5 * (2 + 0.5)) = 1.2
        self.assertAlmostEqual(separation["correct_wrong_l2"], 1.5 * np.sqrt(5), places=4)
        self.assertAlmostEqual(separation["prior_relative_separation_ratio"], 1.2, places=4)
        self.assertAlmostEqual(separation["correct_wrong_cosine"], 1.0, places=5)

    def test_split_casebook_provenance_enforcement(self) -> None:
        def book(value: str) -> dict:
            return {
                "train_pool": [{"case_id": f"t_{value}"}],
                "validation_support": [{"case_id": f"s_{value}"}],
                "validation_query": [{"case_id": f"q_{value}"}],
            }

        books = {"task_a": book("a"), "task_b": book("b")}
        matching = {"task_a": {k: content_hash(v) for k, v in books["task_a"].items()},
                    "task_b": {k: content_hash(v) for k, v in books["task_b"].items()}}
        provenance = verify_casebook_split_provenance(
            {"casebook_split_hashes": matching}, books, None, None
        )
        self.assertIn("casebook_split_hashes", provenance)
        self.assertNotIn("query_screening_manifest_hash", provenance)
        # A revised query group demands the screening manifest and a passing oracle.
        revised_query = {"task_a": {**matching["task_a"], "validation_query": "other"},
                         "task_b": matching["task_b"]}
        with self.assertRaisesRegex(ValueError, "query-screening-manifest"):
            verify_casebook_split_provenance({"casebook_split_hashes": revised_query}, books, None, None)
        manifest = {
            "schema": "gate3_query_candidate_screening_v1",
            "provenance": {"uses_test_or_ood": False},
        }
        with self.assertRaisesRegex(ValueError, "feasibility = pass"):
            verify_casebook_split_provenance(
                {"casebook_split_hashes": revised_query}, books, manifest, None
            )
        with self.assertRaisesRegex(ValueError, "feasibility = pass"):
            verify_casebook_split_provenance(
                {"casebook_split_hashes": revised_query}, books, manifest,
                {"feasibility": {"status": "fail"}},
            )
        provenance = verify_casebook_split_provenance(
            {"casebook_split_hashes": revised_query}, books, manifest,
            {"feasibility": {"status": "pass"}},
        )
        self.assertIn("query_screening_manifest_hash", provenance)
        self.assertEqual(provenance["query_revision_oracle_feasibility"], "pass")
        # train_pool / validation_support are frozen support evidence.
        changed_support = {"task_a": {**matching["task_a"], "validation_support": "other"},
                           "task_b": matching["task_b"]}
        with self.assertRaisesRegex(ValueError, "validation_support"):
            verify_casebook_split_provenance(
                {"casebook_split_hashes": changed_support}, books, manifest,
                {"feasibility": {"status": "pass"}},
            )
        # Legacy checkpoints predating split-level hashing stay loadable.
        legacy = verify_casebook_split_provenance({}, books, None, None)
        self.assertEqual(legacy["casebook_split_provenance"], "checkpoint predates split-level casebook hashing")

    def test_checkpoint_records_split_casebook_hashes(self) -> None:
        cfg = self.cfg
        torch.manual_seed(0)
        agent = PEARLAgent(24, 2, cfg, torch.device("cpu"))
        splits = {"task_a": {"train_pool": "t", "validation_support": "s", "validation_query": "q"}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best_model.pt"
            save_checkpoint(
                path, agent, cfg, "taskbook", 0,
                casebook_hashes={"task_a": "whole"},
                casebook_split_hashes=splits,
                training_seed=1,
                rng_state={"torch": torch.get_rng_state(), "numpy_generator": None},
                trainer_state=None,
            )
            manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["casebook_split_hashes"], splits)
        self.assertEqual(manifest["casebook_hashes"], {"task_a": "whole"})

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
