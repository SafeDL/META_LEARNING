from __future__ import annotations

import copy
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from pearl_learning.src.adapters.base import MetaDriveAdapterBase
from pearl_learning.src.context_encoder import ContextEncoder
from pearl_learning.src.formal_validation import REQUIRED_PRETRAIN_BASELINES, verify_formal_validation
from pearl_learning.src.posterior_adaptation_analysis import paired_method_effect, task_cluster_interval
from pearl_learning.src.collector import Rollout
from pearl_learning.src.evaluator import (
    _fixed_episode_context_block,
    _mask_context_fields,
    _posterior_action_disagreement,
    _posterior_context,
    compact_fewshot_result,
    evaluation_regime,
    infer_support_posteriors,
)
from pearl_learning.src.io import read_config
from pearl_learning.src.io import content_hash, write_json
from pearl_learning.src.metrics import EpisodeMetrics, summarize
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.pearl_trainer import _sample_tasks_without_replacement
from pearl_learning.src.replay import TaskReplayBuffer, TaskReplayBuffers, Transition
from pearl_learning.src.reward import (
    compute_reward,
    required_invalid_event_penalty,
    validate_reward_contract,
)
from pearl_learning.src.observation import OBS_FIELDS, build_observation
from pearl_learning.src.routes import RoutePolyline, wrap_to_pi
from pearl_learning.src.task_spec import LogicalScenarioTaskSpec
from pearl_learning.src.task_representation import (
    INTERACTION_OBSERVATION_FIELDS,
    INTERACTION_OBSERVATION_INDEXES,
    configure_disentangled_representation,
    representation_target,
)
from pearl_learning.src.task_env import compose_route_tracking_action, target_contact_matches_rule
from pearl_learning.src.taskbook import build_taskbook, taskbook_payload
from pearl_learning.src.casebook import (
    build_casebook,
    physical_geometry_id,
    validate_casebook_disjoint,
)
from pearl_learning.src.transferability import task_descriptor, transferability_report
from pearl_learning.src.transferability_calibration import calibration_report
from pearl_learning.src.transferability_decision import transferability_decision_report
from pearl_learning.src.validation_freeze import freeze_validation_protocol, verify_validation_freeze
from pearl_learning.src.support_selection import order_support_cases
from pearl_learning.scripts.run_equal_budget_analysis import _case_groups, _selected_support_cases
from pearl_learning.scripts.build_transferability_taskbook import extend_validation_catalog
from pearl_learning.scripts.run_formal_baseline_suite import baseline_commands
from pearl_learning.scripts.audit_task_heterogeneity import heterogeneity_report
from pearl_learning.scripts.audit_integrity import _audit_taskbook
from pearl_learning.scripts.run_baselines import (
    _bind_training_protocol,
    _latest_sac_checkpoint,
    _partial_payload,
    _partial_protocol_matches,
)
from pearl_learning.scripts.select_per_task_sac_checkpoints import selection_key
from pearl_learning.scripts.select_pooled_sac_checkpoint import aggregate_key


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
        cls.config = read_config("pearl_learning/configs/dense_pearl_baseline.yaml")

    def test_task_schema_rejects_non_current_and_split_hashes_are_disjoint(self):
        taskbook = build_taskbook(self.config)
        task = taskbook["meta_train"][0]
        invalid = task.to_dict(); invalid["schema"] = "unsupported"
        with self.assertRaises(ValueError):
            LogicalScenarioTaskSpec.from_dict(invalid)
        train_hashes = {task.map_hash for task in taskbook["meta_train"]}
        self.assertFalse(train_hashes & {task.map_hash for task in taskbook["meta_validation"]})
        self.assertFalse(train_hashes & {task.map_hash for task in taskbook["meta_test_logical"]})
        relations = {task.priority_spec["target_contact_speed_relation"] for tasks in taskbook.values() for task in tasks}
        self.assertTrue({"adversary_faster", "sut_faster"} <= relations)
        train = taskbook["meta_train"]
        self.assertEqual(len(train), 10)
        by_map: dict[str, set[str]] = {}
        for task in train:
            recipe = json.dumps(task.map_config, sort_keys=True)
            by_map.setdefault(recipe, set()).add(task.priority_spec["target_contact_entry_order"])
        self.assertTrue(all(values == {"adversary_first", "sut_first"} for values in by_map.values()))

    def test_method_flow_pilot_activates_only_the_six_physical_tasks(self):
        config = read_config("pearl_learning/configs/merge_method_flow_pilot.yaml")
        self.assertEqual(config["method_flow_pilot"]["task_ids"]["meta_test_template"], [])
        self.assertEqual(config["method_flow_pilot"]["task_ids"]["meta_test_logical"], [])
        selected = _audit_taskbook(config, build_taskbook(config))
        self.assertEqual(sum(map(len, selected.values())), 6)
        self.assertEqual(len(selected["meta_train"]), 4)
        self.assertEqual(len(selected["meta_validation"]), 2)

    def test_route_projection_wraps_heading_and_uses_arc_length(self):
        route = RoutePolyline((("a", "b", 0),), np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]), np.asarray([0.0, 10.0, 20.0]), (20.0,))
        projection = route.projection([5.0, 1.0], 2 * np.pi - 0.2)
        self.assertAlmostEqual(projection.s_m, 5.0, places=5)
        self.assertAlmostEqual(projection.lateral_m, 1.0, places=5)
        self.assertAlmostEqual(projection.heading_error, -0.2, places=5)
        self.assertAlmostEqual(wrap_to_pi(3 * np.pi), -np.pi, places=5)

    def test_route_builder_smooths_required_lane_change_instead_of_lateral_teleport(self):
        class Lane:
            length = 20.0
            def __init__(self, x, y): self.x, self.y = x, y
            def position(self, s, lateral): return np.asarray([self.x + s, self.y + lateral], dtype=float)
        lanes = {
            ("a", "b", 0): Lane(0.0, 0.0),
            ("b", "c", 0): Lane(20.0, 3.5),
        }
        graph = type("Road", (), {"get_lane": lambda self, index: lanes[index]})()
        env = type("Env", (), {"current_map": type("Map", (), {"road_network": graph})()})()
        route = RoutePolyline.from_env(env, {
            "route_id": "lane_change",
            "lane_sequence": [["a", "b", 0], ["b", "c", 0]],
        })
        segments = np.diff(route.points, axis=0)
        headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
        self.assertLess(float(np.max(np.linalg.norm(segments, axis=1))), 1.0)
        self.assertLess(float(np.max(np.abs(np.diff(headings)))), 0.5)
        self.assertTrue(np.allclose(route.points[0], [0.0, 0.0]))
        self.assertTrue(np.allclose(route.points[-1], [40.0, 3.5]))
        self.assertEqual(len(route.lane_change_intervals_m), 1)
        self.assertTrue(route.in_lane_change(sum(route.lane_change_intervals_m[0]) / 2.0))

    def test_marking_violation_is_separate_from_wrong_route_penalty(self):
        events = {"lane_marking_violation": True, "wrong_route": False}
        reward = compute_reward(5.0, 50.0, np.zeros(2), np.zeros(2), events, self.config["reward"])
        self.assertEqual(reward.wrong_route, 0.0)
        self.assertEqual(reward.lane_marking_violation, 0.0)

    def test_dense_rule_and_route_shaping_is_bounded_and_rule_sensitive(self):
        reward = compute_reward(
            5.0, 50.0, np.zeros(2), np.zeros(2), {}, self.config["reward"],
            {"route_progress": 3.0, "priority_alignment": -2.0, "route_deviation": 1.5},
        )
        self.assertAlmostEqual(reward.route_progress, self.config["reward"]["route_progress_weight"])
        self.assertAlmostEqual(reward.priority_alignment, -self.config["reward"]["priority_alignment_weight"])
        self.assertAlmostEqual(reward.route_deviation, -1.5 * self.config["reward"]["route_deviation_weight"])
        required = required_invalid_event_penalty(self.config["reward"], self.config["environment"]["horizon"])
        self.assertLess(required, self.config["reward"]["non_target_collision_penalty"])

    def test_route_status_accepts_aligned_successor_only_at_route_completion(self):
        adapter = DummyAdapter()
        route = RoutePolyline(
            (("a", "b", 0),),
            np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=float),
            np.asarray([0.0, 10.0], dtype=float),
            (10.0,),
        )
        adapter._routes = {"sut": route}
        graph = {"a": {"b": []}, "b": {"c": []}, "x": {"y": []}}
        env = type("Env", (), {"current_map": type("Map", (), {"road_network": type("Road", (), {"graph": graph})()})()})()
        lane = type("Lane", (), {"width": 3.8})()
        completed = type("Vehicle", (), {
            "position": np.asarray([10.1, 0.0]), "heading_theta": 0.0,
            "lane_index": ("b", "c", 0), "lane": lane, "LENGTH": 5.0,
        })()
        progress, wrong, route_complete = adapter.route_status(env, completed, "sut", 9.0)
        self.assertAlmostEqual(progress, 10.0)
        self.assertFalse(wrong)
        self.assertTrue(route_complete)

        final_planned_lane = type("Vehicle", (), {
            "position": np.asarray([9.0, 0.0]), "heading_theta": 0.0,
            "lane_index": ("a", "b", 0), "lane": lane, "LENGTH": 5.0,
        })()
        _, wrong, route_complete = adapter.route_status(env, final_planned_lane, "sut", 8.0)
        self.assertFalse(wrong)
        self.assertTrue(route_complete)

        mid_route_branch = type("Vehicle", (), {
            "position": np.asarray([5.0, 0.0]), "heading_theta": 0.0,
            "lane_index": ("x", "y", 0), "lane": lane, "LENGTH": 5.0,
        })()
        _, wrong, route_complete = adapter.route_status(env, mid_route_branch, "sut", 4.0)
        self.assertTrue(wrong)
        self.assertFalse(route_complete)

        adapter._routes["sut"] = RoutePolyline(
            route.lane_indices, route.points, route.arc_lengths_m, route.lane_end_s_m, ((4.0, 8.0),),
        )
        _, wrong, route_complete = adapter.route_status(env, mid_route_branch, "sut", 4.0)
        self.assertFalse(wrong)
        self.assertFalse(route_complete)

    def test_route_status_accepts_parallel_merge_connector_that_rejoins_frozen_route(self):
        adapter = DummyAdapter()
        route = RoutePolyline(
            ((">>>", "planned_connector", 0), ("planned_connector", "shared", 0)),
            np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=float),
            np.asarray([0.0, 10.0, 20.0], dtype=float),
            (10.0, 20.0),
        )
        adapter._routes = {"adversary": route}
        graph = {
            ">>>": {"parallel_connector": []},
            "parallel_connector": {"shared": []},
            "planned_connector": {"shared": []},
        }
        env = type("Env", (), {"current_map": type("Map", (), {"road_network": type("Road", (), {"graph": graph})()})()})()
        lane = type("Lane", (), {"width": 3.8})()
        vehicle = type("Vehicle", (), {
            "position": np.asarray([8.0, 0.0]), "heading_theta": 0.0,
            "lane_index": (">>>", "parallel_connector", 0), "lane": lane, "LENGTH": 5.0,
        })()
        _, wrong, completed = adapter.route_status(env, vehicle, "adversary", 7.5)
        self.assertFalse(wrong)
        self.assertFalse(completed)

    def test_conflict_frame_ignores_close_parallel_approach_before_shared_lane(self):
        adapter = DummyAdapter()
        adv_points = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 6.0], [30.0, 0.0], [40.0, 0.0]])
        sut_points = np.asarray([[0.0, 3.5], [10.0, 3.5], [20.0, -6.0], [30.0, 0.0], [40.0, 0.0]])
        adv_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(adv_points, axis=0), axis=1))))
        sut_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(sut_points, axis=0), axis=1))))
        adapter._routes = {
            "adversary": RoutePolyline((("a", "b", 0), ("b", "c", 0), ("shared", "out", 0)), adv_points, adv_arc, (10.0, 30.0, float(adv_arc[-1]))),
            "sut": RoutePolyline((("d", "e", 0), ("e", "c", 0), ("shared", "out", 0)), sut_points, sut_arc, (10.0, 30.0, float(sut_arc[-1]))),
        }
        task = type("Task", (), {"task_id": "parallel_then_merge", "conflict_spec": {"max_route_distance_m": 1.0, "conflict_radius_m": 4.0}})()
        vehicle = type("Vehicle", (), {"LENGTH": 5.0})()
        frame = adapter.conflict_frame(object(), task, vehicle, vehicle)
        self.assertGreater(frame["adversary_conflict_s_m"], 0.0)
        self.assertGreater(frame["sut_conflict_s_m"], 0.0)

    def test_observation_priority_features_are_complementary(self):
        route = RoutePolyline((("a", "b", 0),), np.asarray([[0.0, 0.0], [20.0, 0.0]]), np.asarray([0.0, 20.0]), (20.0,))
        frame = {"adversary_route": route, "sut_route": route, "adversary_conflict_s_m": 10.0, "sut_conflict_s_m": 10.0, "priority_spec": {"sut_has_priority": True}}
        topology = {"num_incoming_branches": 2.0, "num_outgoing_branches": 1.0, "adversary_lane_count": 1.0, "sut_lane_count": 1.0, "merge_length_m": 20.0, "conflict_radius_m": 3.0, "adversary_route_curvature": 0.0, "sut_route_curvature": 0.0, "adversary_speed_limit_mps": 15.0, "sut_speed_limit_mps": 15.0, "num_conflict_zones": 1.0}
        observation = build_observation(DummyVehicle([0.0, 0.0]), DummyVehicle([1.0, 0.0]), frame, topology, self.config)
        self.assertEqual(float(observation[OBS_FIELDS.index("adversary_priority")]), 0.0)
        self.assertEqual(float(observation[OBS_FIELDS.index("sut_priority")]), 1.0)

    def test_no_topology_ablation_masks_only_the_topology_descriptor(self):
        route = RoutePolyline((("a", "b", 0),), np.asarray([[0.0, 0.0], [20.0, 0.0]]), np.asarray([0.0, 20.0]), (20.0,))
        frame = {"adversary_route": route, "sut_route": route, "adversary_conflict_s_m": 10.0, "sut_conflict_s_m": 10.0, "priority_spec": {"sut_has_priority": True}}
        topology = {"num_incoming_branches": 2.0, "num_outgoing_branches": 1.0, "adversary_lane_count": 1.0, "sut_lane_count": 1.0, "merge_length_m": 20.0, "conflict_radius_m": 3.0, "adversary_route_curvature": 0.0, "sut_route_curvature": 0.0, "adversary_speed_limit_mps": 15.0, "sut_speed_limit_mps": 15.0, "num_conflict_zones": 1.0}
        full = build_observation(DummyVehicle([0.0, 0.0]), DummyVehicle([1.0, 0.0]), frame, topology, self.config)
        ablated_config = copy.deepcopy(self.config); ablated_config["ablation"] = {"no_topology": True}
        ablated = build_observation(DummyVehicle([0.0, 0.0]), DummyVehicle([1.0, 0.0]), frame, topology, ablated_config)
        self.assertTrue(np.allclose(full[:24], ablated[:24]))
        self.assertTrue(np.allclose(ablated[24:], 0.0))

    def test_pairwise_collision_uses_obb_without_crash_flags(self):
        adapter = DummyAdapter()
        first, second = DummyVehicle([0.0, 0.0]), DummyVehicle([1.0, 0.0])
        hit, method = adapter.target_contact(object(), first, second)
        self.assertTrue(hit); self.assertEqual(method, "obb_overlap")
        hit, method = adapter.target_contact(object(), DummyVehicle([0.0, 0.0], crash=True), DummyVehicle([100.0, 0.0], crash=True))
        self.assertFalse(hit); self.assertEqual(method, "no_pairwise_contact")

    def test_frozen_contact_rule_is_hidden_from_observation_but_uses_physical_speed(self):
        adversary_first = {"sut_has_priority": True, "target_contact_speed_relation": "adversary_faster", "target_contact_speed_margin_mps": 0.5}
        sut_first = {"sut_has_priority": True, "target_contact_speed_relation": "sut_faster", "target_contact_speed_margin_mps": 0.5}
        self.assertTrue(target_contact_matches_rule(adversary_first, 8.0, 7.0))
        self.assertFalse(target_contact_matches_rule(adversary_first, 7.0, 8.0))
        self.assertTrue(target_contact_matches_rule(sut_first, 7.0, 8.0))
        self.assertFalse(target_contact_matches_rule(sut_first, 8.0, 7.0))
        entry_rule = {"sut_has_priority": True, "target_contact_entry_order": "adversary_first"}
        self.assertTrue(target_contact_matches_rule(entry_rule, 0.0, 0.0, "adversary"))
        self.assertFalse(target_contact_matches_rule(entry_rule, 0.0, 0.0, "sut"))

    def test_low_ttc_is_strict_success_only_when_hidden_rule_is_satisfied(self):
        events = {
            "target_collision": False,
            "physical_critical_proximity": True,
            "rule_satisfied_critical_proximity": False,
            "non_target_collision": False,
            "adversary_out_of_road": False,
            "sut_out_of_road": False,
            "wrong_route": False,
            "lane_marking_violation": False,
        }
        metrics = EpisodeMetrics("task", "case")
        metrics.update(0.0, 1.0, 3.0, events, "no_pairwise_contact")
        wrong_rule = metrics.record(1.5)
        self.assertTrue(wrong_rule["physical_critical"])
        self.assertFalse(wrong_rule["critical"])
        self.assertFalse(wrong_rule["valid_critical_strict"])

        events["rule_satisfied_critical_proximity"] = True
        metrics.update(0.0, 1.0, 3.0, events, "no_pairwise_contact")
        matched_rule = metrics.record(1.5)
        self.assertTrue(matched_rule["critical"])
        self.assertTrue(matched_rule["valid_critical_strict"])

    def test_rule_satisfied_near_miss_receives_the_success_bonus(self):
        cfg = self.config["reward"]
        events = {"rule_satisfied_critical_proximity": True}
        reward = compute_reward(
            1.0, 3.0, np.zeros(2), np.zeros(2), events, cfg,
        )
        self.assertEqual(reward.target_collision, float(cfg["target_collision_bonus"]))

    def test_route_tracker_preserves_throttle_and_bounds_steering_residual(self):
        config = {"control": {"route_tracking": {
            "enabled": True, "heading_gain": 1.5,
            "lateral_gain": 0.12, "residual_scale": 0.25,
        }}}
        applied = compose_route_tracking_action(
            np.asarray([0.4, -0.3], dtype=np.float32), 0.2, 0.5, config,
        )
        self.assertAlmostEqual(float(applied[0]), 1.5 * 0.2 - 0.12 * 0.5 + 0.25 * 0.4)
        self.assertAlmostEqual(float(applied[1]), -0.3)
        saturated = compose_route_tracking_action(
            np.asarray([1.0, 0.2], dtype=np.float32), 2.0, -5.0, config,
        )
        self.assertEqual(float(saturated[0]), 1.0)

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

    def test_transition_product_is_invariant_to_episode_regrouping(self):
        torch.manual_seed(11)
        encoder = ContextEncoder(4, 2, [8], aggregation="transition_product")
        rows = torch.randn((2, 1, 12, 4))
        first_mu, first_log_var = encoder(rows)
        second_mu, second_log_var = encoder(rows.reshape(2, 3, 4, 4))
        self.assertTrue(torch.allclose(first_mu, second_mu))
        self.assertTrue(torch.allclose(first_log_var, second_log_var))
        self.assertEqual(tuple(first_mu.shape), (2, 2))
        with self.assertRaises(ValueError):
            encoder(torch.zeros((2, 3, 4)))

    def test_episode_product_is_an_explicit_non_default_ablation(self):
        encoder = ContextEncoder(4, 2, [8], aggregation="episode_product")
        self.assertEqual(encoder.aggregation, "episode_product")
        with self.assertRaises(ValueError):
            ContextEncoder(4, 2, [8], aggregation="implicit_pooling")

    def test_recent_context_buffer_is_bounded_and_rl_batch_is_disjoint(self):
        buffers = TaskReplayBuffers(["task"], recent_context_episodes=2)
        for index in range(3):
            rows = [transition(f"episode-{index}"), transition(f"episode-{index}", truncated=True)]
            buffers.add_episode("task", rows)
        self.assertEqual(len(buffers.buffers["task"].episodes), 3)
        self.assertEqual(
            [episode.episode_id for episode in buffers.recent_context_buffers["task"].episodes],
            ["episode-1", "episode-2"],
        )
        contexts = buffers.context_per_task(["task"], 4, 2, np.random.default_rng(2))
        context_ids = {group[0].episode_id for group in contexts[0]}
        self.assertEqual(context_ids, {"episode-1", "episode-2"})
        rl_batch = buffers.sample_per_task_excluding_context(
            ["task"], contexts, 16, np.random.default_rng(3),
        )[0]
        self.assertTrue(all(row.episode_id == "episode-0" for row in rl_batch))
        buffers.clear_recent_context()
        self.assertFalse(buffers.recent_context_buffers["task"].episodes)

    def test_invalid_event_penalties_dominate_maximum_positive_episode_return(self):
        reward_cfg = self.config["reward"]
        horizon = int(self.config["environment"]["horizon"])
        required = required_invalid_event_penalty(reward_cfg, horizon)
        expected = reward_cfg["target_collision_bonus"] + horizon * sum(
            reward_cfg[key] for key in (
                "ttc_weight", "proximity_weight", "route_progress_weight", "priority_alignment_weight",
            )
        ) + reward_cfg["invalid_penalty_margin"]
        self.assertAlmostEqual(required, expected)
        self.assertEqual(validate_reward_contract(reward_cfg, horizon), required)
        weak = copy.deepcopy(reward_cfg)
        weak["wrong_route_penalty"] = required - 0.01
        with self.assertRaises(ValueError):
            validate_reward_contract(weak, horizon)
        no_margin = copy.deepcopy(reward_cfg)
        no_margin["invalid_penalty_margin"] = 0.0
        with self.assertRaises(ValueError):
            validate_reward_contract(no_margin, horizon)

    def test_meta_batch_sampling_has_no_duplicate_tasks(self):
        tasks = list(range(10))
        sampled = _sample_tasks_without_replacement(tasks, 6, np.random.default_rng(8))
        self.assertEqual(len(sampled), 6)
        self.assertEqual(len(set(sampled)), 6)
        saturated = _sample_tasks_without_replacement(tasks, 20, np.random.default_rng(9))
        self.assertEqual(set(saturated), set(tasks))
        with self.assertRaises(ValueError):
            _sample_tasks_without_replacement(tasks, 0, np.random.default_rng(10))

    def test_evaluation_regimes_separate_id_and_ood_logical_types(self):
        self.assertEqual(evaluation_regime("meta_test_template"), "id_known_logical_type")
        self.assertEqual(evaluation_regime("meta_test_logical"), "ood_unseen_logical_type")
        with self.assertRaises(ValueError):
            evaluation_regime("meta_test")

    def test_empty_context_prior_is_exact_unit_normal(self):
        agent = PEARLAgent(37, 2, self.config, torch.device("cpu"))
        mu, log_var = agent.prior(3)
        self.assertTrue(torch.equal(mu, torch.zeros_like(mu)))
        self.assertTrue(torch.equal(log_var, torch.zeros_like(log_var)))

    def test_fixed_context_is_a_nested_episode_prefix(self):
        first = Rollout(
            [transition("episode-a") for _ in range(40)],
            {"case_id": "case-a"},
            "prior_support",
            "episode-a",
        )
        second = Rollout(
            [transition("episode-b") for _ in range(40)],
            {"case_id": "case-b"},
            "posterior_rollout",
            "episode-b",
        )
        first_block, first_audit = _fixed_episode_context_block(
            first, 32, base_seed=17, task_id="task", scheme="random",
        )
        again, again_audit = _fixed_episode_context_block(
            first, 32, base_seed=17, task_id="task", scheme="random",
        )
        second_block, second_audit = _fixed_episode_context_block(
            second, 32, base_seed=17, task_id="task", scheme="random",
        )
        self.assertEqual(first_audit, again_audit)
        self.assertEqual(
            [row.episode_id for row in first_block],
            [row.episode_id for row in again],
        )
        context_one, audit_one = _posterior_context(
            [first],
            [first_block],
            [first_audit],
            total_size=64,
            per_episode=32,
        )
        context_two, audit_two = _posterior_context(
            [first, second],
            [first_block, second_block],
            [first_audit, second_audit],
            total_size=64,
            per_episode=32,
        )
        self.assertEqual(len(context_one), 1)
        self.assertEqual(len(context_two), 2)
        self.assertEqual(
            audit_one["context_episode_sample_hashes"],
            audit_two["context_episode_sample_hashes"][:1],
        )
        self.assertEqual(audit_two["context_transition_count"], 64)

    def test_adaptation_task_pairs_match_initial_conditions_but_not_ids_or_seeds(self):
        config = read_config("pearl_learning/configs/posterior_adaptation_protocol.yaml")
        taskbook = build_taskbook(config)
        self.assertEqual(
            {split: len(tasks) for split, tasks in taskbook.items()},
            {
                "meta_train": 10,
                "meta_validation": 4,
                "meta_test_template": 4,
                "meta_test_logical": 8,
            },
        )
        casebooks = {
            task.task_id: build_casebook(task, config)
            for tasks in taskbook.values()
            for task in tasks
        }
        validate_casebook_disjoint(casebooks)
        for split in ("meta_validation", "meta_test_template", "meta_test_logical"):
            pairs: dict[str, list[LogicalScenarioTaskSpec]] = {}
            for task in taskbook[split]:
                pairs.setdefault(physical_geometry_id(task.geometry_id), []).append(task)
            for tasks in pairs.values():
                self.assertEqual(len(tasks), 2)
                self.assertEqual(
                    {task.priority_spec["target_contact_entry_order"] for task in tasks},
                    {"adversary_first", "sut_first"},
                )
                left, right = tasks
                self.assertEqual(left.map_config, right.map_config)
                self.assertEqual(left.adversary_route, right.adversary_route)
                self.assertEqual(left.sut_route, right.sut_route)
                for case_split in ("validation_support", "validation_query", "test_support", "test_query"):
                    left_cases = casebooks[left.task_id][case_split]
                    right_cases = casebooks[right.task_id][case_split]
                    for left_case, right_case in zip(left_cases, right_cases):
                        for field in ("adversary_speed_mps", "adversary_spawn_m", "sut_spawn_m"):
                            self.assertEqual(left_case[field], right_case[field])
                        self.assertNotEqual(left_case["case_id"], right_case["case_id"])
                        self.assertNotEqual(left_case["case_seed"], right_case["case_seed"])

    def test_adaptation_statistics_pair_methods_by_seed_and_task(self):
        rows = []
        for seed in (11, 22, 33):
            for task_id, gain in (("task-a", 0.2), ("task-b", 0.1)):
                rows.extend((
                    {"method": "pearl_full", "training_seed": seed, "task_id": task_id, "K": 4, "valid_critical_strict_rate": 0.5 + gain},
                    {"method": "pearl_no_context", "training_seed": seed, "task_id": task_id, "K": 4, "valid_critical_strict_rate": 0.5},
                ))
        effect = paired_method_effect(
            rows,
            method="pearl_full",
            reference="pearl_no_context",
            shot=4,
            metric="valid_critical_strict_rate",
            samples=200,
            confidence=0.95,
        )
        self.assertEqual(effect["training_seed_count"], 3)
        self.assertAlmostEqual(float(effect["mean"]), 0.15)
        self.assertGreater(float(effect["ci_lower"]), 0.0)
        first = task_cluster_interval(
            {"task-a": [0.1, 0.2], "task-b": [0.3, 0.4]},
            samples=100,
            confidence=0.95,
            seed=7,
        )
        second = task_cluster_interval(
            {"task-a": [0.1, 0.2], "task-b": [0.3, 0.4]},
            samples=100,
            confidence=0.95,
            seed=7,
        )
        self.assertEqual(first, second)

    def test_parameter_hash_includes_target_critics_and_alpha(self):
        agent = PEARLAgent(37, 2, self.config, torch.device("cpu"))
        self.assertEqual(
            set(agent.module_hashes()),
            {"context_encoder", "actor", "q1", "q2", "target_q1", "target_q2", "log_alpha"},
        )
        before = agent.parameter_hash()
        with torch.no_grad():
            next(agent.target_q1.parameters()).add_(1.0)
        self.assertNotEqual(before, agent.parameter_hash())
        before = agent.parameter_hash()
        with torch.no_grad():
            agent.log_alpha.add_(0.5)
        self.assertNotEqual(before, agent.parameter_hash())

    def test_topology_only_training_keeps_context_encoder_frozen(self):
        config = copy.deepcopy(self.config)
        config["ablation"] = {"no_context_training": True}
        agent = PEARLAgent(37, 2, config, torch.device("cpu"))
        before = agent.module_hashes()
        context = [[[transition("context") for _ in range(32)]]]
        rl = [[transition("rl") for _ in range(8)]]
        agent.update(context, rl)
        after = agent.module_hashes()
        self.assertEqual(before["context_encoder"], after["context_encoder"])
        self.assertNotEqual(before["actor"], after["actor"])

    def test_truncated_transition_is_not_bootstrap_terminal(self):
        row = transition("episode", terminated=False, truncated=True)
        self.assertFalse(row.terminated)
        self.assertTrue(row.truncated)

    def test_formal_validation_has_no_circular_pearl_no_context_requirement(self):
        self.assertNotIn("pearl_no_context", REQUIRED_PRETRAIN_BASELINES)
        descriptor, path = tempfile.mkstemp(suffix=".json")
        os.close(descriptor)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({
                "schema": "logical_merge_formal_validation", "taskbook_hash": "frozen",
                "topology_audit": "pass", "integrity_audit": "pass", "heterogeneity_audit": "pass",
                "baseline_environment_steps": 5000,
                "completed_baselines": sorted(REQUIRED_PRETRAIN_BASELINES),
                }, handle)
            verify_formal_validation(path, "frozen")
        finally:
            os.unlink(path)

    def test_formal_baseline_suite_covers_required_names_and_dependencies(self):
        args = argparse.Namespace(
            config="config", taskbook="taskbook", casebook_root="casebooks",
            output="baselines", seed=7, environment_steps=5000, checkpoint_interval_steps=1000,
        )
        commands = dict(baseline_commands(args))
        self.assertEqual(set(commands), REQUIRED_PRETRAIN_BASELINES)
        cross = " ".join(commands["cross_task_policy_matrix"]).replace("\\", "/")
        finetune = " ".join(commands["pooled_finetune_sac"]).replace("\\", "/")
        pooled = " ".join(commands["topology_conditioned_pooled_sac"])
        oracle = " ".join(commands["oracle_task_conditioned_sac"])
        self.assertIn("baselines/per_task_sac/policies", cross)
        self.assertIn("baselines/topology_conditioned_pooled_sac/model.zip", finetune)
        self.assertIn("--pooled-steps-per-task 5000", pooled)
        self.assertIn("--pooled-steps-per-task 5000", oracle)
        self.assertIn("--checkpoint-interval-steps 1000", " ".join(commands["per_task_sac"]))

    def test_heterogeneity_audit_aggregates_distinct_seed_roots_without_records(self):
        taskbook = build_taskbook(self.config)
        digest = content_hash(taskbook_payload(taskbook))
        train_ids = [task.task_id for task in taskbook["meta_train"]]
        with tempfile.TemporaryDirectory() as directory:
            roots = []
            for seed in (3, 4):
                root = os.path.join(directory, f"seed_{seed}"); roots.append(root)
                for baseline in ("per_task_sac", "cross_task_policy_matrix", "topology_conditioned_pooled_sac"):
                    budget = {
                        "scope": "pooled_shared" if baseline == "topology_conditioned_pooled_sac" else "per_task_independent",
                        "per_task_environment_steps": 5000,
                        "total_environment_steps": 5000 * len(train_ids),
                    }
                    write_json(os.path.join(root, baseline, "baseline_manifest.json"), {
                        "baseline": baseline, "taskbook_hash": digest, "status": "completed", "seed": seed,
                        "training_budget": budget,
                    })
                per_task = {task_id: {"summary": {"valid_critical_strict_rate": 0.6}} for task_id in train_ids}
                pooled = {task_id: {"summary": {"valid_critical_strict_rate": 0.5}} for task_id in train_ids}
                write_json(os.path.join(root, "per_task_sac", "per_task_metrics.json"), {"tasks": per_task})
                write_json(os.path.join(root, "topology_conditioned_pooled_sac", "pooled_metrics.json"), {"tasks": pooled})
                write_json(os.path.join(root, "cross_task_policy_matrix", "cross_task_matrix.json"), {
                    "policy_tasks": train_ids, "evaluation_tasks": [], "matrix": {},
                })
            report = heterogeneity_report(taskbook, roots, minimum_gap=0.02, minimum_seeds=2)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["seed_count"], 2)
        self.assertEqual(report["seed_evidence"][0]["training_budget"]["pooled_total_environment_steps"], 5000 * len(train_ids))
        self.assertTrue(all("records" not in str(item) for item in report["seed_evidence"]))

    def test_heterogeneity_audit_rejects_unbalanced_pooled_budget(self):
        taskbook = build_taskbook(self.config)
        digest = content_hash(taskbook_payload(taskbook))
        train_ids = [task.task_id for task in taskbook["meta_train"]]
        with tempfile.TemporaryDirectory() as directory:
            for baseline, budget in {
                "per_task_sac": {"per_task_environment_steps": 20000, "total_environment_steps": 20000 * len(train_ids)},
                "cross_task_policy_matrix": {"per_task_environment_steps": 20000, "total_environment_steps": 0},
                "topology_conditioned_pooled_sac": {"per_task_environment_steps": 2000, "total_environment_steps": 20000},
            }.items():
                write_json(os.path.join(directory, baseline, "baseline_manifest.json"), {
                    "baseline": baseline, "taskbook_hash": digest, "status": "completed", "seed": 3,
                    "training_budget": budget,
                })
            values = {task_id: {"summary": {"valid_critical_strict_rate": 0.5}} for task_id in train_ids}
            write_json(os.path.join(directory, "per_task_sac", "per_task_metrics.json"), {"tasks": values})
            write_json(os.path.join(directory, "topology_conditioned_pooled_sac", "pooled_metrics.json"), {"tasks": values})
            write_json(os.path.join(directory, "cross_task_policy_matrix", "cross_task_matrix.json"), {
                "policy_tasks": train_ids, "evaluation_tasks": [], "matrix": {},
            })
            with self.assertRaises(SystemExit):
                heterogeneity_report(taskbook, [directory], minimum_gap=0.02, minimum_seeds=1)

    def test_sac_checkpoint_resume_prefers_latest_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.join(directory, "checkpoints"); os.makedirs(root)
            for step in (5000, 15000, 10000):
                open(os.path.join(root, f"task_{step}_steps.zip"), "wb").close()
            latest = _latest_sac_checkpoint(Path(root), "task")
        self.assertEqual(latest.name, "task_15000_steps.zip")

    def test_partial_baseline_metrics_are_invalidated_when_budget_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            write_json(path, {
                "protocol": {"environment_steps": 5000, "seed": 11},
                "tasks": {"task": {"summary": {"valid_critical_strict_rate": 0.5}}},
            })
            old = {"environment_steps": 5000, "seed": 11}
            new = {"environment_steps": 20000, "seed": 11}
            self.assertIn("task", _partial_payload(path, old, True, "tasks"))
            self.assertTrue(_partial_protocol_matches(path, old, True))
            self.assertEqual(_partial_payload(path, new, True, "tasks"), {})
            self.assertFalse(_partial_protocol_matches(path, new, True))

    def test_baseline_resume_rejects_unbound_or_incompatible_code_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.zip").write_bytes(b"checkpoint")
            with self.assertRaises(SystemExit):
                _bind_training_protocol(root, {"implementation_hash": "current"}, True)
            _bind_training_protocol(root, {"implementation_hash": "old"}, False)
            with self.assertRaises(SystemExit):
                _bind_training_protocol(root, {"implementation_hash": "current"}, True)
            _bind_training_protocol(root, {"implementation_hash": "old"}, True)

    def test_per_task_checkpoint_selection_uses_validation_primary_metric(self):
        weak = {"valid_critical_strict_rate": 0.2, "invalid_rate": 0.0, "mean_episode_return": 10.0}
        strong = {"valid_critical_strict_rate": 0.3, "invalid_rate": 0.2, "mean_episode_return": -10.0}
        self.assertGreater(selection_key(strong, 10000), selection_key(weak, 5000))
        valid = {"valid_critical_strict_rate": 0.3, "invalid_rate": 0.0, "mean_episode_return": 0.0}
        self.assertGreater(selection_key(valid, 10000), selection_key(strong, 5000))
        self.assertGreater(selection_key(valid, 5000), selection_key(valid, 10000))

    def test_pooled_checkpoint_selection_aggregates_validation_tasks(self):
        metrics = {
            "a": {"summary": {"valid_critical_strict_rate": 0.2, "invalid_rate": 0.1, "mean_episode_return": 1.0}},
            "b": {"summary": {"valid_critical_strict_rate": 0.4, "invalid_rate": 0.3, "mean_episode_return": 3.0}},
        }
        actual = aggregate_key(metrics, 5000)
        self.assertAlmostEqual(actual[0], 0.3)
        self.assertAlmostEqual(actual[1], -0.2)
        self.assertEqual(actual[2:], (2.0, -5000))

    def test_compact_fewshot_result_removes_episode_records(self):
        result = {"split": "meta_test_logical", "parameter_hash_before": "same", "parameter_hash_after": "same", "no_gradient_adaptation": True, "no_topology_ablation": False, "context_protocol": {}, "provenance": {}, "tasks": {"task": {"5": {"summary": {"valid_critical_strict_rate": 0.6}, "records": [{"case_id": "query"}], "support_environment_steps": 57}}}}
        compact = compact_fewshot_result(result)
        self.assertEqual(compact["tasks"]["task"]["5"]["summary"]["valid_critical_strict_rate"], 0.6)
        self.assertNotIn("records", compact["tasks"]["task"]["5"])

    def test_summary_reports_valid_critical_initial_condition_diversity_as_secondary_diagnostic(self):
        records = [
            {"case_id": "a", "episode_return": 2.0, "episode_length": 4, "valid_critical_strict": True, "target_collision": True, "critical": True, "valid": True, "min_ttc": 0.5, "min_distance": 1.0},
            {"case_id": "b", "episode_return": 4.0, "episode_length": 6, "valid_critical_strict": True, "target_collision": True, "critical": True, "valid": True, "min_ttc": 0.4, "min_distance": 0.8},
            {"case_id": "c", "episode_return": 0.0, "episode_length": 8, "valid_critical_strict": False, "target_collision": False, "critical": False, "valid": True, "min_ttc": 3.0, "min_distance": 8.0},
        ]
        metadata = {
            "a": {"adversary_speed_mps": 10.0, "adversary_spawn_m": 0.0, "sut_spawn_m": 0.0},
            "b": {"adversary_speed_mps": 20.0, "adversary_spawn_m": 10.0, "sut_spawn_m": 10.0},
            "c": {"adversary_speed_mps": 15.0, "adversary_spawn_m": 5.0, "sut_spawn_m": 5.0},
        }
        summary = summarize(records, case_metadata=metadata)
        self.assertEqual(summary["valid_critical_case_count"], 2)
        self.assertGreater(float(summary["valid_critical_initial_condition_diversity"]), 0.0)
        self.assertEqual(summary["valid_critical_case_metadata_coverage"], 1.0)
        self.assertEqual(summary["mean_episode_return"], 2.0)
        self.assertEqual(summary["query_environment_steps"], 18)
        self.assertEqual(summary["environment_steps_to_first_valid_critical"], 4)

    def test_support_selection_is_deterministic_and_pre_execution_only(self):
        cases = [
            {"case_id": "case_0", "adversary_speed_mps": 10.0, "adversary_spawn_m": 0.0, "sut_spawn_m": 0.0},
            {"case_id": "case_1", "adversary_speed_mps": 11.0, "adversary_spawn_m": 10.0, "sut_spawn_m": 10.0},
            {"case_id": "case_2", "adversary_speed_mps": 16.0, "adversary_spawn_m": 100.0, "sut_spawn_m": -20.0},
        ]
        first, provenance = order_support_cases(cases, "initial_condition_diversity", seed=9)
        second, again = order_support_cases(cases, "initial_condition_diversity", seed=9)
        self.assertEqual([row["case_id"] for row in first], [row["case_id"] for row in second])
        self.assertEqual(provenance, again)
        self.assertEqual(provenance["selected_case_ids"][0], "case_0")
        self.assertFalse(provenance["uses_query_cases"])
        self.assertFalse(provenance["uses_rollout_outcomes"])
        randomized, _ = order_support_cases(cases, "random", seed=9)
        self.assertEqual(sorted(row["case_id"] for row in randomized), sorted(row["case_id"] for row in cases))
        with self.assertRaises(ValueError):
            order_support_cases(cases, "posterior_action_disagreement", seed=9)

    def test_posterior_action_disagreement_is_deterministic_and_parameter_free(self):
        agent = PEARLAgent(37, 2, self.config, torch.device("cpu"))
        before = agent.parameter_hash()
        mu, log_var = agent.prior()
        observations = np.vstack([np.zeros(37, dtype=np.float32), np.ones(37, dtype=np.float32)])
        first = _posterior_action_disagreement(agent, observations, mu, log_var, seed=17)
        second = _posterior_action_disagreement(agent, observations, mu, log_var, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(np.isfinite(score) and score >= 0.0 for score in first))
        self.assertEqual(before, agent.parameter_hash())

    def test_representation_intervention_masks_only_declared_observation_fields(self):
        original = Transition(
            np.ones(37, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.0,
            np.ones(37, dtype=np.float32), False, True, "horizon", "task", "episode", "case", "prior_support", 0,
        )
        masked = _mask_context_fields([[original]], (1, 3, 24))[0][0]
        self.assertTrue(np.all(masked.obs[[1, 3, 24]] == 0.0))
        self.assertTrue(np.all(masked.next_obs[[1, 3, 24]] == 0.0))
        self.assertTrue(np.all(masked.obs[[0, 2, 4]] == 1.0))
        self.assertTrue(np.all(original.obs == 1.0))
        self.assertEqual(masked.case_id, original.case_id)

    def test_support_posterior_diagnostic_rejects_negative_shots_before_environment_creation(self):
        agent = PEARLAgent(37, 2, self.config, torch.device("cpu"))
        with self.assertRaises(ValueError):
            infer_support_posteriors(agent, self.config, [], {}, "meta_test_logical", [-1])
        report = infer_support_posteriors(agent, self.config, [], {}, "meta_test_logical", [0], {"taskbook_hash": "frozen"})
        self.assertEqual(report["taskbook_hash"], "frozen")

    def test_equal_budget_validation_uses_disjoint_validation_case_groups(self):
        self.assertEqual(_case_groups("meta_validation"), ("validation_support", "validation_query"))
        self.assertEqual(_case_groups("meta_test_logical"), ("test_support", "test_query"))

    def test_equal_budget_reuses_exact_selected_support_prefix(self):
        cases = [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]
        self.assertEqual([case["case_id"] for case in _selected_support_cases(cases, ["c", "a"])], ["c", "a"])
        for selected in ([], ["a", "a"], ["missing"]):
            with self.assertRaises(RuntimeError):
                _selected_support_cases(cases, selected)

    def test_transferability_calibration_rejects_insufficient_or_single_class_validation(self):
        descriptor = {
            "schema": "logical_merge_transferability_report_v1", "taskbook_hash": "frozen",
            "candidate_split": "meta_validation", "uses_hidden_rules": False,
            "candidates": [{"task_id": "task", "nearest_meta_train": {"similarity": 0.5, "distance": {"total": 0.2}}, "coverage_flags": {"unseen_logical_type": False, "unseen_map_kind": False}}],
        }
        taskwise = {
            "schema": "pearl_equal_new_task_budget", "taskbook_hash": "frozen", "split": "meta_validation",
            "budgets": {"5": {"tasks": {"task": {"support_environment_steps": 57, "pearl_valid_critical_strict_rate": 0.6, "scratch_sac_mean": 0.2, "pooled_finetune_sac_mean": 0.3}}}},
        }
        report = calibration_report(descriptor, taskwise, shot=5, minimum_independent_tasks=2)
        self.assertEqual(report["status"], "insufficient_validation_evidence_no_threshold")
        self.assertIn("too_few_independent_validation_tasks", report["reasons"])
        self.assertIn("validation_labels_have_only_one_class", report["reasons"])

    def test_transferability_decision_defers_without_validated_threshold(self):
        descriptor = {
            "schema": "logical_merge_transferability_report_v1", "taskbook_hash": "frozen",
            "candidate_split": "meta_test_logical", "uses_hidden_rules": False,
            "candidates": [{"task_id": "task", "nearest_meta_train": {"similarity": 0.9, "distance": {"total": 0.1}}, "coverage_flags": {"unseen_logical_type": True, "unseen_map_kind": True}}],
        }
        insufficient = {
            "schema": "logical_merge_transferability_calibration_v1", "taskbook_hash": "frozen",
            "split": "meta_validation", "status": "insufficient_validation_evidence_no_threshold",
            "reasons": ["too_few_independent_validation_tasks"],
        }
        decision = transferability_decision_report(descriptor, insufficient)
        self.assertEqual(decision["status"], "deferred_insufficient_validation_evidence")
        self.assertFalse(decision["uses_query_cases_for_decision"])
        self.assertFalse(decision["decisions"][0]["allow_meta_adaptation"])
        calibrated = dict(insufficient) | {"status": "calibrated_validation_only", "threshold": 0.8}
        out_of_sample_deferred = transferability_decision_report(descriptor, calibrated)
        self.assertEqual(out_of_sample_deferred["status"], "deferred_insufficient_out_of_sample_validation")
        calibrated["leave_one_task_out"] = {"coverage": 1.0, "evaluable_task_count": 3}
        accepted = transferability_decision_report(descriptor, calibrated)
        self.assertTrue(accepted["decisions"][0]["allow_meta_adaptation"])

    def test_transferability_calibration_reports_leave_one_task_out_coverage(self):
        descriptor = {
            "schema": "logical_merge_transferability_report_v1", "taskbook_hash": "frozen",
            "candidate_split": "meta_validation", "uses_hidden_rules": False,
            "candidates": [
                {"task_id": "positive_a", "nearest_meta_train": {"similarity": 0.9, "distance": {"total": 0.1}}, "coverage_flags": {"unseen_logical_type": False, "unseen_map_kind": False}},
                {"task_id": "negative", "nearest_meta_train": {"similarity": 0.1, "distance": {"total": 0.9}}, "coverage_flags": {"unseen_logical_type": True, "unseen_map_kind": True}},
                {"task_id": "positive_b", "nearest_meta_train": {"similarity": 0.8, "distance": {"total": 0.2}}, "coverage_flags": {"unseen_logical_type": False, "unseen_map_kind": False}},
            ],
        }
        taskwise = {
            "schema": "pearl_equal_new_task_budget", "taskbook_hash": "frozen", "split": "meta_validation",
            "budgets": {"5": {"tasks": {
                "positive_a": {"support_environment_steps": 57, "pearl_valid_critical_strict_rate": 0.8, "scratch_sac_mean": 0.2, "pooled_finetune_sac_mean": 0.3},
                "negative": {"support_environment_steps": 57, "pearl_valid_critical_strict_rate": 0.1, "scratch_sac_mean": 0.2, "pooled_finetune_sac_mean": 0.3},
                "positive_b": {"support_environment_steps": 57, "pearl_valid_critical_strict_rate": 0.7, "scratch_sac_mean": 0.2, "pooled_finetune_sac_mean": 0.3},
            }}},
        }
        report = calibration_report(descriptor, taskwise, shot=5, minimum_independent_tasks=3)
        self.assertEqual(report["status"], "calibrated_validation_only")
        self.assertEqual(report["leave_one_task_out"]["evaluable_task_count"], 2)
        self.assertEqual(report["leave_one_task_out"]["skipped_task_ids"], ["negative"])

    def test_transferability_calibrates_and_enforces_support_only_uncertainty_threshold(self):
        task_ids = ["positive_a", "positive_b", "negative_a", "negative_b"]
        descriptor = {
            "schema": "logical_merge_transferability_report_v1", "taskbook_hash": "frozen",
            "candidate_split": "meta_validation", "uses_hidden_rules": False,
            "candidates": [{"task_id": task_id, "nearest_meta_train": {"similarity": 0.9, "distance": {"total": 0.1}}, "coverage_flags": {"unseen_logical_type": False, "unseen_map_kind": False}} for task_id in task_ids],
        }
        taskwise = {"schema": "pearl_equal_new_task_budget", "taskbook_hash": "frozen", "split": "meta_validation", "budgets": {"5": {"tasks": {
            "positive_a": {"support_environment_steps": 50, "pearl_valid_critical_strict_rate": 0.8, "scratch_sac_mean": 0.2, "pooled_finetune_sac_mean": 0.3},
            "positive_b": {"support_environment_steps": 50, "pearl_valid_critical_strict_rate": 0.7, "scratch_sac_mean": 0.2, "pooled_finetune_sac_mean": 0.3},
            "negative_a": {"support_environment_steps": 50, "pearl_valid_critical_strict_rate": 0.1, "scratch_sac_mean": 0.2, "pooled_finetune_sac_mean": 0.3},
            "negative_b": {"support_environment_steps": 50, "pearl_valid_critical_strict_rate": 0.1, "scratch_sac_mean": 0.2, "pooled_finetune_sac_mean": 0.3},
        }}}}
        posterior = {
            "schema": "logical_merge_support_posterior_diagnostic_v1", "taskbook_hash": "frozen", "split": "meta_validation",
            "uses_query_cases": False, "no_gradient_adaptation": True,
            "tasks": {task_id: {"5": {"posterior_variance": [[value, value]]}} for task_id, value in {"positive_a": 0.1, "positive_b": 0.2, "negative_a": 0.8, "negative_b": 0.9}.items()},
        }
        report = calibration_report(descriptor, taskwise, shot=5, minimum_independent_tasks=4, posterior_audit=posterior)
        self.assertEqual(report["status"], "calibrated_validation_only")
        self.assertIsNotNone(report["uncertainty_threshold"])
        self.assertEqual(report["posterior_uncertainty_input"], "support_only_posterior_variance")
        candidate = dict(descriptor) | {"candidate_split": "meta_test_logical", "candidates": [descriptor["candidates"][0], descriptor["candidates"][2]]}
        deployed_posterior = dict(posterior) | {"split": "meta_test_logical", "tasks": {"positive_a": posterior["tasks"]["positive_a"], "negative_a": posterior["tasks"]["negative_a"]}}
        without = transferability_decision_report(candidate, report)
        self.assertEqual(without["status"], "deferred_missing_posterior_uncertainty")
        decision = transferability_decision_report(candidate, report, posterior_audit=deployed_posterior)
        self.assertTrue(decision["decisions"][0]["allow_meta_adaptation"])
        self.assertFalse(decision["decisions"][1]["allow_meta_adaptation"])

    def test_validation_freeze_binds_validation_policies_to_one_checkpoint_before_holdout(self):
        policies = ["fixed", "random", "initial_condition_diversity", "posterior_action_disagreement"]
        evaluations = {
            policy: {
                "split": "meta_validation", "support_selection": policy, "no_gradient_adaptation": True,
                "parameter_hash_before": "unchanged", "parameter_hash_after": "unchanged",
                "provenance": {"taskbook_hash": "frozen", "checkpoint_hash": "checkpoint"},
            }
            for policy in policies
        }
        budgets = {
            policy: {
                "schema": "pearl_equal_new_task_budget", "taskbook_hash": "frozen", "split": "meta_validation",
                "protocol": {"support_selection": policy},
            }
            for policy in policies
        }
        representation = {
            "schema": "logical_merge_task_representation_audit_v1", "split": "meta_validation",
            "uses_query_cases": False, "parameter_hash_before": "same", "parameter_hash_after": "same",
            "provenance": {"taskbook_hash": "frozen"},
        }
        calibration = {
            "schema": "logical_merge_transferability_calibration_v1", "taskbook_hash": "frozen",
            "split": "meta_validation", "status": "insufficient_validation_evidence_no_threshold",
        }
        frozen = freeze_validation_protocol(
            taskbook_hash="frozen", evaluations=evaluations, required_policies=policies,
            equal_budget=budgets, representation_audits=[representation], calibration=calibration,
        )
        self.assertEqual(frozen["status"], "validation_frozen_for_holdout")
        self.assertFalse(frozen["uses_holdout_query_results"])
        verify_validation_freeze(frozen, taskbook_hash="frozen", checkpoint_hash="checkpoint")
        with self.assertRaises(ValueError):
            verify_validation_freeze(frozen, taskbook_hash="changed", checkpoint_hash="checkpoint")
        evaluations["random"]["provenance"]["checkpoint_hash"] = "other"
        with self.assertRaises(ValueError):
            freeze_validation_protocol(taskbook_hash="frozen", evaluations=evaluations, required_policies=policies)

    def test_validation_freeze_selects_deterministic_mean_from_current_suite(self):
        selected = {
            "split": "meta_validation", "support_selection": "fixed",
            "no_gradient_adaptation": True,
            "parameter_hash_before": "same", "parameter_hash_after": "same",
            "provenance": {"taskbook_hash": "frozen", "checkpoint_hash": "checkpoint"},
        }
        suite = {
            "schema": "pearl_fewshot_evaluation_suite",
            "evaluation_regimes": {
                "validation_known_logical_type": {
                    "split": "meta_validation",
                    "query_modes": {"posterior_mean_deterministic": selected},
                },
            },
        }
        frozen = freeze_validation_protocol(
            taskbook_hash="frozen", evaluations={"fixed": suite},
            required_policies=["fixed"],
        )
        self.assertEqual(frozen["checkpoint_hash"], "checkpoint")

    def test_transferability_candidate_catalog_has_eight_disjoint_validation_tasks(self):
        expanded = extend_validation_catalog(self.config, [36.0, 44.0, 52.0])
        taskbook = build_taskbook(expanded)
        validation = taskbook["meta_validation"]
        self.assertEqual(len(validation), 8)
        self.assertEqual(len({task.geometry_id for task in validation}), 8)
        self.assertTrue(all(task.split == "meta_validation" for task in validation))
        self.assertTrue({"adversary_first", "sut_first"} <= {task.priority_spec["target_contact_entry_order"] for task in validation})

    def test_disentangled_agent_updates_semantic_heads_without_task_id_input(self):
        config = configure_disentangled_representation(self.config, enabled=True, latent_dims=[2, 2, 1], geometry_weight=0.1, interaction_weight=0.1, rule_weight=0.1)
        task = build_taskbook(config)["meta_train"][0]
        agent = PEARLAgent(37, 2, config, torch.device("cpu"))
        rows = [transition("representation") for _ in range(2)]
        metrics = agent.update([[rows]], [rows], [representation_target(task)])
        self.assertGreaterEqual(metrics["auxiliary_loss"], 0.0)
        self.assertIn("task_representation", agent.state_dict())
        self.assertTrue(agent.disentangled)

    def test_disentangled_decoder_uses_declared_blocks_and_interaction_fields(self):
        config = configure_disentangled_representation(self.config, enabled=True, latent_dims=[2, 2, 1], geometry_weight=0.1, interaction_weight=0.1, rule_weight=0.1)
        agent = PEARLAgent(37, 2, config, torch.device("cpu"))
        decoded = agent.decode_task_representation(torch.zeros((3, 5), dtype=torch.float32))
        self.assertEqual(tuple(decoded["geometry"].shape), (3, 5))
        self.assertEqual(tuple(decoded["interaction"].shape), (3, 3))
        self.assertEqual(tuple(decoded["entry_order_probability"].shape), (3, 1))
        self.assertEqual(INTERACTION_OBSERVATION_FIELDS, ("arrival_time_difference", "relative_route_speed", "ttc"))
        self.assertEqual(INTERACTION_OBSERVATION_INDEXES, (16, 18, 20))
        with self.assertRaises(ValueError):
            agent.decode_task_representation(torch.zeros((3, 4), dtype=torch.float32))

    def test_transferability_descriptor_hides_rules_by_default(self):
        taskbook = build_taskbook(self.config)
        task = taskbook["meta_test_logical"][0]
        cases = build_casebook(task, self.config)["test_support"]
        normal = task_descriptor(task, cases)
        oracle = task_descriptor(task, cases, include_hidden_rules=True)
        self.assertFalse(normal["uses_hidden_rules"])
        self.assertNotIn("oracle_hidden", normal["groups"]["rules"])
        self.assertIn("oracle_hidden", oracle["groups"]["rules"])
        self.assertTrue(all("test_support" in row["case_id"] for row in cases))

    def test_transferability_report_uses_support_only_and_flags_unseen_logical_type(self):
        taskbook = build_taskbook(self.config)
        casebooks = {task.task_id: build_casebook(task, self.config) for tasks in taskbook.values() for task in tasks}
        report = transferability_report(taskbook, casebooks, candidate_split="meta_test_logical")
        self.assertEqual(report["status"], "diagnostic_only_not_calibrated")
        self.assertTrue(all(row["descriptor"]["uses_query_cases"] is False for row in report["candidates"]))
        self.assertTrue(all(row["coverage_flags"]["unseen_logical_type"] for row in report["candidates"]))
        self.assertTrue(all(len(row["top_meta_train_neighbors"]) == 3 for row in report["candidates"]))


if __name__ == "__main__":
    unittest.main()
