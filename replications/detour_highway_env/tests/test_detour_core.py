from __future__ import annotations

import numpy as np
import pytest

from replications.detour_highway_env.detour.features import RoadCurvatureFeatures, encode_scenarios
from replications.detour_highway_env.detour.retrieve import Retriever
from replications.detour_highway_env.detour.selector import prioritize, select
from replications.detour_highway_env.detour.tree import build_tree
from replications.detour_highway_env.detour.contracts import ScenarioSpec
from replications.detour_highway_env.detour.history import within_sut_split


def _tree(failed: list[bool]):
    history = np.array([[0.0], [1.0], [8.0], [9.0]], dtype=float)
    candidates = np.array([[0.1], [1.1], [8.1], [9.1]], dtype=float)
    return build_tree(history, candidates, np.asarray(failed, dtype=bool))


def test_road_feature_reconstruction_preserves_straight_polyline():
    points = np.column_stack((np.arange(8.0), np.zeros(8)))
    features = RoadCurvatureFeatures.from_points(points)
    assert np.allclose(features.reconstruct(points[0]), points)
    assert np.allclose(features.curvatures, 0.0)


def test_road_feature_compression_handles_circle_and_s_bend_without_identity_loss():
    theta = np.linspace(0.0, np.pi / 2.0, 81)
    circle = np.column_stack((20.0 * np.sin(theta), 20.0 * (1.0 - np.cos(theta))))
    s_x = np.linspace(0.0, 60.0, 121)
    s_bend = np.column_stack((s_x, 4.0 * np.sin(s_x / 12.0)))
    for points in (circle, s_bend):
        compact = RoadCurvatureFeatures.from_points(points).compressed(12)
        rebuilt = compact.reconstruct(points[0])
        assert len(compact.lengths) == 12
        assert np.isfinite(rebuilt).all()
        assert np.linalg.norm(rebuilt[-1] - points[-1]) < 8.0


def test_scenario_mode_is_one_hot_not_an_ordinal_value():
    scenarios = (ScenarioSpec("a", 5.0, -8.0,
                              "cutin_braking"), ScenarioSpec("b", 40.0, 2.0, "fast_intrusion"))
    encoded = encode_scenarios(scenarios)
    assert encoded.shape == (2, 7)
    assert np.isclose(np.linalg.norm(encoded[0, 2:] - encoded[1, 2:]), 1.0)


def test_distances_are_symmetric_and_static_counts_do_not_change():
    tree = _tree([True, False, False, True])
    before = [(row["executed_count"], row["fail_count"]) for row in tree.node_rows()]
    order, _traces, _decisions = prioritize(Retriever(tree, seed=3), 3)
    assert len(order) == len(set(order)) == 3
    assert np.allclose(tree.distances, tree.distances.T)
    assert before == [(row["executed_count"], row["fail_count"]) for row in tree.node_rows()]


def test_recorded_branch_probabilities_follow_failure_ratios():
    tree = _tree([True, False, False, True])
    trace = Retriever(tree, seed=5).retrieve()
    assert trace is not None
    for depth, probabilities in enumerate(trace.branch_probabilities):
        assert np.isclose(sum(probabilities.values()), 1.0)
        chosen_child = tree.nodes[trace.path_node_ids[depth + 1]]
        expected = tree.node_fail_count(chosen_child) / chosen_child.executed_count
        raw_total = sum(
            tree.node_fail_count(tree.nodes[int(node_id)]) /
            tree.nodes[int(node_id)].executed_count for node_id in probabilities)
        assert np.isclose(probabilities[str(chosen_child.node_id)], expected / raw_total)
    assert all({"node_id", "executed_count", "fail_count", "selectable_count"} <= set(row)
               for row in trace.node_counts)


def test_no_failure_is_explicit_seeded_uniform_fallback():
    tree = _tree([False, False, False, False])
    _order, traces, _decisions = prioritize(Retriever(tree, seed=7), 2)
    assert all(trace.selected_reason == "fallback_no_failure" for trace in traces)


def test_all_failure_and_duplicate_coordinates_keep_execution_identity():
    history = np.array([[0.0], [0.0], [1.0]], dtype=float)
    candidates = np.array([[0.0], [0.0], [1.0]], dtype=float)
    tree = build_tree(history, candidates, np.array([True, True, True]))
    order, traces, _decisions = prioritize(Retriever(tree, seed=2), 3)
    assert len(order) == len(set(order)) == 3
    assert all(trace.selected_reason != "fallback_no_failure" for trace in traces)
    assert tree.history_count == 3 and tree.candidate_count == 3


def test_empty_candidate_set_is_rejected_and_insufficient_neighbors_cannot_early_stop():
    with pytest.raises(ValueError, match="non-empty"):
        build_tree(np.array([[0.0]]), np.empty((0, 1)), np.array([False]))
    tree = _tree([False, False, False, False])
    order, _traces, decisions = select(Retriever(tree, seed=1),
                                       min_count=1,
                                       max_count=3,
                                       m_neighbors=5,
                                       w_streak=1)
    assert len(order) == 3
    assert decisions[-1].reason == "max_count_reached"
    assert all(decision.safe_neighbor is None for decision in decisions[:-1])


def test_within_sut_split_is_deterministic_and_disjoint_without_using_outcomes_for_membership():
    class Bank:
        anchors = np.array([[5.0, -2.0], [8.0, -1.0], [12.0, 0.0], [16.0, 1.0]])
        modes = np.array(["single", "single", "fast_intrusion", "cutin_braking"])
        sut_names = ("A", )
        collisions = np.array([[False, True, False, True]])
        near_misses = np.zeros((1, 4), dtype=bool)

        @staticmethod
        def index_of(_name):
            return 0

    first_history, first_candidates, first_indices = within_sut_split(Bank(), "A")
    second_history, second_candidates, second_indices = within_sut_split(Bank(), "A")
    assert tuple(item.scenario.scenario_id
                 for item in first_history) == tuple(item.scenario.scenario_id
                                                     for item in second_history)
    assert tuple(item.scenario_id
                 for item in first_candidates) == tuple(item.scenario_id
                                                        for item in second_candidates)
    assert np.array_equal(first_indices, second_indices)
    assert {item.scenario.scenario_id
            for item in first_history}.isdisjoint({item.scenario_id
                                                   for item in first_candidates})


def test_selection_stops_only_after_minimum_and_safe_streak():
    tree = _tree([False, False, False, False])
    order, _traces, decisions = select(Retriever(tree, seed=1),
                                       min_count=2,
                                       max_count=4,
                                       m_neighbors=2,
                                       w_streak=2)
    assert len(order) == 2
    assert decisions[-1].reason == "safe_neighbor_streak"
