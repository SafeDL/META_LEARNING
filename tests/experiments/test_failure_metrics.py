from dataclasses import replace

import numpy as np
import pytest

from experiments.failure_metrics import (
    aggregate,
    equal_count_distances,
    parameter_distances,
    summarize,
)
from experiments.scenarios import DOMAINS, Domain
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario


def scenes_at(*fractions):
    return [
        Scenario(
            f"scene_{i}",
            "car_following",
            {
                name: low + f * (high - low)
                for name, (low, high) in DOMAINS["car_following"].bounds.items()
            },
        )
        for i, f in enumerate(fractions)
    ]


def measured_summary(scenes, labels, order, initial_count=1):
    queries, traces, ledger = [], [], []
    for index in order:
        feedback = QueryFeedback(
            scenes[index].scenario_id,
            labels[index],
            SafetyMeasurements(0, 0, rss_margin=-0.5),
            2.0,
        ).to_dict()
        queries.append(
            {
                "index": index,
                "feedback": feedback,
                "prequery_probability": 0.5,
                "selection_s": 0.2,
                "adaptation_s": 0.3,
            }
        )
        traces.append([{"collision": labels[index]}])
        ledger.append(
            {
                "index": index,
                "feedback": feedback,
                "status": "complete",
                "rss_compute_s": 0.1,
                "wall_s": 0.01,
            }
        )
    return summarize(
        scenes, labels, queries, traces, initial_count=initial_count, query_ledger=ledger
    )


def test_manual_truth_distances_and_first_discovery_order():
    scenes = scenes_at(0, 0.5, 1, 0.25)
    result = measured_summary(scenes, [True, True, True, False], [0, 3, 2, 1])
    assert result["discovery_curve"] == [1, 1, 2, 3]
    assert result["post_initial_curve"] == [0, 0, 1, 2]
    assert result["post_initial_collisions"] == 2
    assert result["discovery_auc"] == 1.75
    assert result["recall_curve"] == pytest.approx([1 / 3, 1 / 3, 2 / 3, 1])
    assert result["pool_unique_collisions"] == 3
    assert result["discovered_ids"] == ["scene_0", "scene_2", "scene_1"]
    assert result["distances_by_discovery_count"]["pair_distance"] == [None, 1, 2 / 3]
    assert result["distances_by_discovery_count"]["nearest_neighbor_distance"] == [None, 1, 0.5]
    assert result["distance_curves"]["pair_distance"] == [None, None, 1, 2 / 3]
    assert result["queries_to_recall"] == {"0.25": 1, "0.5": 3, "0.75": 4, "1.0": 4}
    assert result["brier"] == 0.25
    assert result["collision_nll"] == pytest.approx(np.log(2))
    assert result["negative_rss_executions"] == 4
    assert result["unique_collisions"] == 3
    assert result["equivalent_testing_curve_s"] == pytest.approx([2.6, 5.2, 7.8, 10.4])
    assert result["replay_wall_s"] == pytest.approx(0.04)


def test_nearby_ids_count_separately_but_repeated_execution_does_not():
    scenes = scenes_at(0, 0.001, 1)
    assert scenes[0].coordinate() != scenes[1].coordinate()
    result = measured_summary(scenes, [True, True, True], [0, 0, 1])
    assert result["discovery_curve"] == [1, 1, 2]
    assert result["collision_executions"] == 3
    assert result["unique_collisions"] == 2
    assert result["pool_unique_collisions"] == 3
    assert result["collision_recall"] == pytest.approx(2 / 3)
    assert result["pair_distance"] == pytest.approx(0.001)


@pytest.mark.parametrize("labels", [[False, False], [True, False]])
def test_undefined_distances_and_empty_failure_library_remain_missing(labels):
    scenes = scenes_at(0, 1)
    result = measured_summary(scenes, labels, [0, 1])
    assert result["pair_distance"] is None
    assert result["nearest_neighbor_distance"] is None
    assert equal_count_distances([result])["counts"] == []
    if not any(labels):
        assert result["collision_recall"] is None
        assert result["recall_curve"] == [None, None]
        assert result["first_collision_query"] is None
        assert result["no_collision_within_budget"]
        assert result["negative_rss_executions"] == 2
        assert result["unique_collisions"] == 0
        assert aggregate([result])["pooled_recall"] is None


def test_equal_count_uses_first_k_and_reports_no_comparable_pair():
    scenes = scenes_at(0, 0.2, 1)
    first = measured_summary(scenes, [True] * 3, [0, 1, 2])
    second = measured_summary(scenes, [True] * 3, [0, 2])
    common = equal_count_distances([first, second])
    assert common["counts"] == [2]
    assert common["methods"][0]["pair_distance"] == pytest.approx([0.2])
    assert common["methods"][1]["pair_distance"] == [1]
    one = measured_summary(scenes, [True] * 3, [0])
    assert equal_count_distances([first, one])["counts"] == []


def test_normalization_is_invariant_to_physical_units(monkeypatch):
    scenes = scenes_at(0.1, 0.7, 1)
    original = parameter_distances(scenes, scenes)
    domain = DOMAINS["car_following"]
    scaled = [replace(s, parameters={k: 1000 * v for k, v in s.parameters.items()}) for s in scenes]
    monkeypatch.setitem(
        DOMAINS,
        "car_following",
        Domain(
            {k: (1000 * a, 1000 * b) for k, (a, b) in domain.bounds.items()},
            {k: 1000 * v for k, v in domain.quantization.items()},
            {},
        ),
    )
    np.testing.assert_allclose(parameter_distances(scaled, scaled), original)


def test_categorical_attribute_has_one_vote_not_one_hot_width():
    domain = DOMAINS["cut_in_interaction"]
    parameters = {k: (a + b) / 2 for k, (a, b) in domain.bounds.items()}
    first = Scenario(
        "a", "cut_in_interaction", parameters, {"side": "left", "adjacent_space": "open"}
    )
    second = replace(first, scenario_id="b", categories={"side": "right", "adjacent_space": "open"})
    assert parameter_distances([first], [second])[0, 0] == pytest.approx(1 / 7)
    with pytest.raises(ValueError, match="one functional"):
        parameter_distances([first], scenes_at(0))


def test_aggregate_keeps_equal_task_and_pooled_recall_distinct():
    scenes = scenes_at(0, 0.5, 1)
    first = measured_summary(scenes, [True, False, False], [0])
    second = measured_summary(scenes, [True] * 3, [0])
    empty = measured_summary(scenes, [False] * 3, [0])
    result = aggregate([first, second, empty])
    assert result["tasks"] == 3
    assert result["zero_failure_tasks"] == 1
    assert result["equal_task_means"]["collision_recall"] == {"mean": 2 / 3, "tasks": 2}
    assert result["pooled_recall"] == 0.5
    assert result["distance_undefined_tasks"] == 3


def test_wrong_truth_identity_and_duplicate_ids_are_rejected():
    scenes = scenes_at(0, 1)
    feedback = QueryFeedback(scenes[0].scenario_id, True, SafetyMeasurements(0, 0), 0).to_dict()
    query = {"index": 0, "feedback": feedback}
    record = {"index": 0, "feedback": feedback, "status": "complete", "wall_s": 0}
    with pytest.raises(ValueError, match="candidate truth"):
        summarize(
            scenes,
            [False, True],
            [query],
            [[{"collision": True}]],
            initial_count=1,
            query_ledger=[record],
        )
    with pytest.raises(ValueError, match="actual execution"):
        summarize(
            scenes,
            [True, True],
            [query],
            [[{"collision": False}]],
            initial_count=1,
            query_ledger=[record],
        )
    with pytest.raises(ValueError, match="IDs must be unique"):
        measured_summary([scenes[0], scenes[0]], [True, True], [0])
