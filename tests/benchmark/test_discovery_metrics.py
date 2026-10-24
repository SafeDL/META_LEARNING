from dataclasses import replace

import pytest

from benchmark.collision_metrics import collision_recall
from benchmark.scenarios import sample


def queries(scenes, labels, indices):
    return [{'index': i, 'feedback': {'scenario_id': scenes[i].scenario_id, 'collision': labels[i]}}
            for i in indices]


def test_recall_uses_full_failure_groups_and_safe_duplicate_is_not_a_discovery():
    scenes = sample('cut_in', 3, 'development')
    scenes[1] = replace(scenes[0], scenario_id='same_group_safe')
    labels = [True, False, True]
    result = collision_recall(scenes, labels, queries(scenes, labels, [1, 0, 2]), checkpoints=(1, 2, 3))
    assert result['pool_unique_collisions'] == 2
    assert result['recall_curve'] == [0, 0.5, 1]
    assert result['checkpoints']['2'] == {'unique_collisions': 1, 'collision_recall': 0.5}
    result = collision_recall(scenes, [True, True, True], queries(scenes, [True]*3, [0, 1, 2]))
    assert result['recall_curve'] == [0.5, 0.5, 1]


def test_zero_failure_is_undefined_and_all_failure_is_monotone():
    scenes = sample('cut_in', 5, 'development')
    result = collision_recall(scenes, [False]*5, queries(scenes, [False]*5, [2, 0]))
    assert result['collision_recall'] is None
    assert result['recall_curve'] == [None, None]
    result = collision_recall(scenes, [True]*5, queries(scenes, [True]*5, [2, 0, 1, 3, 4]))
    assert result['recall_curve'] == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_incomplete_denominator_and_wrong_query_labels_are_rejected():
    scenes = sample('cut_in', 2, 'development')
    with pytest.raises(ValueError, match='complete pool'):
        collision_recall(scenes, [True], [])
    with pytest.raises(ValueError, match='sealed execution pool'):
        collision_recall(scenes, [True, False], queries(scenes, [False, False], [0]))
