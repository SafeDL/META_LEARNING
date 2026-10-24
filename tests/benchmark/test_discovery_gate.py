import copy

import pytest

from benchmark.discovery import pilot_gate, summary
from benchmark.discovery_config import METHODS, development_tasks


def records():
    rows = []
    for task in development_tasks():
        empty = task['lineage'] == 'uct'
        for method in METHODS:
            count = (50 if method == 'source_local' else 44 if method == 'source_global'
                     else 41 if method == 'fsbo' else 40)
            count = 0 if empty else count
            rows.append({**task, 'method': method, 'queries': 128,
                         'pool_unique_collisions': 0 if empty else 100,
                         'unique_collisions': count, 'collision_recall': None if empty else count/100,
                         'discovery_auc': count / 2, 'selection_s': 1, 'adaptation_s': 1})
    return rows


def test_gate_matches_fixed_matrix_and_preserves_zero_denominators():
    result = pilot_gate(records())
    assert result['status'] == 'passed'
    assert result['strongest_baseline'] == 'fsbo'
    assert result['summary']['source_local']['collision_recall']['valid_lineages'] == 3
    assert result['summary']['source_local']['unique_collisions']['valid_lineages'] == 4
    assert all(c['independent_lineages'] == (3 if c['metric'] == 'collision_recall' else 4)
               for c in result['comparisons'])


def test_failed_gate_cannot_be_promoted_by_averaging_zero_failures_as_full_recall():
    rows = records()
    for row in rows:
        if row['method'] == 'source_local' and row['collision_recall'] is not None:
            row.update(unique_collisions=42, collision_recall=0.42)
    result = pilot_gate(rows)
    assert result['status'] == 'not_passed'
    assert not result['conditions']['recall_gain_at_least_5pp']
    assert not result['conditions']['local_recall_exceeds_global']
    assert result['next_stage'] == 'report'


def test_gate_rejects_missing_runs_unmatched_denominators_and_partial_budgets():
    rows = records()
    with pytest.raises(ValueError, match='440-run'):
        pilot_gate(rows[:-1])
    altered = copy.deepcopy(rows)
    altered[0]['pool_unique_collisions'] = 99
    with pytest.raises(ValueError, match='same sealed denominator'):
        pilot_gate(altered)
    altered[0]['queries'] = 127
    with pytest.raises(ValueError, match='partial paid budget'):
        pilot_gate(altered)


def test_lineages_have_equal_weight_and_null_is_not_zero():
    rows = [{'lineage': 'a', 'value': 1.0}] * 10 + [{'lineage': 'b', 'value': 0.0},
                                                     {'lineage': 'c', 'value': None}]
    assert summary(rows, 'value') == {'mean': 0.5, 'by_lineage': {'a': 1.0, 'b': 0.0},
                                      'valid_lineages': 2, 'valid_runs': 11}
