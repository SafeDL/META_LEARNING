"""A feasible gate cannot relabel the original failed experiment."""

import copy

import pytest

from benchmark.discovery_config import STRONG_BASELINES, development_tasks
from benchmark.discovery_recovery import METHODS, recovery_gate


def records():
    rows, baselines = [], []
    for task in development_tasks():
        empty = task['lineage'] == 'uct'
        for method in (*METHODS,*STRONG_BASELINES):
            count = (98 if method == 'response_adaptive' else
                     97 if method == 'response_fixed' else 95 if method == 'fsbo' else 94)
            row = {**task,'method':method,'queries':128,
                   'pool_unique_collisions':0 if empty else 100,
                   'unique_collisions':0 if empty else count,
                   'collision_recall':None if empty else count/100,
                   'discovery_auc':count/2,'selection_s':1,'adaptation_s':1}
            (rows if method in METHODS else baselines).append(row)
    return rows,baselines


def test_gate_uses_relative_remaining_misses_and_preserves_null_recall():
    result = recovery_gate(*records())
    assert result['status'] == 'passed'
    assert result['remaining_miss_reduction'] == pytest.approx(.6)
    assert result['required_recall'] == pytest.approx(.975)
    assert result['original_gate_changed'] is False
    assert result['summary']['response_adaptive']['collision_recall']['valid_lineages'] == 3
    assert result['summary']['response_adaptive']['unique_collisions']['valid_lineages'] == 4


def test_positive_absolute_gain_does_not_automatically_pass():
    rows,baselines = records()
    for row in rows:
        if row['method'] == 'response_adaptive' and row['collision_recall'] is not None:
            row.update(unique_collisions=97,collision_recall=.97)
    result = recovery_gate(rows,baselines)
    assert result['status'] == 'not_passed'
    assert result['remaining_miss_reduction'] == pytest.approx(.4)
    assert not result['conditions']['remaining_misses_reduced_by_half']
    assert not result['conditions']['adaptation_recall_exceeds_fixed']


def test_perfect_baseline_has_no_defined_remaining_miss_reduction():
    rows,baselines = records()
    for row in baselines:
        if row['method'] == 'fsbo' and row['collision_recall'] is not None:
            row.update(unique_collisions=100,collision_recall=1.)
    result = recovery_gate(rows,baselines)
    assert result['status'] == 'not_passed'
    assert result['remaining_miss_reduction'] is None


def test_gate_rejects_missing_pairs_different_denominators_and_unpaid_budget():
    rows,baselines = records()
    with pytest.raises(ValueError,match='80-run'):
        recovery_gate(rows[:-1],baselines)
    altered = copy.deepcopy(rows)
    altered[0]['pool_unique_collisions'] = 99
    with pytest.raises(ValueError,match='denominator'):
        recovery_gate(altered,baselines)
    altered[0]['queries'] = 127
    with pytest.raises(ValueError,match='budget'):
        recovery_gate(altered,baselines)
    with pytest.raises(ValueError,match='every registered'):
        recovery_gate(rows,baselines[:-1])
