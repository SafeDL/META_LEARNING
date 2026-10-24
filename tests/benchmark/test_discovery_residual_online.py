import copy

import pytest

from benchmark.discovery_config import development_tasks
from benchmark.discovery_residual_online import METHODS, PREVIOUS, freeze, gate


def records():
    runs, previous = [], []
    counts = {'residual_local': 99, 'residual_global': 98, 'residual_unit_local': 99,
              'residual_unit_global': 99, 'residual_target': 94, 'residual_dual_target': 95,
              'physical_local': 97}
    for task in development_tasks():
        empty = task['lineage'] == 'uct'
        for method in (*METHODS, *PREVIOUS):
            count = counts.get(method, 96)
            row = {**task, 'method': method, 'queries': 128, 'pool_unique_collisions': 0 if empty else 100,
                   'unique_collisions': 0 if empty else count, 'collision_recall': None if empty else count/100,
                   'discovery_auc': 0 if empty else count/2, 'selection_s': 1., 'adaptation_s': 1.}
            (runs if method in METHODS else previous).append(row)
    return runs, previous


@pytest.fixture(autouse=True)
def bootstrap_stub(monkeypatch):
    monkeypatch.setattr('benchmark.discovery_residual_online.paired_lineage_bootstrap',
                        lambda rows, main, control, metric: {'method': main, 'baseline': control, 'metric': metric})


def test_all_previous_and_both_target_controls_enter_reference_selection():
    runs, previous = records()
    result = gate(runs, previous)
    assert result['status'] == 'passed'
    assert result['strongest_baseline'] == 'physical_local'
    assert result['required_recall'] == pytest.approx(.985)
    assert result['remaining_miss_reduction'] == pytest.approx(2/3)
    for row in runs:
        if row['method'] == 'residual_dual_target' and row['collision_recall'] is not None:
            row.update(unique_collisions=100, collision_recall=1.)
    result = gate(runs, previous)
    assert result['strongest_baseline'] == 'residual_dual_target'
    assert result['status'] == 'not_passed'
    assert not result['conditions']['recall_exceeds_same_information_dual_target']


@pytest.mark.parametrize('control', ['residual_global', 'residual_unit_local', 'residual_unit_global'])
def test_matched_mechanism_controls_cannot_be_worse_than_the_main(control):
    runs, previous = records()
    for row in runs:
        if row['method'] == control and row['collision_recall'] is not None:
            row.update(unique_collisions=100, collision_recall=1.)
    assert gate(runs, previous)['status'] == 'not_passed'


def test_tied_local_and_global_recall_does_not_validate_local_reliability():
    runs, previous = records()
    for row in runs:
        if row['method'] == 'residual_global' and row['collision_recall'] is not None:
            row.update(unique_collisions=99, collision_recall=.99)
    result = gate(runs, previous)
    assert result['conditions']['remaining_misses_reduced_by_half']
    assert not result['conditions']['local_recall_exceeds_global']
    assert result['status'] == 'not_passed'


def test_exact_pairing_budget_and_full_denominators_are_required():
    runs, previous = records()
    with pytest.raises(ValueError, match='240 complete'):
        gate(runs[:-1], previous)
    with pytest.raises(ValueError, match='240 complete'):
        gate(runs[:-1]+[runs[0]], previous)
    with pytest.raises(ValueError, match='22 prior'):
        gate(runs, previous[:-1])
    changed = copy.deepcopy(runs)
    changed[0]['pool_unique_collisions'] += 1
    with pytest.raises(ValueError, match='denominator'):
        gate(changed, previous)
    changed = copy.deepcopy(runs)
    changed[0]['queries'] = 127
    with pytest.raises(ValueError, match='budget'):
        gate(changed, previous)
    changed = copy.deepcopy(runs)
    changed[0]['collision_recall'] = .5
    with pytest.raises(ValueError, match='count or recall'):
        gate(changed, previous)


def test_perfect_reference_and_all_zero_pools_are_not_success():
    runs, previous = records()
    for row in previous:
        if row['method'] == 'physical_local' and row['collision_recall'] is not None:
            row.update(unique_collisions=100, collision_recall=1.)
    assert gate(runs, previous)['remaining_miss_reduction'] is None
    for row in runs+previous:
        row.update(pool_unique_collisions=0, unique_collisions=0, collision_recall=None, discovery_auc=0)
    result = gate(runs, previous)
    assert result['required_recall'] is None and result['remaining_miss_reduction'] is None
    assert result['status'] == 'not_passed'


def test_freeze_does_not_accept_failed_historical_selection_audit(monkeypatch, tmp_path):
    from benchmark import discovery_residual_online as module
    real_read = module.read_json
    monkeypatch.setattr(module, 'REPORT', tmp_path)
    monkeypatch.setattr(module, 'read_json', lambda p: {'status': 'failed'}
                        if p == module.LEARNING/'selection_audit.json' else real_read(p))
    with pytest.raises(ValueError, match='independently verify'):
        freeze()
    assert not (tmp_path/'protocol.json').exists()
