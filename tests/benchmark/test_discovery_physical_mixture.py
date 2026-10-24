import copy

import pytest

from benchmark.discovery_config import development_tasks
from benchmark.discovery_physical_mixture import METHODS, PREVIOUS, freeze, gate, protocol


def records():
    runs,previous = [],[]
    for task in development_tasks():
        empty = task['lineage'] == 'uct'
        for method in (*METHODS,*PREVIOUS):
            count = {'physical_global':99,'physical_local':98,'physical_mean':98,
                     'physical_target':97,'physical_adaptive':98,'joint_global':96}.get(method,95)
            row = {**task,'method':method,'queries':128,'pool_unique_collisions':0 if empty else 100,
                   'unique_collisions':0 if empty else count,'collision_recall':None if empty else count/100,
                   'discovery_auc':0 if empty else count/2,'selection_s':1,'adaptation_s':1}
            (runs if method in METHODS else previous).append(row)
    return runs,previous


def patch_references(monkeypatch,previous):
    monkeypatch.setattr('benchmark.discovery_physical_mixture.previous_rows',lambda:previous)
    monkeypatch.setattr('benchmark.discovery_physical_mixture.paired_lineage_bootstrap',
                        lambda rows,main,control,metric:{'method':main,'baseline':control,'metric':metric})


def test_physical_gate_includes_matched_target_and_all_previous_joint_variants(monkeypatch):
    runs,previous = records()
    patch_references(monkeypatch,previous)
    result = gate(runs)
    assert result['status'] == 'passed'
    assert result['strongest_baseline'] == 'physical_target'
    assert result['required_recall'] == pytest.approx(.985)
    assert result['remaining_miss_reduction'] == pytest.approx(2/3)
    assert result['local_reliability_claimed_as_necessary'] is False
    for row in previous:
        if row['method'] == 'joint_global' and row['collision_recall'] is not None:
            row.update(unique_collisions=98,collision_recall=.98)
    result = gate(runs)
    assert result['strongest_baseline'] == 'joint_global'
    assert result['required_recall'] == pytest.approx(.99)
    assert result['remaining_miss_reduction'] == pytest.approx(.5)
    for row in runs:
        if row['method'] == 'physical_fsbo' and row['collision_recall'] is not None:
            row.update(unique_collisions=99,collision_recall=.99)
    result = gate(runs)
    assert result['strongest_baseline'] == 'physical_fsbo'
    assert result['status'] == 'not_passed'


def test_fixed_mass_requires_an_effect_beyond_the_same_initial_adaptive_mass(monkeypatch):
    runs,previous = records()
    patch_references(monkeypatch,previous)
    for row in runs:
        if row['method'] == 'physical_adaptive' and row['collision_recall'] is not None:
            row.update(unique_collisions=99,collision_recall=.99)
    result = gate(runs)
    assert result['conditions']['remaining_misses_reduced_by_half'] is True
    assert result['conditions']['fixed_mass_recall_exceeds_adaptive'] is False
    assert result['status'] == 'not_passed'


def test_physical_gate_rejects_missing_pairs_changed_denominators_and_budget(monkeypatch):
    runs,previous = records()
    patch_references(monkeypatch,previous)
    with pytest.raises(ValueError,match='320-run'):
        gate(runs[:-1])
    previous.pop()
    with pytest.raises(ValueError,match='previous strong'):
        gate(runs)
    runs,previous = records()
    patch_references(monkeypatch,previous)
    changed = copy.deepcopy(runs)
    changed[0]['pool_unique_collisions'] += 1
    with pytest.raises(ValueError,match='denominators'):
        gate(changed)
    changed = copy.deepcopy(runs)
    changed[0]['queries'] = 127
    with pytest.raises(ValueError,match='budget'):
        gate(changed)


def test_perfect_reference_and_all_zero_collision_pools_cannot_pass(monkeypatch):
    runs,previous = records()
    patch_references(monkeypatch,previous)
    for row in runs:
        if row['method'] == 'physical_target' and row['collision_recall'] is not None:
            row.update(unique_collisions=100,collision_recall=1.)
    result = gate(runs)
    assert result['strongest_baseline'] == 'physical_target'
    assert result['remaining_miss_reduction'] is None
    assert result['required_recall'] == 1
    assert result['status'] == 'not_passed'
    for row in runs+previous:
        row.update(pool_unique_collisions=0,unique_collisions=0,collision_recall=None,discovery_auc=0)
    result = gate(runs)
    assert result['remaining_miss_reduction'] is None
    assert result['required_recall'] is None
    assert result['conditions']['remaining_misses_reduced_by_half'] is False
    assert result['status'] == 'not_passed'


def test_freeze_waits_for_the_complete_joint_matrix_and_its_independent_audit(monkeypatch,tmp_path):
    monkeypatch.setattr('benchmark.discovery_physical_mixture.joint.REPORT',tmp_path/'joint')
    monkeypatch.setattr('benchmark.discovery_physical_mixture.REPORT',tmp_path/'physical')
    with pytest.raises(ValueError,match='entire 280-run'):
        freeze()
    with pytest.raises(ValueError,match='entire 280-run'):
        protocol()
    assert not (tmp_path/'physical/protocol.json').exists()
