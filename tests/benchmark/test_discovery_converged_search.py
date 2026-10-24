import copy

import numpy as np
import pytest
import torch

from benchmark import discovery_converged_search as study
from benchmark.discovery_config import development_tasks, write_json
from benchmark.discovery_converged_search_audit import probabilities_numpy
from methods.gcrt.converged_response import component_probabilities


def records():
    runs, previous = [], []
    counts = {'converged_local': 99, 'converged_global': 98,
              'converged_target': 94, 'converged_dual_target': 95, 'physical_local': 97}
    for task in development_tasks():
        empty = task['lineage'] == 'uct'
        for method in (*study.METHODS, *study.REFERENCES):
            count = counts.get(method, 96)
            row = {**task, 'method': method, 'queries': 128, 'pool_unique_collisions': 0 if empty else 100,
                'unique_collisions': 0 if empty else count, 'collision_recall': None if empty else count/100,
                'discovery_auc': 0 if empty else count/2, 'selection_s': 1., 'adaptation_s': 1.}
            (runs if method in study.METHODS else previous).append(row)
    return runs, previous


@pytest.fixture(autouse=True)
def bootstrap_stub(monkeypatch):
    monkeypatch.setattr(study, 'paired_lineage_bootstrap',
                        lambda rows, main, control, metric: {'method': main, 'baseline': control, 'metric': metric})


def test_design_keeps_all_strong_references_and_both_proper_target_controls():
    runs, previous = records()
    assert len(study.REFERENCES) == len(set(study.REFERENCES)) == 34
    assert len(runs) == 160 and len(previous) == 1360
    result = study.gate(runs, previous)
    assert result['status'] == 'passed'
    assert result['strongest_baseline'] == 'physical_local'
    assert result['remaining_miss_reduction'] == pytest.approx(2/3)
    for row in runs:
        if row['method'] == 'converged_dual_target' and row['collision_recall'] is not None:
            row.update(unique_collisions=100, collision_recall=1.)
    result = study.gate(runs, previous)
    assert result['strongest_baseline'] == 'converged_dual_target'
    assert result['remaining_miss_reduction'] is None and result['status'] == 'not_passed'


def test_tied_local_global_does_not_validate_local_mechanism():
    runs, previous = records()
    for row in runs:
        if row['method'] == 'converged_global' and row['collision_recall'] is not None:
            row.update(unique_collisions=99, collision_recall=.99)
    result = study.gate(runs, previous)
    assert result['conditions']['remaining_misses_reduced_by_half']
    assert not result['conditions']['local_recall_exceeds_global']
    assert result['status'] == 'not_passed'


def test_missing_previous_or_new_task_cannot_enter_admission():
    runs, previous = records()
    with pytest.raises(ValueError, match='40 paired tasks'):
        study.gate(runs[:-1], previous)
    with pytest.raises(ValueError, match='40 paired tasks'):
        study.gate(runs, previous[:-1])
    changed = copy.deepcopy(runs)
    changed[0].update(pool_unique_collisions=200, unique_collisions=198, collision_recall=.99)
    with pytest.raises(ValueError, match='budget, deduplicated pool or recall'):
        study.gate(changed, previous)


def test_two_positive_lineages_needs_two_lineages_with_headroom():
    _, previous = records()
    result = study.feasibility(previous)
    assert result['attainable_necessary_conditions']
    for row in previous:
        if row['method'] == 'physical_local' and row['lineage'] != 'ppo' and row['collision_recall'] is not None:
            row.update(unique_collisions=100, collision_recall=1.)
    result = study.feasibility(previous)
    assert result['lineages_with_recall_headroom'] == ['ppo']
    assert not result['attainable_necessary_conditions']


def test_freeze_waits_for_scaml_complete_independent_results(monkeypatch, tmp_path):
    monkeypatch.setattr(study.scaml, 'REPORT', tmp_path / 'scaml')
    monkeypatch.setattr(study, 'REPORT', tmp_path / 'converged')
    for name, result in (('run_index.json', {'status': 'incomplete'}),
                         ('audit/report.json', {'status': 'incomplete'}),
                         ('report.json', {'status': 'incomplete'}),
                         ('audit/negative_controls.json', {'status': 'incomplete'})):
        write_json(study.scaml.REPORT / name, result)
    with pytest.raises(ValueError, match='Complete all ScaML conditions'):
        study.freeze()
    assert not study.REPORT.exists()


def test_final_report_refuses_partial_independent_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'REPORT', tmp_path)
    monkeypatch.setattr(study, 'protocol', lambda: {'frozen': True})
    write_json(tmp_path / 'run_index.json', {'status': 'complete', 'protocol': {'frozen': True}})
    write_json(tmp_path / 'audit/report.json', {'status': 'passed', 'runs': 159})
    write_json(tmp_path / 'audit/negative_controls.json', {'status': 'passed'})
    with pytest.raises(ValueError, match='all 160 converged-search conditions'):
        study.report()


def test_independent_numpy_audit_handles_legacy_oscillation_and_no_observations():
    means = np.array([[-5., -4., -3.], [1., 2., 3.]])
    covariance = np.array([[[100., 25., 10.], [25., 10., 2.], [10., 2., 5.]], np.eye(3)*.01])
    for paid, labels in (([], []), ([0], [True])):
        independent = probabilities_numpy(means, covariance, paid, labels)
        actual, info = component_probabilities(torch.tensor(means), torch.tensor(covariance), paid, labels)
        np.testing.assert_allclose(actual, independent, atol=2e-9, rtol=2e-9)
        assert all(i['gradient_max'] <= 1e-10 for i in info)
