import copy

import numpy as np
import pytest

from benchmark import discovery_rgpe as study
from benchmark.discovery_config import development_tasks, write_json
from benchmark.discovery_rgpe_audit import posterior_numpy


def rows_for(methods=study.METHODS):
    return [{**task, 'method': method, 'queries': 128, 'pool_unique_collisions': 20,
             'unique_collisions': 2, 'collision_recall': .1}
            for method in methods for task in development_tasks()]


def test_comparison_requires_each_exact_paired_task():
    rows = rows_for()
    assert len(rows) == 80
    study.validate_rows(rows)
    with pytest.raises(ValueError, match='40 complete paired tasks'):
        study.validate_rows(rows[:-1])
    duplicated = copy.deepcopy(rows)
    duplicated[-1] = duplicated[0]
    with pytest.raises(ValueError, match='without duplicates'):
        study.validate_rows(duplicated)


@pytest.mark.parametrize('changes', [
    {'queries': 127}, {'collision_recall': .11}, {'unique_collisions': 21},
    {'unique_collisions': 129, 'pool_unique_collisions': 200, 'collision_recall': 129 / 200},
    {'pool_unique_collisions': 21, 'collision_recall': 2 / 21},
    {'unique_collisions': 0, 'pool_unique_collisions': 0, 'collision_recall': 0.},
])
def test_invalid_budget_denominator_or_recall_cannot_enter_comparison(changes):
    rows = rows_for()
    rows[0].update(changes)
    with pytest.raises(ValueError, match='inconsistent budget, denominator or recall'):
        study.validate_rows(rows)


def test_zero_collision_tasks_remain_with_undefined_recall():
    rows = rows_for()
    task = rows[0]
    for row in rows:
        if all(row[k] == task[k] for k in ('target_id', 'scenario', 'algorithm_seed')):
            row.update(pool_unique_collisions=0, unique_collisions=0, collision_recall=None)
    study.validate_rows(rows)
    assert len(rows) == 80


def test_all_twenty_eight_prior_references_are_required():
    rows = rows_for(study.REFERENCES)
    assert len(study.REFERENCES) == len(set(study.REFERENCES)) == 28
    assert len(rows) == 1120
    study.validate_rows(rows, study.REFERENCES)
    with pytest.raises(ValueError, match='complete paired tasks'):
        study.validate_rows(rows[:-1], study.REFERENCES)


def test_freeze_does_not_start_before_current_full_audit(monkeypatch, tmp_path):
    foundation, destination = tmp_path / 'residual', tmp_path / 'rgpe'
    monkeypatch.setattr(study.residual, 'REPORT', foundation)
    monkeypatch.setattr(study, 'REPORT', destination)
    write_json(foundation / 'audit/report.json', {'status': 'available_completed_runs_checked', 'runs': 52})
    write_json(foundation / 'run_index.json', {'status': 'incomplete'})
    with pytest.raises(ValueError, match='Finish all residual controls'):
        study.freeze()
    assert not destination.exists()


def test_report_requires_complete_independent_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'REPORT', tmp_path)
    monkeypatch.setattr(study, 'protocol', lambda: {'fixed': True})
    write_json(tmp_path / 'run_index.json', {'status': 'complete', 'protocol': {'fixed': True}})
    write_json(tmp_path / 'audit/report.json', {'status': 'available_completed_runs_checked', 'runs': 79})
    with pytest.raises(ValueError, match='all 80 RGPE runs'):
        study.report()


def test_runtime_change_cannot_replace_registered_design(monkeypatch, tmp_path):
    foundation, destination = tmp_path / 'residual', tmp_path / 'rgpe'
    monkeypatch.setattr(study.residual, 'REPORT', foundation)
    monkeypatch.setattr(study, 'REPORT', destination)
    monkeypatch.setattr(study.residual, 'protocol', lambda: {'foundation': True})
    monkeypatch.setattr(study.residual, 'gate', lambda rows: {'status': 'not_passed'})
    monkeypatch.setattr(study, 'protocol', lambda: {'changed': True})
    monkeypatch.setattr(study, 'previous_rows', lambda: rows_for(study.REFERENCES))
    write_json(foundation / 'audit/report.json', {'status': 'passed', 'runs': 240,
               'independent_full_pool_choices': 720, 'equivalent_parameter_pairs': [None] * 40})
    write_json(foundation / 'run_index.json', {'status': 'complete',
               'protocol': {'foundation': True}, 'runs': []})
    write_json(foundation / 'gate.json', {'status': 'not_passed'})
    write_json(destination / 'design.json', {'changed': False})
    with pytest.raises(ValueError, match='differs from the registered design'):
        study.freeze()
    assert not (destination / 'protocol.json').exists()


def test_independent_precision_solver_obeys_binary_label_symmetry():
    mean = np.array([-.4, .8, -.7, .3])
    a = np.array([[1., .4, .2, .1], [.4, 1., .3, .2], [.2, .3, 1., .4], [.1, .2, .4, 1.]])
    covariance = a @ a.T
    support, labels = [0, 2, 3], np.array([0., 1., 1.])
    m, k = posterior_numpy(mean, covariance, support, labels)
    opposite_m, opposite_k = posterior_numpy(-mean, covariance, support, 1 - labels)
    np.testing.assert_allclose(opposite_m, -m, atol=1e-11)
    np.testing.assert_allclose(opposite_k, k, atol=1e-11)
