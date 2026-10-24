import copy
from types import SimpleNamespace

import pytest

from benchmark import discovery_scaml as study
from benchmark.discovery_config import development_tasks, write_json


def rows_for(methods=study.METHODS):
    return [{**task, 'method': method, 'queries': 128, 'pool_unique_collisions': 20,
             'unique_collisions': 2, 'collision_recall': .1}
            for method in methods for task in development_tasks()]


def test_all_four_modes_need_complete_paired_matrix():
    rows = rows_for()
    assert len(rows) == 160 and len(study.METHODS) == 4
    study.validate_rows(rows)
    with pytest.raises(ValueError, match='40 paired tasks'):
        study.validate_rows(rows[:-1])
    duplicate = copy.deepcopy(rows)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match='without duplicates'):
        study.validate_rows(duplicate)


@pytest.mark.parametrize('changes', [
    {'queries': 127}, {'collision_recall': .11}, {'unique_collisions': 21},
    {'unique_collisions': 129, 'pool_unique_collisions': 200, 'collision_recall': 129 / 200},
    {'pool_unique_collisions': 21, 'collision_recall': 2 / 21},
    {'unique_collisions': 0, 'pool_unique_collisions': 0, 'collision_recall': 0.},
])
def test_invalid_metric_or_budget_cannot_enter_report(changes):
    rows = rows_for()
    rows[0].update(changes)
    with pytest.raises(ValueError, match='budget, deduplicated pool or recall'):
        study.validate_rows(rows)


def test_zero_collision_tasks_remain_with_none_recall():
    rows = rows_for()
    task = rows[0]
    for row in rows:
        if all(row[k] == task[k] for k in ('target_id', 'scenario', 'algorithm_seed')):
            row.update(pool_unique_collisions=0, unique_collisions=0, collision_recall=None)
    study.validate_rows(rows)
    assert len(rows) == 160


def test_all_thirty_existing_references_required():
    assert len(study.REFERENCES) == len(set(study.REFERENCES)) == 30
    rows = rows_for(study.REFERENCES)
    assert len(rows) == 1200
    study.validate_rows(rows, study.REFERENCES)
    with pytest.raises(ValueError, match='40 paired tasks'):
        study.validate_rows(rows[:-1], study.REFERENCES)


def test_freeze_waits_for_rgpe_audit_and_report(monkeypatch, tmp_path):
    foundation, destination = tmp_path / 'rgpe', tmp_path / 'scaml'
    monkeypatch.setattr(study.rgpe, 'REPORT', foundation)
    monkeypatch.setattr(study, 'REPORT', destination)
    write_json(foundation / 'run_index.json', {'status': 'incomplete'})
    write_json(foundation / 'audit/report.json', {'status': 'available_completed_runs_checked', 'runs': 79})
    write_json(foundation / 'report.json', {'status': 'incomplete'})
    write_json(foundation / 'audit/negative_controls.json', {'status': 'passed', 'positive_controls': 2,
                                                           'rejected_corruptions': 8})
    with pytest.raises(ValueError, match='Complete and independently audit/report'):
        study.freeze()
    assert not destination.exists()


def test_changed_runtime_cannot_replace_registered_design(monkeypatch, tmp_path):
    foundation, destination = tmp_path / 'rgpe', tmp_path / 'scaml'
    monkeypatch.setattr(study.rgpe, 'REPORT', foundation)
    monkeypatch.setattr(study, 'REPORT', destination)
    monkeypatch.setattr(study.rgpe, 'protocol', lambda: {'foundation': True})
    monkeypatch.setattr(study, 'protocol', lambda: {'changed': True})
    monkeypatch.setattr(study, 'previous_rows', lambda: rows_for(study.REFERENCES))
    write_json(foundation / 'run_index.json', {'status': 'complete', 'protocol': {'foundation': True}})
    write_json(foundation / 'audit/report.json', {'status': 'passed', 'runs': 80, 'independent_full_pool_choices': 240})
    write_json(foundation / 'report.json', {'status': 'complete_audited_comparison'})
    write_json(foundation / 'audit/negative_controls.json', {'status': 'passed', 'positive_controls': 2,
                                                           'rejected_corruptions': 8})
    write_json(destination / 'design.json', {'changed': False})
    with pytest.raises(ValueError, match='differs from its registered design'):
        study.freeze()
    assert not (destination / 'protocol.json').exists()


def test_prepare_rejects_source_information_in_target_control(monkeypatch, tmp_path):
    history = [SimpleNamespace(source_id=f'h{i}') for i in range(8)]
    monkeypatch.setattr(study, 'PREPARATION', tmp_path / 'preparation')
    monkeypatch.setattr(study, 'REPORT', tmp_path / 'study')
    monkeypatch.setattr(study, 'load_construction', lambda kind: (history, []))
    write_json(study.PREPARATION / 'checks.json', {'status': 'passed', 'dedicated_unit_checks': 17,
                                                'failures': 0, 'errors': 0})
    source = [{'kind': kind, 'historical_pseudo_target': a.source_id, 'support_count': count,
               'source_ids': [b.source_id for b in history if b != a], 'source_reference': -1., 'full_pool_size': 512}
              for kind in ('cut_in', 'cut_out') for a in history for count in (8, 128)]
    target = [{**r, 'source_ids': [], 'source_reference': -1.} for r in source]
    for name, rows in (('historical_numeric.json', source), ('target_numeric.json', target)):
        write_json(study.PREPARATION / name, {'status': 'passed', 'checks': rows,
                                             'independent_gradient_vectors': 32})
    with pytest.raises(ValueError, match='no-source historical numerical condition'):
        study.prepare()
    assert not study.REPORT.exists()


def test_report_refuses_partial_independent_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'REPORT', tmp_path)
    monkeypatch.setattr(study, 'protocol', lambda: {'fixed': True})
    write_json(tmp_path / 'run_index.json', {'status': 'complete', 'protocol': {'fixed': True}})
    write_json(tmp_path / 'audit/report.json', {'status': 'available_completed_runs_checked', 'runs': 159})
    with pytest.raises(ValueError, match='all 160 ScaML conditions'):
        study.report()
