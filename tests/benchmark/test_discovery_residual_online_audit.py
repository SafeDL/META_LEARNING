import copy

import pytest

from benchmark.discovery_config import development_tasks
from benchmark.discovery_residual_online import MAIN
from benchmark.discovery_residual_online_audit import equivalent_parameter_cells


def test_identical_parameter_cells_do_not_supply_spurious_scale_effects(monkeypatch):
    from benchmark import discovery_residual_online_audit as module
    task = next(t for t in development_tasks() if t['scenario'] == 'cut_out')
    rows = [{**task, 'method': method, 'path': f'{method}.json',
             'discovery_curve': [0, 1, 2], 'recall_curve': [0., .5, 1.]}
            for method in (MAIN, 'residual_unit_local')]
    monkeypatch.setattr(module, 'read_json', lambda p: {'selected_beta': {'cut_in': .3, 'cut_out': 1.}}
                        if p.name == 'protocol.json' else {'result': {'selected_indices': [3, 1, 4]}})
    result = equivalent_parameter_cells(rows)
    assert len(result) == 1 and result[0]['curves_identical'] and result[0]['selected_paths_identical']
    changed = copy.deepcopy(rows)
    changed[1]['discovery_curve'] = [0, 1, 1]
    changed[1]['recall_curve'] = [0., .5, .5]
    with pytest.raises(ValueError, match='Identical residual parameters'):
        equivalent_parameter_cells(changed)


def test_final_report_rejects_an_incomplete_audit(monkeypatch):
    from benchmark import discovery_residual_online_report as module
    data = {'run_index.json': {'protocol': {'frozen': True}},
            'report.json': {'status': 'passed', 'runs': 6},
            'negative_controls.json': {'status': 'passed'}}
    monkeypatch.setattr(module, 'read_json', lambda p: data[p.name])
    monkeypatch.setattr(module, 'protocol', lambda: {'frozen': True})
    with pytest.raises(ValueError, match='complete residual matrix'):
        module.report()
