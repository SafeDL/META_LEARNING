import pytest

from benchmark.discovery_adaptive_target import METHODS, gate
from benchmark.discovery_config import STRONG_BASELINES, development_tasks


def records():
    rows,baselines = [],[]
    for task in development_tasks():
        empty = task['lineage'] == 'uct'
        for method in (*METHODS,*STRONG_BASELINES,'kinematic_kernel'):
            count = {'hybrid_local':99,'hybrid_global':98,'hybrid_shared':98.5,
                     'target_local':96,'gp_target_local':96.5,'fsbo':95,
                     'kinematic_kernel':96}.get(method,94)
            row = {**task,'method':method,'queries':128,'pool_unique_collisions':0 if empty else 100,
                   'unique_collisions':0 if empty else count,'collision_recall':None if empty else count/100,
                   'discovery_auc':count/2,'selection_s':1,'adaptation_s':1}
            (rows if method in METHODS else baselines).append(row)
    return rows,baselines


def setup(monkeypatch,baselines):
    def read(path):
        source = str(path)
        return {'runs':[r for r in baselines if r['method'] in
                        (('kinematic_kernel',) if 'geometry_search' in source else STRONG_BASELINES)]}
    monkeypatch.setattr('benchmark.discovery_adaptive_target.read_json',read)


def test_strong_target_control_participates_in_gate_and_null_recall_is_not_imputed(monkeypatch):
    rows,baselines = records()
    setup(monkeypatch,baselines)
    result = gate(rows)
    assert result['status'] == 'passed'
    assert result['strongest_baseline'] == 'gp_target_local'
    assert result['required_recall'] == pytest.approx(.9825)
    assert result['remaining_miss_reduction'] == pytest.approx((.99-.965)/.035)
    assert result['summary']['hybrid_local']['collision_recall']['valid_lineages'] == 3
    assert result['summary']['hybrid_local']['unique_collisions']['valid_lineages'] == 4


def test_mean_improvement_does_not_replace_local_mechanism_evidence(monkeypatch):
    rows,baselines = records()
    setup(monkeypatch,baselines)
    for row in rows:
        if row['method'] == 'hybrid_global' and row['collision_recall'] is not None:
            row.update(unique_collisions=99,collision_recall=.99)
    result = gate(rows)
    assert result['status'] == 'not_passed'
    assert not result['conditions']['recall_exceeds_hybrid_global']
    with pytest.raises(ValueError,match='200-run'):
        gate(rows[:-1])
