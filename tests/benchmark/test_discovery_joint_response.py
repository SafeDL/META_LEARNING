import copy

import pytest

from benchmark.discovery_config import STRONG_BASELINES, development_tasks
from benchmark.discovery_joint_response import METHODS, STRONG, gate


def records():
    runs,baselines = [],[]
    for task in development_tasks():
        empty = task['lineage'] == 'uct'
        for method in (*METHODS,*STRONG[:-2]):
            count = {'joint_local':99,'joint_global':98,'joint_shared':98,'joint_mean':98,
                     'joint_binary':97,'joint_target':97,'joint_uncorrelated':96,
                     'fsbo':95,'kinematic_kernel':96}.get(method,94)
            row = {**task,'method':method,'queries':128,'pool_unique_collisions':0 if empty else 100,
                   'unique_collisions':0 if empty else count,'collision_recall':None if empty else count/100,
                   'discovery_auc':count/2,'selection_s':1,'adaptation_s':1}
            (runs if method in METHODS else baselines).append(row)
    return runs,baselines


def patch_baselines(monkeypatch,baselines):
    def read(path):
        directory = str(path)
        methods = (('kinematic_kernel',) if 'geometry_search' in directory else
                   ('target_local','gp_target_local') if 'adaptive_target' in directory else STRONG_BASELINES)
        return {'runs':[r for r in baselines if r['method'] in methods]}
    monkeypatch.setattr('benchmark.discovery_joint_response.read_json',read)


def test_joint_gate_compares_against_feedback_matched_target_model(monkeypatch):
    runs,baselines = records()
    patch_baselines(monkeypatch,baselines)
    result = gate(runs)
    assert result['status'] == 'passed'
    assert result['strongest_baseline'] == 'joint_target'
    assert result['required_recall'] == pytest.approx(.985)
    assert result['remaining_miss_reduction'] == pytest.approx(2/3)
    assert result['summary']['joint_local']['collision_recall']['valid_lineages'] == 3
    assert result['summary']['joint_local']['unique_collisions']['valid_lineages'] == 4


def test_joint_gate_requires_local_mechanism_and_complete_matching_matrix(monkeypatch):
    runs,baselines = records()
    patch_baselines(monkeypatch,baselines)
    for row in runs:
        if row['method'] == 'joint_global' and row['collision_recall'] is not None:
            row.update(unique_collisions=99,collision_recall=.99)
    assert gate(runs)['conditions']['local_recall_exceeds_global'] is False
    with pytest.raises(ValueError,match='280-run'):
        gate(runs[:-1])
    altered = copy.deepcopy(runs)
    altered[0]['pool_unique_collisions'] += 1
    with pytest.raises(ValueError,match='denominators'):
        gate(altered)
