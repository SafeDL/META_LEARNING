import copy

import numpy as np
import pytest
import torch

from benchmark.query import PaidExecution, QueryLedger, ReplayExecutor, run_isolated
from benchmark.scenarios import features, sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from methods.gcrt.joint_response import JointResponseKernel
from methods.gcrt.residual_search import MODES, select


def fixture():
    torch.manual_seed(37)
    candidates = sample('cut_in', 8, 'development')
    x = features(candidates)
    model = ClassificationKernel(x.shape[1], deep=True)
    auxiliary = JointResponseKernel(x.shape[1])
    sources = {'names': ['source_a', 'source_b'], 'weights': np.array([.6, .4]),
               'means': np.stack((-1.2+x[:, 0], -1.2-x[:, 0])),
               'covariances': np.stack((np.eye(8)*.03, np.eye(8)*.02))}
    return candidates, x, model, auxiliary, sources


class Oracle:
    def __init__(self, candidates, clearance=.7):
        self.candidates, self.clearance = candidates, clearance
        self.paid = []

    def query(self, index, *, purpose):
        assert not any(i == index for i, _ in self.paid)
        self.paid.append((index, purpose))
        return QueryFeedback(self.candidates[index].scenario_id, index % 2 == 0,
                             SafetyMeasurements(self.clearance, 2), .1)


def arguments(mode, model, auxiliary, sources):
    return {'pretrained': model, 'sources': None if mode in ('residual_target', 'residual_dual_target') else sources,
            'auxiliary': auxiliary if mode == 'residual_dual_target' else None,
            'mode': mode, 'beta': .3, 'initial_indices': [1, 3], 'budget': 5, 'device': 'cpu'}


@pytest.mark.parametrize('mode', MODES)
def test_budget_probabilities_prequery_scores_and_paid_only_fitting(monkeypatch, mode):
    candidates, x, model, auxiliary, sources = fixture()
    original = copy.deepcopy(model.state_dict())
    fits = []

    def fit(adapted, paid_x, labels, **settings):
        kind = 'joint' if hasattr(adapted, 'margin_mean') else 'fsbo' if adapted.deep else 'physical'
        fits.append((kind, paid_x.copy(), list(labels), settings))
        with torch.no_grad():
            adapted.mean.add_(.02*len(labels))

    monkeypatch.setattr('methods.gcrt.residual_search.fit_target', fit)
    oracle = Oracle(candidates)
    result = select(candidates, oracle, **arguments(mode, model, auxiliary, sources))
    assert result['selected_indices'] == [i for i, _ in oracle.paid]
    assert len(set(result['selected_indices'])) == result['actual_queries'] == 5
    assert [p for _, p in oracle.paid] == ['initial', 'initial', 'target', 'target', 'target']
    for n, row in enumerate(result['queries'], 1):
        p = np.array(row['prequery_component_probabilities'])
        weights = np.array(row['prequery_component_weights'])
        assert weights.sum() == pytest.approx(1.) and weights[-1] == pytest.approx(.1)
        assert row['prequery_probability'] == pytest.approx(weights @ p, abs=2e-12)
        expected = np.log(np.maximum(p[:-1] if row['feedback']['collision'] else 1-p[:-1], 1e-15))
        np.testing.assert_allclose(row['prequery_group_log_likelihood'], expected, atol=2e-12)
        updated = np.array(row['prequery_group_weights'])*np.exp(expected)
        np.testing.assert_allclose(row['group_weights_after_query'], updated/updated.sum(), atol=2e-12)
        relevant = [(k, px, y, setting) for k, px, y, setting in fits if len(y) == n]
        assert {r[0] for r in relevant} == ({'joint', 'fsbo', 'physical'} if mode == 'residual_dual_target' else {'fsbo', 'physical'})
        for kind, px, y, setting in relevant:
            paid = result['selected_indices'][:n]
            np.testing.assert_array_equal(px, x[paid])
            assert y == [i % 2 == 0 for i in paid]
            assert setting == {'steps': 10, 'learning_rate': .01 if kind == 'physical' else .003}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    assert result['beta'] == (None if mode in ('residual_target', 'residual_dual_target') else 1. if 'unit' in mode else .3)


@pytest.mark.parametrize('mode', MODES[:4])
def test_source_permutation_does_not_change_queries(monkeypatch, mode):
    candidates, _, model, auxiliary, sources = fixture()
    monkeypatch.setattr('methods.gcrt.residual_search.fit_target', lambda *a, **k: None)
    permuted = {k: v[::-1].copy() for k, v in sources.items() if k != 'names'}
    permuted['names'] = sources['names'][::-1]
    results = [select(candidates, Oracle(candidates), **arguments(mode, model, auxiliary, context))
               for context in (sources, permuted)]
    assert results[0]['selected_indices'] == results[1]['selected_indices']
    np.testing.assert_allclose([q['prequery_probability'] for q in results[0]['queries']],
                               [q['prequery_probability'] for q in results[1]['queries']], atol=2e-12)


@pytest.mark.parametrize('mode', MODES)
def test_all_modes_ignore_online_safety_values(mode):
    candidates, _, model, auxiliary, sources = fixture()
    kwargs = arguments(mode, model, auxiliary, sources)
    kwargs.update(initial_indices=[0], budget=3)
    results = [select(candidates, Oracle(candidates, clearance), **kwargs) for clearance in (.01, 90.)]
    assert results[0]['selected_indices'] == results[1]['selected_indices']
    np.testing.assert_array_equal([q['prequery_probability'] for q in results[0]['queries']],
                                  [q['prequery_probability'] for q in results[1]['queries']])


def test_unit_amplitude_and_uniform_influence_recover_matching_rules(monkeypatch):
    candidates, x, model, auxiliary, sources = fixture()
    monkeypatch.setattr('methods.gcrt.residual_search.fit_target', lambda *a, **k: None)
    outputs = []
    for mode in MODES[:4]:
        kwargs = arguments(mode, model, auxiliary, sources)
        kwargs.update(beta=1., feature_fn=lambda c: np.zeros_like(x))
        outputs.append(select(candidates, Oracle(candidates), **kwargs))
    for result in outputs[1:]:
        assert result['selected_indices'] == outputs[0]['selected_indices']
        np.testing.assert_allclose([q['prequery_probability'] for q in result['queries']],
                                  [q['prequery_probability'] for q in outputs[0]['queries']], atol=2e-12)


def test_target_control_recovers_the_original_single_classifier():
    candidates, x, base, auxiliary, _ = fixture()
    kwargs = arguments('residual_target', base, auxiliary, None)
    kwargs.update(initial_indices=[0, 1, 4], budget=3)
    result = select(candidates, Oracle(candidates), **kwargs)
    paid, labels, model = [], [], copy.deepcopy(base)
    for row in result['queries']:
        with torch.no_grad():
            p = float(collision_probabilities(model, x[paid], labels, x)[row['index']])
        assert row['prequery_group_probability'] == pytest.approx(p, abs=2e-7, rel=2e-6)
        paid.append(row['index'])
        labels.append(row['feedback']['collision'])
        model = copy.deepcopy(base)
        fit_target(model, x[paid], labels, steps=10, learning_rate=.003)


def test_contract_errors_happen_before_any_paid_query():
    candidates, _, model, auxiliary, sources = fixture()
    oracle = Oracle(candidates)
    kwargs = arguments('residual_local', model, auxiliary, sources)
    with pytest.raises(ValueError, match='initialization'):
        select(candidates, oracle, **(kwargs | {'initial_indices': [1, 1]}))
    with pytest.raises(ValueError, match='target controls'):
        select(candidates, oracle, **(kwargs | {'sources': None}))
    with pytest.raises(ValueError, match='target controls'):
        select(candidates, oracle, **(kwargs | {'auxiliary': auxiliary}))
    assert oracle.paid == []


def test_isolated_selector_receives_only_metered_feedback():
    candidates, _, model, auxiliary, sources = fixture()
    executions = {s.scenario_id: PaidExecution(QueryFeedback(s.scenario_id, i % 2 == 0,
                    SafetyMeasurements(.7, 2), .1), [{'evaluator_private': True}]) for i, s in enumerate(candidates)}
    ledger = QueryLedger(candidates, 3, ReplayExecutor(executions))
    public = arguments('residual_local', model, auxiliary, sources)
    public.update(candidates=candidates, initial_indices=[1], budget=3)
    result = run_isolated(select, public, ledger)
    assert len(ledger.records) == 3 and len(ledger.trajectories) == 3
    serialized = str(result)
    assert 'evaluator_private' not in serialized and 'pool_unique_collisions' not in serialized
