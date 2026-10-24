import copy

import numpy as np
import pytest
import torch

from benchmark.scenarios import features, sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.common.gp_classifier import ClassificationKernel
from methods.gcrt.converged_search import MODES, select
from methods.gcrt.joint_response import JointResponseKernel


class Oracle:
    def __init__(self, candidates, clearance=.5):
        self.candidates, self.clearance, self.paid = candidates, clearance, []

    def query(self, index, *, purpose):
        assert index not in [i for i, _ in self.paid]
        self.paid.append((index, purpose))
        return QueryFeedback(self.candidates[index].scenario_id, index % 2 == 0,
                             SafetyMeasurements(self.clearance, 2.), .1)


def fixture(mode):
    torch.manual_seed(1327)
    candidates = sample('cut_in', 8, 'development')
    x = features(candidates)
    base = ClassificationKernel(x.shape[1], deep=True)
    auxiliary = JointResponseKernel(x.shape[1])
    sources = {'names': ['a', 'b'], 'weights': np.array([.5, .5]),
        'means': np.stack((-1+x[:, 0], -1-x[:, 0])),
        'covariances': np.stack((np.eye(8)*.02, np.eye(8)*.03))}
    kwargs = {'pretrained': base, 'sources': None if mode in MODES[2:] else sources,
        'auxiliary': auxiliary if mode == 'converged_dual_target' else None,
        'mode': mode, 'beta': 1., 'initial_indices': [1, 3], 'budget': 5, 'device': 'cpu'}
    return candidates, base, kwargs


@pytest.mark.parametrize('mode', MODES)
def test_budget_causal_scores_converged_modes_and_collision_only_updates(monkeypatch, mode):
    candidates, base, kwargs = fixture(mode)
    original = copy.deepcopy(base.state_dict())
    fits = []

    def fit(model, x, labels, **settings):
        fits.append((x.copy(), list(labels), settings))

    monkeypatch.setattr('methods.gcrt.converged_search.fit_target', fit)
    oracle = Oracle(candidates)
    result = select(candidates, oracle, **kwargs)
    assert result['selected_indices'] == [i for i, _ in oracle.paid]
    assert result['actual_queries'] == 5 and len(set(result['selected_indices'])) == 5
    assert [purpose for _, purpose in oracle.paid] == ['initial', 'initial', 'target', 'target', 'target']
    for count, row in enumerate(result['queries'], 1):
        p, weight = np.array(row['prequery_component_probabilities']), np.array(row['prequery_component_weights'])
        assert weight.sum() == pytest.approx(1.) and weight[-1] == pytest.approx(.1)
        assert row['prequery_probability'] == pytest.approx(weight @ p, abs=2e-12)
        expected = np.log(np.maximum(p[:-1] if row['feedback']['collision'] else 1-p[:-1], 1e-15))
        np.testing.assert_allclose(row['prequery_group_log_likelihood'], expected, atol=2e-12)
        assert all(info['gradient_max'] <= 1e-10 for info in [*row['conditional_modes'], row['physical_mode']])
        relevant = [r for r in fits if len(r[1]) == count]
        assert len(relevant) == (3 if mode == 'converged_dual_target' else 2)
        for x, labels, settings in relevant:
            paid = result['selected_indices'][:count]
            np.testing.assert_array_equal(x, features(candidates)[paid])
            assert labels == [i % 2 == 0 for i in paid] and settings['steps'] == 10
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)


@pytest.mark.parametrize('mode', MODES)
def test_safety_values_do_not_change_selection(monkeypatch, mode):
    candidates, _, kwargs = fixture(mode)
    monkeypatch.setattr('methods.gcrt.converged_search.fit_target', lambda *args, **kw: None)
    a = select(candidates, Oracle(candidates, .2), **kwargs)
    b = select(candidates, Oracle(candidates, 50.), **kwargs)
    assert a['selected_indices'] == b['selected_indices']


def test_source_order_does_not_change_selection(monkeypatch):
    candidates, _, kwargs = fixture('converged_local')
    monkeypatch.setattr('methods.gcrt.converged_search.fit_target', lambda *args, **kw: None)
    a = select(candidates, Oracle(candidates), **kwargs)
    kwargs['sources'] = {k: (v[::-1] if k == 'names' else v[::-1].copy()) for k, v in kwargs['sources'].items()}
    b = select(candidates, Oracle(candidates), **kwargs)
    assert a['selected_indices'] == b['selected_indices']


@pytest.mark.parametrize('mode', ['converged_local', 'converged_dual_target'])
def test_cuda_kernel_and_cpu_conditional_inference_keep_the_same_budget(monkeypatch, mode):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    candidates, _, kwargs = fixture(mode)
    kwargs['device'] = 'cuda'
    monkeypatch.setattr('methods.gcrt.converged_search.fit_target', lambda *args, **kw: None)
    result = select(candidates, Oracle(candidates), **kwargs)
    assert result['actual_queries'] == 5
    assert result['conditional_inference_device'] == 'cpu'
    assert result['kernel_and_adaptation_device'] == 'cuda'
    assert all(info['gradient_max'] <= 1e-10 for row in result['queries']
               for info in [*row['conditional_modes'], row['physical_mode']])
