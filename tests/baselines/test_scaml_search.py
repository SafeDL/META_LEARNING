import copy
import math

import numpy as np
import pytest
from scipy.optimize import minimize, root_scalar
from scipy.special import expit
import torch

from benchmark.query import PaidExecution, QueryLedger, ReplayExecutor, run_isolated
from benchmark.scenarios import features, sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.baselines.scaml_gp.search import (MODES, SOURCE_MODES, TARGET_MODES, WEIGHT_FLOOR, fit_prior,
                                              residual_covariance, select, source_prior)
from methods.common.classification import torch_moments
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities
from methods.common.gp_laplace import laplace, posterior


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def fixture():
    torch.manual_seed(37)
    candidates = sample('cut_in', 8, 'development')
    x = features(candidates)
    model = ClassificationKernel(x.shape[1], deep=True)
    sources = {'names': ['source_b', 'source_a'],
               'means': np.stack((-1.2 + x[:, 0], -1.2 - x[:, 0])),
               'covariances': np.stack((np.eye(8) * .03, np.eye(8) * .02))}
    return candidates, x, model, sources


class Oracle:
    def __init__(self, candidates, clearance=.7):
        self.candidates, self.clearance = candidates, clearance
        self.paid = []

    def query(self, index, *, purpose):
        assert index not in [i for i, _ in self.paid]
        self.paid.append((index, purpose))
        return QueryFeedback(self.candidates[index].scenario_id, index % 2 == 0,
                             SafetyMeasurements(self.clearance, 2), .1)


def test_general_mean_evidence_matches_independently_optimized_laplace_mode():
    mean = np.array([-.7, .1, .4, -.2])
    a = np.array([[1., .2, .1, .0], [.2, 1., .1, .2], [.1, .1, 1., .3], [.0, .2, .3, 1.]])
    covariance = a @ a.T
    labels = np.array([0., 1., 0., 1.])
    k = covariance + 1e-5 * np.eye(4)
    precision = np.linalg.inv(k)

    def objective(f):
        delta = f - mean
        return .5 * delta @ precision @ delta + np.sum(np.logaddexp(0, f) - labels * f)

    def gradient(f):
        return precision @ (f - mean) + expit(f) - labels

    def hessian(f):
        p = expit(f)
        return precision + np.diag(p * (1 - p))

    optimum = minimize(objective, mean, method='trust-exact', jac=gradient, hess=hessian,
                       options={'gtol': 1e-12})
    assert np.max(np.abs(gradient(optimum.x))) < 1e-8
    p = expit(optimum.x)
    root = np.sqrt(p * (1 - p))
    expected = objective(optimum.x) + .5 * np.linalg.slogdet(np.eye(4) + root[:, None] * k * root[None, :])[1]
    actual = laplace(torch.tensor(mean), torch.tensor(covariance), labels).nll
    assert float(actual) == pytest.approx(expected, abs=1e-10)


def test_scalar_evidence_gradients_match_finite_differences():
    encoded = torch.tensor([[.0, .2], [.3, .4], [.7, .3], [1., .1]], dtype=torch.float64)
    means = torch.tensor([[-1., -.3, .8, 1.], [.8, .2, -.5, -.9]], dtype=torch.float64)
    covariance = torch.stack((torch.eye(4) * .03, torch.eye(4) * .02)).double()
    labels = [False, False, True, True]

    def objective(parameters):
        mean, prior = source_prior(torch.tensor(-.2), residual_covariance(encoded, *parameters[:2]),
                                   means, covariance, parameters[2:], 0.)
        return laplace(mean, prior, labels).nll + parameters[2:].sum()

    parameters = torch.tensor([-.5, .2, .4, .2], dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(objective(parameters), parameters)[0]
    numerical = []
    for i in range(4):
        delta = torch.zeros(4, dtype=torch.float64)
        delta[i] = 1e-5
        numerical.append(float((objective(parameters.detach() + delta) - objective(parameters.detach() - delta)) / 2e-5))
    np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=2e-6)


def test_prior_preserves_squared_weights_and_recovers_target_at_zero_weights():
    _, x, model, sources = fixture()
    source_m, source_k = torch.tensor(sources['means']), torch.tensor(sources['covariances'])
    covariance = model.covariance(x)
    weights = torch.tensor([2., .3], dtype=torch.float64)
    mean, prior = source_prior(model.mean, covariance, source_m, source_k, weights, -1.2)
    torch.testing.assert_close(mean, model.mean + 2 * (source_m[0] + 1.2) + .3 * (source_m[1] + 1.2))
    torch.testing.assert_close(prior, covariance + 4 * source_k[0] + .09 * source_k[1])
    assert torch.linalg.eigvalsh(prior).min() > 0
    mean, prior = source_prior(model.mean, covariance, source_m, source_k, torch.zeros(2), -1.2)
    paid, labels = [0, 2, 5], [True, False, True]
    m, k, _ = posterior(mean, prior, paid, labels)
    actual = torch_moments(m, k.diagonal().clamp_min(1e-10))[0]
    expected = collision_probabilities(model, x[paid], labels, x)
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)


def test_positive_weight_fit_selects_relevant_source_and_reduces_objective():
    encoded = torch.stack((torch.linspace(0, 1, 8), torch.zeros(8)), dim=1).double()
    labels = [False] * 4 + [True] * 4
    response = torch.tensor([-2.] * 4 + [2.] * 4, dtype=torch.float64)
    means = torch.stack((response, -response))
    covariances = torch.stack((torch.eye(8), torch.eye(8))).double() * .03
    mean, covariance, parameters, info = fit_prior(encoded, torch.tensor(0.), means, covariances,
        0., list(range(8)), labels, math.log(.3), math.log(1.))
    assert parameters[2] > parameters[3]
    assert torch.all(parameters[2:] >= WEIGHT_FLOOR)
    assert info['objective'] < info['initial_objective']
    assert info['evaluations'] > 1 and np.isfinite(info['projected_gradient_max'])
    assert mean.shape == (8,) and covariance.shape == (8, 8)


def test_empty_feedback_keeps_registered_initial_coefficients():
    encoded = torch.eye(4, dtype=torch.float64)
    means = torch.zeros((2, 4), dtype=torch.float64)
    covariances = torch.stack((torch.eye(4), torch.eye(4))).double() * .03
    _, _, parameters, info = fit_prior(encoded, torch.tensor(-1.), means, covariances,
                                      0., [], [], math.log(.3), 0.)
    np.testing.assert_allclose(parameters, [math.log(.3), 0., .5, .5])
    assert info['status'] == 'no_feedback' and info['evaluations'] == 0


@pytest.mark.parametrize('mode', MODES)
def test_selector_fits_only_paid_labels_and_preserves_input_and_budget(monkeypatch, mode):
    candidates, x, model, sources = fixture()
    original = copy.deepcopy(model.state_dict())
    original_sources = copy.deepcopy(sources)
    target_fits, prior_fits = [], []

    def fit(adapted, paid_x, labels, **settings):
        target_fits.append((adapted.deep, paid_x.copy(), list(labels), settings))
        with torch.no_grad():
            adapted.mean.add_(.02 * len(labels))

    def fit_sources(encoded, mean, sm, sk, reference, paid, labels, initial_l, initial_v):
        prior_fits.append((list(paid), list(labels), float(mean)))
        return fit_prior(encoded, mean, sm, sk, reference, paid, labels, initial_l, initial_v)

    monkeypatch.setattr('methods.baselines.scaml_gp.search.fit_target', fit)
    monkeypatch.setattr('methods.baselines.scaml_gp.search.fit_prior', fit_sources)
    oracle = Oracle(candidates)
    target_only = mode in TARGET_MODES
    result = select(candidates, oracle, pretrained=model, sources=None if target_only else sources,
                    source_reference=None if target_only else -1.2,
                    initial_indices=[0, 1], budget=4, device='cpu', mode=mode)
    assert len(set(result['selected_indices'])) == result['actual_queries'] == 4
    assert result['selected_indices'] == [i for i, _ in oracle.paid]
    assert [p for _, p in oracle.paid] == ['initial', 'initial', 'target', 'target']
    assert result['source_names'] == ([] if target_only else ['source_a', 'source_b'])
    assert result['uses_source_posteriors'] is not target_only
    assert result['source_weights_normalized'] is False
    for step, (paid, labels, mean) in enumerate(prior_fits):
        assert paid == result['selected_indices'][:step]
        assert labels == [i % 2 == 0 for i in paid]
        assert mean == pytest.approx(.02 * len(labels))
    mass = .1 if mode.startswith('physical_') else 0.
    for n, row in enumerate(result['queries'], 1):
        assert all(w >= WEIGHT_FLOOR for w in row['prequery_source_weights'])
        expected = (1 - mass) * row['prequery_learned_probability']
        if mass:
            expected += mass * row['prequery_physical_probability']
        assert row['prequery_probability'] == pytest.approx(expected)
        assert 0 <= row['prior_optimization_s'] <= row['selection_s']
        relevant = [(deep, px, y, settings) for deep, px, y, settings in target_fits if len(y) == n]
        assert len(relevant) == (2 if mass else 1)
        for deep, px, y, settings in relevant:
            selected = result['selected_indices'][:n]
            np.testing.assert_array_equal(px, x[selected])
            assert y == [i % 2 == 0 for i in selected]
            assert settings == {'steps': 10, 'learning_rate': .003 if deep else .01}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    for key in ('means', 'covariances'):
        np.testing.assert_array_equal(sources[key], original_sources[key])


@pytest.mark.parametrize('mode', SOURCE_MODES)
def test_canonical_sources_and_safety_values_leave_selection_unchanged(mode):
    candidates, _, model, sources = fixture()
    reversed_sources = {'names': sources['names'][::-1],
                        'means': sources['means'][::-1].copy(),
                        'covariances': sources['covariances'][::-1].copy()}
    results = [select(candidates, Oracle(candidates, clearance), pretrained=model, sources=context,
                      source_reference=-1.2, initial_indices=[0, 1], budget=3, mode=mode, device='cpu')
               for context, clearance in ((sources, .01), (reversed_sources, 90.))]
    assert results[0]['selected_indices'] == results[1]['selected_indices']
    np.testing.assert_array_equal([r['prequery_probability'] for r in results[0]['queries']],
                                  [r['prequery_probability'] for r in results[1]['queries']])


def test_invalid_contract_fails_before_paid_queries():
    candidates, _, model, sources = fixture()
    oracle = Oracle(candidates)
    with pytest.raises(ValueError, match='initialization'):
        select(candidates, oracle, pretrained=model, sources=sources, source_reference=-1.2,
               initial_indices=[0, 0], budget=3, device='cpu')
    with pytest.raises(ValueError, match='reference mean'):
        select(candidates, oracle, pretrained=model, sources=sources, source_reference=np.nan,
               initial_indices=[0, 1], budget=3, device='cpu')
    assert oracle.paid == []


@pytest.mark.parametrize('mode', TARGET_MODES)
def test_target_controls_reject_source_context_and_ignore_online_safety(mode):
    candidates, _, model, sources = fixture()
    oracle = Oracle(candidates)
    with pytest.raises(ValueError, match='cannot receive explicit source'):
        select(candidates, oracle, pretrained=model, sources=sources, source_reference=-1.2,
               initial_indices=[0, 1], budget=3, device='cpu', mode=mode)
    assert oracle.paid == []
    results = [select(candidates, Oracle(candidates, clearance), pretrained=model, sources=None,
                      initial_indices=[0, 1], budget=3, device='cpu', mode=mode)
               for clearance in (.01, 90.)]
    assert results[0]['selected_indices'] == results[1]['selected_indices']
    assert results[0]['source_reference_mean'] is None
    assert results[0]['source_weight_prior'] is None
    for a, b in zip(results[0]['queries'], results[1]['queries']):
        assert a['prequery_source_weights'] == b['prequery_source_weights'] == []
        assert a['prequery_probability'] == b['prequery_probability']


def test_spawned_selector_has_only_metered_feedback():
    candidates, _, model, sources = fixture()
    executions = {s.scenario_id: PaidExecution(QueryFeedback(s.scenario_id, i % 2 == 0,
                   SafetyMeasurements(.7, 2), .1), [{'evaluator_private': True}])
                  for i, s in enumerate(candidates)}
    ledger = QueryLedger(candidates, 3, ReplayExecutor(executions))
    result = run_isolated(select, {'candidates': candidates, 'pretrained': model, 'sources': sources,
                                  'source_reference': -1.2, 'initial_indices': [0, 1],
                                  'budget': 3, 'device': 'cpu'}, ledger)
    assert len(ledger.records) == result['actual_queries'] == 3
    assert 'evaluator_private' not in str(result) and 'pool_unique_collisions' not in str(result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_general_mean_evidence_and_gradients_match_between_cpu_and_cuda():
    values = torch.tensor([-.7, .1, .4, -.2], dtype=torch.float64)
    covariance = torch.eye(4, dtype=torch.float64) * .5 + .3
    results = []
    for device in ('cpu', 'cuda'):
        mean = values.to(device).clone().requires_grad_(True)
        loss = laplace(mean, covariance.to(device), [False, True, True, False]).nll
        gradient = torch.autograd.grad(loss, mean)[0]
        results.append((loss.detach().cpu(), gradient.cpu()))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(results[0][1], results[1][1], atol=1e-10, rtol=1e-10)


def test_damped_mode_resolves_large_variance_opposing_prior_without_oscillation():
    mean = torch.tensor([-5.], dtype=torch.float64)
    covariance = torch.tensor([[100.]], dtype=torch.float64)
    state = laplace(mean, covariance, [True])
    optimum = root_scalar(lambda f: (f + 5) / 100.00001 + expit(f) - 1,
                          bracket=(-5, 100), method='brentq').root
    expected_nll = (.5 * (optimum + 5)**2 / 100.00001 + np.logaddexp(0, -optimum)
                    + .5 * np.log1p(100.00001 * expit(optimum) * (1 - expit(optimum))))
    assert float(state.nll) == pytest.approx(expected_nll, abs=1e-10)
    assert state.backtracks > 0 and state.gradient_max <= 1e-8
