import copy

import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.special import expit

from benchmark.query import PaidExecution, QueryLedger, ReplayExecutor, run_isolated
from benchmark.scenarios import features, sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.baselines.fsbo.search import select as select_fsbo
from methods.baselines.rgpe.search import (MODES, ensemble_probability, gaussian_samples,
                                         rank_weights, ranking_losses, select, weights_from_losses)
from methods.common.classification import torch_moments
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities
from methods.common.gp_joint import bernoulli_posterior


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def fixture():
    torch.manual_seed(37)
    candidates = sample('cut_in', 9, 'development')
    x = features(candidates)
    model = ClassificationKernel(x.shape[1], deep=True)
    sources = {'names': ['source_b', 'source_a'],
               'means': np.stack((-1.2 + x[:, 0], -1.2 - x[:, 0])),
               'covariances': np.stack((np.eye(9) * .03, np.eye(9) * .02))}
    return candidates, x, model, sources


class Oracle:
    def __init__(self, candidates, *, clearance=.7, single_class=False):
        self.candidates, self.clearance = candidates, clearance
        self.single_class = single_class
        self.paid = []

    def query(self, index, *, purpose):
        assert index not in [i for i, _ in self.paid]
        self.paid.append((index, purpose))
        collision = False if self.single_class else index % 2 == 0
        return QueryFeedback(self.candidates[index].scenario_id, collision,
                             SafetyMeasurements(self.clearance, 2), .1)


def test_joint_posterior_reproduces_existing_classifier_and_prior():
    _, x, model, _ = fixture()
    mean = model.mean.expand(len(x))
    covariance = model.covariance(x)
    for observed, labels in (([], []), ([0, 2, 5], [True, False, True])):
        m, k = bernoulli_posterior(mean, covariance, observed, labels)
        expected = collision_probabilities(model, x[observed], labels, x)
        actual = torch_moments(m, k.diagonal().clamp_min(1e-10))[0]
        # Full-pool and support-only FP32 embedding batches differ slightly.
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(k, k.T)
        assert torch.linalg.eigvalsh(k).min() > -1e-10


def test_joint_posterior_matches_independent_mode_optimization():
    mean = np.array([-.7, .1, .4, -.2])
    a = np.array([[1., .2, .1, .0], [.2, 1., .1, .2], [.1, .1, 1., .3], [.0, .2, .3, 1.]])
    covariance = a @ a.T
    observed, labels = np.array([0, 2, 3]), np.array([0., 1., 0.])
    support = covariance[np.ix_(observed, observed)] + 1e-5 * np.eye(3)
    precision = np.linalg.inv(support)
    prior_mean = mean[observed]

    def objective(f):
        deviation = f - prior_mean
        return .5 * deviation @ precision @ deviation + np.sum(np.logaddexp(0, f) - labels * f)

    def gradient(f):
        return precision @ (f - prior_mean) + expit(f) - labels

    def hessian(f):
        p = expit(f)
        return precision + np.diag(p * (1 - p))

    fitted = minimize(objective, prior_mean, jac=gradient, hess=hessian,
                      method='trust-exact', options={'gtol': 1e-12})
    assert np.max(np.abs(gradient(fitted.x))) < 1e-8
    alpha = precision @ (fitted.x - prior_mean)
    cross = covariance[observed]
    expected_m = mean + cross.T @ alpha
    weight = expit(fitted.x) * (1 - expit(fitted.x))
    expected_k = covariance - cross.T @ np.linalg.solve(support + np.diag(1 / weight), cross)
    m, k = bernoulli_posterior(torch.tensor(mean), torch.tensor(covariance), observed, labels)
    np.testing.assert_allclose(m.detach(), expected_m, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(k.detach(), expected_k, atol=1e-9, rtol=1e-9)


def test_batched_loo_refits_match_individual_folds_and_exclude_held_label():
    _, x, model, _ = fixture()
    mean, covariance = model.mean.expand(6), model.covariance(x[:6])
    indices = torch.arange(6).expand(6, -1)[~torch.eye(6, dtype=torch.bool)].reshape(6, 5)
    y = torch.tensor([0., 1., 1., 0., 1., 0.], dtype=torch.float64)
    m, k = bernoulli_posterior(mean, covariance, indices, y[indices])
    for fold in range(6):
        individual = bernoulli_posterior(mean, covariance, indices[fold], y[indices[fold]])
        torch.testing.assert_close(m[fold], individual[0])
        torch.testing.assert_close(k[fold], individual[1])
        changed = y.clone()
        changed[fold] = 1 - changed[fold]
        held = bernoulli_posterior(mean, covariance, indices, changed[indices])
        torch.testing.assert_close(held[0][fold], m[fold], atol=0, rtol=0)
        torch.testing.assert_close(held[1][fold], k[fold], atol=0, rtol=0)


def test_ranking_losses_match_ordered_pairs_and_within_fold_comparisons():
    y = torch.tensor([0., 1., 0.], dtype=torch.float64)
    source = torch.tensor([[.1, .8, .2], [.7, .3, .1]], dtype=torch.float64)
    target = torch.tensor([[[.1, .5, .7], [.1, .6, .4], [.6, .1, .3]],
                           [[.9, .2, .1], [.8, .1, .4], [.6, .5, .3]]], dtype=torch.float64)
    for draws in (source, target):
        expected = []
        for draw in draws:
            expected.append(sum(int(bool((draw[j] < draw[k]) if draw.ndim == 1
                                          else (draw[j, j] < draw[j, k])) != bool(y[j] < y[k]))
                                for j in range(3) for k in range(3) if y[j] != y[k]))
        assert ranking_losses(draws, y).tolist() == expected
    assert ranking_losses(source, torch.zeros(3)).tolist() == [0, 0]


def test_pruning_and_target_ties_follow_paper_instead_of_first_argmin():
    generator = torch.Generator().manual_seed(17)
    losses = torch.tensor([[0, 0, 1, 3], [9, 9, 9, 9], [0, 1, 1, 2]])
    weights, pruned, _, threshold = weights_from_losses(losses, generator)
    assert pruned.tolist() == [False, True, False]
    assert float(threshold) == pytest.approx(1.85)
    assert weights.tolist() == [.25, 0., .75]
    tied = torch.tensor([[0] * 10000, [0] * 10000, [1] * 10000])
    weights, _, _, _ = weights_from_losses(tied, generator)
    assert weights[-1] == 0
    assert abs(float(weights[0]) - .5) < .02


def test_single_class_and_empty_feedback_use_target_tie_rule():
    _, x, model, sources = fixture()
    means, covariance = torch.tensor(sources['means']), torch.tensor(sources['covariances'])
    for paid, labels in (([], []), ([0], [False]), ([0, 1], torch.zeros(2))):
        weights, info = rank_weights(means, covariance, model.mean.expand(9), model.covariance(x),
                                     paid, labels, seed=37)
        assert weights.tolist() == [0., 0., 1.]
        assert info['informative_ordered_pairs'] == 0


def test_ensemble_uses_squared_covariance_weights_before_logistic_integration():
    means = torch.tensor([[-8., .1], [2., .3]], dtype=torch.float64)
    variances = torch.tensor([[.1, 2.], [.8, 3.]], dtype=torch.float64)
    weights = torch.tensor([.25, .75], dtype=torch.float64)
    actual = ensemble_probability(means, variances, weights)
    expected = torch_moments(.25 * means[0] + .75 * means[1],
                             .25**2 * variances[0] + .75**2 * variances[1])[0]
    torch.testing.assert_close(actual, expected)
    probability_mixture = (weights[:, None] * torch_moments(means, variances)[0]).sum(0)
    assert float(torch.max(torch.abs(actual - probability_mixture))) > .01


def test_correlated_gaussian_draws_preserve_joint_covariance():
    mean = torch.tensor([[.2, -.5], [-.4, .6]], dtype=torch.float64)
    covariance = torch.tensor([[[.3, .2], [.2, .4]], [[.5, -.3], [-.3, .6]]], dtype=torch.float64)
    draws = gaussian_samples(mean, covariance, 50000, torch.Generator().manual_seed(71))
    for i in range(2):
        np.testing.assert_allclose(draws[:, i].mean(0), mean[i], atol=.008)
        np.testing.assert_allclose(np.cov(draws[:, i].T), covariance[i], atol=.008)


@pytest.mark.parametrize('mode', MODES)
def test_selector_uses_only_paid_labels_and_preserves_budget_and_input(monkeypatch, mode):
    candidates, x, model, sources = fixture()
    original = copy.deepcopy(model.state_dict())
    original_sources = copy.deepcopy(sources)
    fits = []

    def fit(adapted, paid_x, labels, **settings):
        fits.append((adapted.deep, paid_x.copy(), list(labels), settings))
        with torch.no_grad():
            adapted.mean.add_(.02 * len(labels))

    monkeypatch.setattr('methods.baselines.rgpe.search.fit_target', fit)
    oracle = Oracle(candidates)
    result = select(candidates, oracle, pretrained=model, sources=sources, initial_indices=[0, 1],
                    sampling_seed=37, mode=mode, budget=5, device='cpu')
    assert result['selected_indices'] == [i for i, _ in oracle.paid]
    assert len(set(result['selected_indices'])) == result['actual_queries'] == 5
    assert [p for _, p in oracle.paid] == ['initial', 'initial', 'target', 'target', 'target']
    assert result['component_names'] == ['source_a', 'source_b', 'target_fsbo']
    mass = .1 if mode == 'physical_rgpe_classifier' else 0
    for n, row in enumerate(result['queries'], 1):
        assert np.sum(row['prequery_rank_weights']) == pytest.approx(1.)
        if n <= 2:
            assert row['prequery_rank_weights'] == [0., 0., 1.]
        assert row['prequery_ranking']['informative_ordered_pairs'] <= (n - 1)**2
        expected = (1 - mass) * row['prequery_ensemble_probability']
        if mass:
            expected += mass * row['prequery_physical_probability']
        assert row['prequery_probability'] == pytest.approx(expected)
        relevant = [(deep, px, y, settings) for deep, px, y, settings in fits if len(y) == n]
        assert len(relevant) == (2 if mass else 1)
        for deep, px, y, settings in relevant:
            paid = result['selected_indices'][:n]
            np.testing.assert_array_equal(px, x[paid])
            assert y == [i % 2 == 0 for i in paid]
            assert settings == {'steps': 10, 'learning_rate': .003 if deep else .01}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    for name in ('means', 'covariances'):
        np.testing.assert_array_equal(sources[name], original_sources[name])


@pytest.mark.parametrize('mode', MODES)
def test_source_order_and_side_safety_values_do_not_change_selection(monkeypatch, mode):
    candidates, _, model, sources = fixture()
    monkeypatch.setattr('methods.baselines.rgpe.search.fit_target', lambda *a, **k: None)
    reversed_sources = {'names': sources['names'][::-1],
                        'means': sources['means'][::-1].copy(),
                        'covariances': sources['covariances'][::-1].copy()}
    results = [select(candidates, Oracle(candidates, clearance=clearance), pretrained=model,
                      sources=context, initial_indices=[0, 1], sampling_seed=53,
                      mode=mode, budget=5, device='cpu')
               for context, clearance in ((sources, .01), (reversed_sources, 90.))]
    assert results[0]['selected_indices'] == results[1]['selected_indices']
    np.testing.assert_array_equal([r['prequery_probability'] for r in results[0]['queries']],
                                  [r['prequery_probability'] for r in results[1]['queries']])


def test_single_class_rgpe_recovers_target_classifier_with_real_adaptation():
    candidates, _, model, sources = fixture()
    args = {'pretrained': model, 'initial_indices': [0, 1], 'budget': 4, 'device': 'cpu'}
    rgpe = select(candidates, Oracle(candidates, single_class=True), sources=sources, sampling_seed=11, **args)
    fsbo = select_fsbo(candidates, Oracle(candidates, single_class=True), **args)
    assert rgpe['selected_indices'] == fsbo['selected_indices']
    np.testing.assert_allclose([r['prequery_probability'] for r in rgpe['queries']],
                               [r['prequery_probability'] for r in fsbo['queries']], atol=2e-8, rtol=2e-8)


def test_invalid_budget_rejected_before_query_and_spawn_receives_no_private_labels():
    candidates, _, model, sources = fixture()
    oracle = Oracle(candidates)
    with pytest.raises(ValueError, match='initialization'):
        select(candidates, oracle, pretrained=model, sources=sources,
               initial_indices=[0, 0], sampling_seed=11, budget=3, device='cpu')
    assert oracle.paid == []
    executions = {s.scenario_id: PaidExecution(QueryFeedback(s.scenario_id, i % 2 == 0,
                   SafetyMeasurements(.7, 2), .1), [{'evaluator_private': True}])
                  for i, s in enumerate(candidates)}
    ledger = QueryLedger(candidates, 3, ReplayExecutor(executions))
    result = run_isolated(select, {'candidates': candidates, 'pretrained': model, 'sources': sources,
                                  'initial_indices': [0, 1], 'sampling_seed': 11, 'budget': 3,
                                  'device': 'cpu'}, ledger)
    assert result['actual_queries'] == len(ledger.records) == 3
    assert 'evaluator_private' not in str(result) and 'pool_unique_collisions' not in str(result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_batched_classification_folds_agree_between_cpu_and_cuda():
    _, x, model, _ = fixture()
    mean, covariance = model.mean.detach().expand(8), model.covariance(x[:8]).detach()
    indices = torch.arange(8).expand(8, -1)[~torch.eye(8, dtype=torch.bool)].reshape(8, 7)
    y = torch.tensor([0., 1., 1., 0., 1., 0., 0., 1.], dtype=torch.float64)
    cpu = bernoulli_posterior(mean, covariance, indices, y[indices])
    cuda = bernoulli_posterior(mean.cuda(), covariance.cuda(), indices.cuda(), y[indices].cuda())
    torch.testing.assert_close(cuda[0].cpu(), cpu[0], atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(cuda[1].cpu(), cpu[1], atol=2e-10, rtol=2e-10)
