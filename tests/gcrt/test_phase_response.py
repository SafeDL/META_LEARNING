from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import expit
import torch

from methods.common.gp_classifier import ClassificationKernel
from methods.gcrt.phase_response import latent_posterior, phase_bases, phase_labels, phase_support, total_collision_probability


@pytest.mark.parametrize('time,onset,complete,expected', [
    (.5, None, None, (1, None, None)),
    (1.49, 1.5, None, (1, None, None)),
    (1.5, 1.5, None, (0, 1, None)),
    (2., 1.5, 3., (0, 1, None)),
    (3., 1.5, 3., (0, 0, 1)),
    (9., 1.5, 3., (0, 0, 1)),
])
def test_collision_censors_later_phases(time, onset, complete, expected):
    events = {'measured_background_onset_s': onset, 'background_complete_s': complete}
    assert phase_labels(events, time, time) == expected


@pytest.mark.parametrize('end,onset,complete,expected', [
    (.5, None, None, (None, None, None)),
    (2., 1.5, None, (0, None, None)),
    (5., 1.5, 3., (0, 0, None)),
    (10., 1.5, 3., (0, 0, 0)),
])
def test_safe_incomplete_execution_is_censored(end, onset, complete, expected):
    events = {'measured_background_onset_s': onset, 'background_complete_s': complete}
    assert phase_labels(events, None, end) == expected


def test_phase_support_uses_only_paid_indices_and_keeps_missing_labels_out():
    observations = np.array([[1., np.nan, np.nan], [0., 0., 1.], [0., 0., 0.]])
    paid = [0, 2]
    indices, labels = phase_support(paid, observations, 2)
    assert indices == [2] and labels.tolist() == [False]
    changed = observations.copy()
    changed[1] = [1., 1., 1.]
    again, again_labels = phase_support(paid, changed, 2)
    assert again == indices and np.array_equal(again_labels, labels)
    filled_indices, filled = phase_support(paid, observations, 2, zero_fill=True)
    assert filled_indices == paid and filled.tolist() == [False, False]


def test_survival_probability_edges_and_monotonicity():
    hazards = torch.tensor([[0., 1., .2], [0., 0., .3], [0., 0., .4]], dtype=torch.float64)
    actual = total_collision_probability(hazards)
    torch.testing.assert_close(actual, torch.tensor([0., 1., 1-.8*.7*.6], dtype=torch.float64))
    increased = hazards.clone()
    increased[2, 2] = .8
    assert bool((total_collision_probability(increased) >= actual).all())


def test_historical_latent_posterior_matches_independent_numpy_full_covariance():
    torch.set_num_threads(1)
    rng = np.random.default_rng(53)
    support, query = rng.uniform(size=(127, 3)), rng.uniform(size=(31, 3))
    labels = rng.random(127) < .15
    model = ClassificationKernel(3)
    with torch.no_grad():
        mean, covariance = latent_posterior(model, support, labels, query)
        gram = model.covariance(support).numpy()+1e-5*np.eye(127)
        cross = model.covariance(support, query).numpy()
        query_covariance = model.covariance(query).numpy()
    f = np.zeros(127)
    for _ in range(10):
        p = expit(f)
        weight = np.maximum(p*(1-p), 1e-9)
        root = np.sqrt(weight)
        factor = cholesky(np.eye(127)+root[:, None]*gram*root[None], lower=True)
        b = weight*f+labels-p
        f = gram @ (b-root*cho_solve((factor, True), root*(gram @ b)))
    p = expit(f)
    root = np.sqrt(np.maximum(p*(1-p), 1e-9))
    factor = cholesky(np.eye(127)+root[:, None]*gram*root[None], lower=True)
    expected_mean = cross.T @ cho_solve((cholesky(gram, lower=True), True), f)
    solved = solve_triangular(factor, root[:, None]*cross, lower=True)
    expected_covariance = query_covariance-solved.T @ solved
    np.testing.assert_allclose(mean.numpy(), expected_mean, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(covariance.numpy(), expected_covariance, atol=2e-9, rtol=2e-9)
    assert np.linalg.eigvalsh(covariance.numpy()).min() > -1e-9


def test_phase_base_mean_uses_only_observed_historical_risk_sets():
    history = [SimpleNamespace(source_id='a', family='rule', role='history'),
               SimpleNamespace(source_id='b', family='learning', role='history')]
    observations = {'a': np.array([[1., np.nan, np.nan], [0., 1., np.nan], [0., 0., 0.]]),
                    'b': np.array([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]])}
    pretrained = ClassificationKernel(3)
    bases = phase_bases(pretrained, history, observations)
    np.testing.assert_allclose([b.mean.sigmoid().item() for b in bases], [.25, 2/7, 1/3])
    assert pretrained.mean.item() == 0.
    with pytest.raises(ValueError, match='historical sources only'):
        phase_bases(pretrained, [SimpleNamespace(role='development')], observations)
