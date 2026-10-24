"""Virtual-update consistency, causal scoring and candidate reward boundaries."""

import copy

import numpy as np
import torch

from methods.gcrt.converged_lookahead import acquisitions, mix_virtual, virtual_components
from methods.gcrt.converged_response import component_probabilities


def test_virtual_is_real_fixed_kernel_conditioning_and_preserves_inputs():
    means = torch.tensor([[.1, -.3, .5], [-.2, .4, .1]], dtype=torch.float64)
    covariance = torch.tensor([[[1., .2, .1], [.2, 1., .3], [.1, .3, 1.]]] * 2, dtype=torch.float64)
    saved = copy.deepcopy((means, covariance))
    for label in (False, True):
        actual, info = virtual_components(means, covariance, [0], [True], 1, label)
        with torch.no_grad():
            expected, _ = component_probabilities(means, covariance, [0, 1], [True, label])
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
        assert all(row['gradient_max'] <= 1e-10 for row in info)
    torch.testing.assert_close(means, saved[0], atol=0, rtol=0)
    torch.testing.assert_close(covariance, saved[1], atol=0, rtol=0)


def test_virtual_weight_uses_before_likelihood_not_refitted_probability():
    before = np.array([[.1, .8], [.7, .2], [.4, .5]])
    after = np.array([[.99, .6], [.01, .3], [.8, .7]])
    score = np.log(np.array([[.6, .4], [.4, .6]]))
    influence = np.array([1., .25])
    saved = score.copy()
    result = mix_virtual(after, before, score, influence, 0, True)
    manual = []
    for j in range(2):
        weights = [np.exp(score[s, j]) * before[s, 0]**influence[j] for s in range(2)]
        manual.append(.9 * sum(w * after[s, j] for s, w in enumerate(weights)) / sum(weights) + .1 * after[-1, j])
    np.testing.assert_allclose(result, manual, atol=1e-15)
    np.testing.assert_array_equal(score, saved)


def test_horizon_one_is_greedy_and_current_candidate_excluded_from_future():
    p = np.array([.2, .9, .6, .1])
    available = np.array([0, 2, 3])
    conditional = np.full((3, 2, 4), .1)
    for row, index in enumerate(available):
        conditional[row, :, index] = .99
    values = acquisitions(p, conditional, available, (1, 3))
    np.testing.assert_allclose(values[1], p[available], atol=0, rtol=0)
    np.testing.assert_allclose(values[3], p[available] + .2, atol=1e-15)


def test_reordering_learned_sources_preserves_virtual_mixture():
    before = np.array([[.1, .8], [.7, .2], [.4, .5]])
    after = np.array([[.8, .6], [.2, .3], [.8, .7]])
    score = np.log(np.array([[.6, .4], [.4, .6]]))
    influence = np.ones(2)
    original = mix_virtual(after, before, score, influence, 0, True)
    order = [1, 0, 2]
    permuted = mix_virtual(after[order], before[order], score[[1, 0]], influence, 0, True)
    np.testing.assert_allclose(original, permuted, atol=1e-15)
