import numpy as np
import pytest
from threadpoolctl import threadpool_limits
import torch

from benchmark.discovery_conditional_adaptation import probabilities_numpy
from methods.common.gp_classifier import ClassificationKernel
from methods.gcrt.conditional_adaptation import component_probabilities
from methods.gcrt.residual_scale import cached_covariance, residual_prior, update_reliability


def test_cached_deep_kernel_matches_the_complete_candidate_batch():
    torch.manual_seed(23)
    model = ClassificationKernel(6, deep=True)
    x = np.random.default_rng(37).uniform(size=(512, 6))
    chosen = np.random.default_rng(1301).permutation(512)[:160]
    with torch.no_grad():
        encoded = model.features(x)[chosen]
        actual = cached_covariance(encoded, model.log_lengthscale, model.log_scale)
        expected = model.covariance(x)[chosen][:, chosen]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)


def test_residual_scale_changes_only_the_historical_residual_covariance():
    covariance = torch.tensor([[1., .3], [.3, 1.]], dtype=torch.float64)
    sources = torch.stack((.2*covariance, .4*covariance))
    means = torch.tensor([[-2., -1.], [-3., -2.]], dtype=torch.float64)
    target_mean = torch.tensor(-1.4, dtype=torch.float64)
    before = means.clone(), sources.clone()
    m1, c1 = residual_prior(target_mean, covariance, means, sources, -1.5, 1.)
    m0, c0 = residual_prior(target_mean, covariance, means, sources, -1.5, 0.)
    torch.testing.assert_close(m1, m0)
    torch.testing.assert_close(m1[:-1], means+.1)
    torch.testing.assert_close(c1[:-1], sources+covariance[None])
    torch.testing.assert_close(c0[:-1], sources)
    torch.testing.assert_close(c1[-1], covariance)
    torch.testing.assert_close(c0[-1], covariance)
    torch.testing.assert_close(means, before[0], atol=0, rtol=0)
    torch.testing.assert_close(sources, before[1], atol=0, rtol=0)
    with pytest.raises(ValueError, match='nonnegative'):
        residual_prior(target_mean, covariance, means, sources, -1.5, -.1)


def test_prequery_reliability_is_normalized_permutation_invariant_and_functional():
    x = torch.tensor([[0., 0.], [.2, .7], [.9, .9]], dtype=torch.float64)
    probabilities = torch.tensor([[.7, .2, .1], [.3, .8, .1], [.4, .3, .5]], dtype=torch.float64)
    initial = torch.tensor([.3, .45, .25], dtype=torch.float64).log()[:, None].expand(-1, 3).clone()
    before = initial.clone()
    order = torch.tensor([2, 0, 1])
    local = update_reliability(initial, probabilities, 1, True, x, .3)
    permuted = update_reliability(initial[order], probabilities[order], 1, True, x, .3)
    torch.testing.assert_close(permuted, local[order])
    torch.testing.assert_close(local.exp().sum(0), torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(initial, before, atol=0, rtol=0)
    same_x = torch.zeros_like(x)
    global_weights = update_reliability(initial, probabilities, 1, True, x, None)
    uniform_influence = update_reliability(initial, probabilities, 1, True, same_x, .3)
    torch.testing.assert_close(global_weights, uniform_influence)
    expected = initial[:, 1].exp()*probabilities[:, 1]
    torch.testing.assert_close(global_weights[:, 1].exp(), expected/expected.sum())


@pytest.mark.parametrize('beta', [0., .1, .3, 1., 3.])
def test_residual_prediction_at_128_support_points_matches_numpy(beta):
    torch.set_num_threads(1)
    rng = np.random.default_rng(53)
    x = rng.uniform(size=(160, 6))
    labels = rng.random(128) < .2
    model = ClassificationKernel(6)
    with torch.no_grad():
        covariance = model.covariance(x)
        means = torch.tensor(-2+rng.normal(size=(7, 160))*.3, dtype=torch.float64)
        covariances = torch.stack([(.02+.02*i)*covariance for i in range(7)])
        m, c = residual_prior(model.mean, covariance, means, covariances, -.1, beta)
        probability = component_probabilities(m, c, list(range(128)), labels).numpy()
    with threadpool_limits(1):
        independent = probabilities_numpy(m.numpy(), c.numpy(), list(range(128)), labels)
    np.testing.assert_allclose(probability, independent, atol=2e-9, rtol=2e-8)
    assert probability.shape == (8, 160)
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
