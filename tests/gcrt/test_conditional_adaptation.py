import copy

import numpy as np
import pytest
import torch

from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target, laplace
from methods.gcrt.conditional_adaptation import component_probabilities, conditional_prior, fit_conditional
from methods.gcrt.joint_response import logistic_laplace


@pytest.mark.parametrize('label_kind', ['mixed', 'zero', 'one'])
def test_no_source_evidence_gradient_and_adaptation_match_target_gp(label_kind):
    torch.set_num_threads(1)
    x = np.random.default_rng(19).uniform(size=(127, 3))
    labels = np.arange(127) % 3 == 0
    if label_kind != 'mixed':
        labels[:] = label_kind == 'one'
    original = ClassificationKernel(3)
    alternative = copy.deepcopy(original)
    old_loss = laplace(original, x, labels)[0]
    mean, covariance = conditional_prior(alternative, x)
    new_loss = logistic_laplace(mean, covariance, labels)[0].sum()
    torch.testing.assert_close(new_loss, old_loss, atol=2e-10, rtol=2e-10)
    old_loss.backward()
    new_loss.backward()
    for a, b in zip(original.parameters(), alternative.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-9, rtol=2e-9)
    fit_target(original, x, labels, steps=10, learning_rate=.003)
    losses = fit_conditional(alternative, x, labels, [1.], steps=10, learning_rate=.003)
    assert np.isfinite(losses).all()
    for a, b in zip(original.parameters(), alternative.parameters()):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
    with torch.no_grad():
        mean, covariance = conditional_prior(alternative, x)
        p = component_probabilities(mean, covariance, list(range(127)), labels)[0]
        expected = collision_probabilities(original, x, labels, x)
    torch.testing.assert_close(p, expected, atol=2e-9, rtol=2e-9)
    assert bool(((p >= 0) & (p <= 1)).all())


def test_source_permutation_and_context_are_preserved():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(11)
    x = torch.rand(17, 3, generator=generator, dtype=torch.float64)
    means = torch.randn(3, 17, generator=generator, dtype=torch.float64)
    factors = torch.randn(3, 17, 5, generator=generator, dtype=torch.float64)
    covariances = factors @ factors.transpose(-1, -2)/20
    before_means, before_covariances = means.clone(), covariances.clone()
    weights = torch.tensor([.2, .3, .25, .25], dtype=torch.float64)
    before_weights = weights.clone()
    labels = np.arange(17) % 2 == 0
    model = ClassificationKernel(3)
    permuted = copy.deepcopy(model)
    order = torch.tensor([2, 0, 1])
    fit_conditional(model, x, labels, weights, source_means=means,
                    source_covariances=covariances, reference_mean=0.)
    fit_conditional(permuted, x, labels, torch.cat((weights[order], weights[-1:])),
                    source_means=means[order], source_covariances=covariances[order], reference_mean=0.)
    for a, b in zip(model.parameters(), permuted.parameters()):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
        assert a.grad is not None and bool(torch.isfinite(a.grad).all())
    torch.testing.assert_close(means, before_means, atol=0, rtol=0)
    torch.testing.assert_close(covariances, before_covariances, atol=0, rtol=0)
    torch.testing.assert_close(weights, before_weights, atol=0, rtol=0)
    with torch.no_grad():
        m, c = conditional_prior(model, x, means, covariances, 0.)
        expected = component_probabilities(m, c, [], [])
        m2, c2 = conditional_prior(permuted, x, means[order], covariances[order], 0.)
        actual = component_probabilities(m2, c2, [], [])
    torch.testing.assert_close(actual, torch.cat((expected[order], expected[-1:])), atol=2e-9, rtol=2e-9)
    assert m.shape == (4, 17) and c.shape == (4, 17, 17)
    assert bool((torch.linalg.eigvalsh(c) > -1e-9).all())


def test_initialization_reference_keeps_source_means():
    x = np.random.default_rng(23).uniform(size=(8, 2))
    model = ClassificationKernel(2)
    with torch.no_grad():
        model.mean.fill_(-3.7)
    means = torch.zeros(2, 8, dtype=torch.float64)
    covariances = torch.eye(8, dtype=torch.float64)[None].repeat(2, 1, 1)
    m, _ = conditional_prior(model, x, means, covariances, float(model.mean.detach()))
    torch.testing.assert_close(m[:-1], means)


@pytest.mark.parametrize('label', [False, True])
def test_source_conditional_fit_has_finite_gradients_at_127_support_points(label):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(37)
    x = torch.rand(127, 3, generator=generator, dtype=torch.float64)
    means = torch.randn(8, 127, generator=generator, dtype=torch.float64)
    factors = torch.randn(8, 127, 9, generator=generator, dtype=torch.float64)
    covariances = factors @ factors.transpose(-1, -2)/30
    model = ClassificationKernel(3)
    labels = np.full(127, label)
    losses = fit_conditional(model, x, labels, np.r_[np.full(8, .75/8), .25],
        source_means=means, source_covariances=covariances, reference_mean=0.)
    assert np.isfinite(losses).all()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())
    with torch.no_grad():
        m, c = conditional_prior(model, x, means, covariances, 0.)
        p = component_probabilities(m, c, list(range(127)), labels)
    assert p.shape == (9, 127) and bool(torch.isfinite(p).all())
    assert bool(((p >= 0) & (p <= 1)).all())
