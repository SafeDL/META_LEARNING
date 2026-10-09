"""Check the identity control, covariance validity and conditional inference."""
import numpy as np
import pytest
import torch

from .model import joint_prior
from .response_projection import apply_projection, projected_prior
from .session import AdaptiveTestingSession
from .test_method import example


@pytest.mark.parametrize("full", [False, True])
def test_identity_projection_and_equal_means_recover_current_prior(full):
    x, risks, collisions, options = example()
    options.update({
        "transform": "logit",
        "collision_transform": "probit",
        "residual_mode": "learned_sensitivity",
        "positive_loading": True,
        "loading": [[0, 0, 0, 0, 0, 1]] * 2,
        "collision_offset": [-0.1, -0.2]
    })
    identity = torch.eye(6, dtype=torch.float64).repeat(2, 1, 1)
    logits = torch.zeros((2, 6), dtype=torch.float64)
    expected = joint_prior(x,
                           risks,
                           collisions,
                           options,
                           "cpu",
                           full_collision=full)
    actual = projected_prior(x,
                             risks,
                             collisions,
                             options,
                             identity,
                             logits, [0, 2, 4],
                             full_collision=full)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, atol=1e-14, rtol=0)


def test_arbitrary_projection_preserves_psd_and_excluded_source_gradients():
    x, risks, collisions, options = example()
    options.update({
        "transform": "logit",
        "collision_transform": "probit",
        "collision_offset": [0, 0]
    })
    generator = torch.Generator().manual_seed(11)
    projection = torch.randn((2, 6, 6),
                             dtype=torch.float64,
                             generator=generator,
                             requires_grad=True)
    logits = torch.randn((2, 6),
                         dtype=torch.float64,
                         generator=generator,
                         requires_grad=True)
    _, mc, rr, cr, cc, _ = projected_prior(x, risks, collisions, options,
                                           projection, logits, [0, 2, 4], True)
    joint = torch.cat((torch.cat((rr, cr.T), 1), torch.cat((cr, cc), 1)), 0)
    assert torch.linalg.eigvalsh(joint).min() > -1e-12
    assert torch.count_nonzero(cr[:2, 2:]) == 0
    (mc.square().sum() + cr.square().sum() + cc.square().sum()).backward()
    assert torch.isfinite(projection.grad).all() and torch.isfinite(
        logits.grad).all()
    assert torch.count_nonzero(projection.grad[:, [1, 3, 5]]) == 0
    assert torch.count_nonzero(projection.grad[:, :, [1, 3, 5]]) == 0
    assert torch.count_nonzero(logits.grad[:, [1, 3, 5]]) == 0


def test_projected_session_matches_batch_risk_conditioning():
    x, risks, collisions, options = example()
    options.update({
        "transform": "logit",
        "collision_transform": "probit",
        "collision_offset": [0.1, -0.1]
    })
    generator = torch.Generator().manual_seed(23)
    projection = torch.randn((2, 3, 3),
                             dtype=torch.float64,
                             generator=generator)
    logits = torch.randn((2, 3), dtype=torch.float64, generator=generator)
    prior = projected_prior(x, risks, collisions, options, projection, logits,
                            [0, 1, 2])
    model = AdaptiveTestingSession(x,
                                   risks,
                                   collisions,
                                   options,
                                   budget=2,
                                   device="cpu")
    apply_projection(model, x, (risks, collisions), {
        "projection": projection.tolist(),
        "mean_logits": logits.tolist()
    })
    for index, value in ((0, 0.6), (1, 0.8)):
        model.pending = index
        model.records.append({})
        model.observe(value)
    mr, mc, rr, cr, cc, _ = prior
    covariance = rr[:2, :2] + options["noise"] * torch.eye(2, dtype=rr.dtype)
    residual = torch.logit(torch.tensor([0.6, 0.8], dtype=rr.dtype)) - mr[:2]
    expected_mean = mc + cr[:, :2] @ torch.linalg.solve(covariance, residual)
    expected_variance = cc - (
        cr[:, :2] * torch.linalg.solve(covariance, cr[:, :2].T).T).sum(1)
    torch.testing.assert_close(model.mean_c, expected_mean, atol=1e-12, rtol=0)
    torch.testing.assert_close(model.cc, expected_variance, atol=1e-12, rtol=0)
