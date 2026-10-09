"""Exact event-report moments, feedback-order stability and probit equivalence."""
from types import SimpleNamespace

import numpy as np
import torch

from .censored_margin_response import neutral_margin_prior
from .ep_margin_response import ExpectationPropagatedMargin
from .event_report_response import EventReportResponse, linear_constraint_ep


def setup():
    covariance = np.asarray([[1., .6, .2, .1], [.6, 1., .3, .2], [.2, .3, 1., .7], [.1, .2, .7, 1.]])
    reference = SimpleNamespace(session=SimpleNamespace(gp=SimpleNamespace(covariance=covariance, noise=.1)), head=lambda: 3)
    prior = {"probability": np.full(4, .8), "positive_mean": np.full(4, .4), "modes": np.zeros((4, 6))}
    return reference, prior


def build():
    reference, prior = setup()
    return EventReportResponse(reference, [0, 0, 0, 1], prior, device="cpu")


def test_one_pairwise_halfspace_matches_exact_gaussian_moments():
    mean, covariance, *_ = linear_constraint_ep(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64),
        torch.arange(2), torch.tensor([[1., -1.]], dtype=torch.float64), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    np.testing.assert_allclose(mean.numpy(), [1 / np.sqrt(np.pi), -1 / np.sqrt(np.pi)], atol=2e-6)
    np.testing.assert_allclose(covariance.numpy(), [[1 - 1 / np.pi, 1 / np.pi], [1 / np.pi, 1 - 1 / np.pi]], atol=2e-6)




def test_event_report_ignores_numeric_risk_feedback():
    records = [{"index": i, "risk": r, "collision": c} for i, (r, c) in enumerate([(.1, False), (.5, False), (.8, True)])]
    transformed = [{**q, "risk": 1 - q['risk']} for q in records]
    first, second = build(), build()
    first.condition(records)
    second.condition(transformed)
    np.testing.assert_array_equal(first.mean.numpy(), second.mean.numpy())
    np.testing.assert_array_equal(first.covariance.numpy(), second.covariance.numpy())


def test_sequential_orders_agree_with_canonical_batch_and_preserve_psd():
    records = [{"index": i, "risk": r, "collision": c} for i, (r, c) in enumerate([(.1, False), (.5, False), (.8, True)])]
    forward, reverse, batch = build(), build(), build()
    for q in records: forward.observe(q['index'], q['risk'], q['collision'])
    for q in reversed(records): reverse.observe(q['index'], q['risk'], q['collision'])
    batch.condition(records)
    for other in [reverse, batch]:
        np.testing.assert_allclose(forward.mean.numpy(), other.mean.numpy(), atol=2e-5)
        np.testing.assert_allclose(forward.covariance.numpy(), other.covariance.numpy(), atol=2e-5)
    assert np.linalg.eigvalsh(batch.covariance.numpy()).min() >= -1e-10


def test_sign_only_report_ep_matches_probit_ep_at_unqueried_points():
    reference, prior = setup()
    original = ExpectationPropagatedMargin(reference, [0, 0, 0, 1], neutral_margin_prior(prior), continuous_safe=False, device='cpu')
    report = build()
    data = [{"index": 0, "risk": .2, "collision": False}, {"index": 1, "risk": .8, "collision": True}]
    original.condition(data)
    report.condition(data)
    np.testing.assert_allclose(original.probabilities()[2:], report.probabilities()[2:], atol=2e-6)
