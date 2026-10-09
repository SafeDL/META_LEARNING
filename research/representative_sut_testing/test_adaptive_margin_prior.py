"""Prior evidence timing, censoring mass, positive density and fixed decisions."""
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .adaptive_margin_prior import AdaptiveMarginPrior, event_log_likelihood, mixed_log_likelihood
from .censored_margin_response import calibrated_margin_moments
from .ep_margin_response import ExpectationPropagatedMargin
from .event_report_response import EventReportResponse


def build(evidence_score="mixed"):
    reference = SimpleNamespace(session=SimpleNamespace(gp=SimpleNamespace(covariance=np.eye(12), noise=.1)),
                                head=lambda: 10)
    probability, positive = np.full(12, .8), np.full(12, .4)
    mean, variance = calibrated_margin_moments(probability, positive)
    prior = {"probability": probability, "positive_mean": positive, "mean": mean,
             "report_variance": variance, "modes": np.zeros((12, 6))}
    return AdaptiveMarginPrior(reference, [0] * 12, prior, device="cpu", evidence_score=evidence_score)


def test_mixed_likelihood_uses_event_mass_and_positive_value_density():
    means, variances = np.asarray([-.3, .2]), np.asarray([.5, .1])
    np.testing.assert_allclose(mixed_log_likelihood(means, variances, .5, True), norm.logcdf(0, means, np.sqrt(variances)))
    np.testing.assert_allclose(mixed_log_likelihood(means, variances, .5, False), norm.logpdf(.5, means, np.sqrt(variances)))
    np.testing.assert_array_equal(mixed_log_likelihood(means, variances, .51, True), mixed_log_likelihood(means, variances, 1., True))


def test_online_weights_are_updated_after_observation_not_before():
    model = build()
    prior = model.probabilities().copy()
    model.observe(0, .4, False)
    record = model.evidence_records[-1]
    np.testing.assert_array_equal(record['weights_before'], [.5, .5])
    expected = np.log([.5, .5]) + record['log_predictive_likelihoods']
    np.testing.assert_allclose(record['weights_after'], np.exp(expected - logsumexp(expected)))
    assert record['weights_after'][1] > .5
    assert not np.array_equal(model.probabilities(), prior)


def test_event_scoring_is_normalized_and_risk_still_updates_both_models():
    mean, variance = np.asarray([-2., 3.]), np.asarray([.2, .4])
    np.testing.assert_allclose(np.exp(event_log_likelihood(mean, variance, True))
                               + np.exp(event_log_likelihood(mean, variance, False)), 1.)
    first, second = build("event"), build("event")
    first.observe(0, .1, False)
    second.observe(0, .9, False)
    np.testing.assert_array_equal(first.selection_weights(), second.selection_weights())
    assert any(not np.array_equal(a.mean, b.mean) for a, b in zip(first.models, second.models))
    assert all(len(m.records) == 1 for m in first.models)


def test_event_mixture_log_loss_identity_holds_with_adaptive_component_updates():
    model = build("event")
    mixture, components = 0., np.zeros(2)
    for i in range(10):
        model.observe(i, .15 + .07 * i, bool(i % 3))
        r = model.evidence_records[-1]
        likelihood = np.asarray(r['log_predictive_likelihoods'])
        mixture -= logsumexp(np.log(r['weights_before']) + likelihood)
        components -= likelihood
        np.testing.assert_allclose(mixture, -logsumexp(-components) + np.log(2), atol=1e-10)
        assert mixture <= components.min() + np.log(2) + 1e-10


def test_alternative_report_component_receives_paid_feedback_and_keeps_mixed_audit_anchor():
    base = build()
    prior = {"probability": np.full(12, .8), "positive_mean": np.full(12, .4), "modes": np.zeros((12, 6))}
    mean, variance = calibrated_margin_moments(prior['probability'], prior['positive_mean'])
    prior.update(mean=mean, report_variance=variance)
    report = EventReportResponse(base.reference, [0] * 12, prior, device='cpu')
    model = AdaptiveMarginPrior(base.reference, [0] * 12, prior,
                                model_class=ExpectationPropagatedMargin, evidence_score='event', alternative_model=report, device='cpu')
    model.observe(0, .4, False)
    model.observe(1, .8, False)
    model.observe(2, .9, True)
    result = model.diagnostics()
    assert result['margin_audit_anchor'] == 0
    assert len(result['component_margin_records'][1]) == 3
    assert result['component_inference_updates'][1][-1]['constraints'] == 3
    assert result['component_margin_records'][1][-1]['observed_proxy'] is None
