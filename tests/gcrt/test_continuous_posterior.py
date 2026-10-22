import numpy as np

from methods.gcrt.posterior import CollisionPosterior


def full_prior(size=15):
    x = np.linspace(0, 1, size)
    means = np.array([2 * x - 1, np.sin(3 * x)])
    covariance = np.exp(-((x[:, None] - x[None]) / 0.3)**2) + 1e-5 * np.eye(size)
    return means, np.broadcast_to(covariance, (2, size, size)).copy()


def test_dynamic_expansion_equals_conditioning_the_full_unqueried_prior():
    means, covariances = full_prior()
    full = CollisionPosterior(means, covariances, [0.5, 0.5])
    dynamic = CollisionPosterior(means[:, :8], covariances[:, :8, :8], [0.5, 0.5])
    for index, label in ((0, True), (5, False)):
        full.observe(index, label)
        dynamic.observe(index, label)
    dynamic.extend(means[:, 8:11], covariances[:, 8:11, :8], covariances[:, 8:11, 8:11])
    full.observe(9, True)
    dynamic.observe(9, True)
    before_weights = dynamic.weights.copy()
    dynamic.extend(means[:, 11:], covariances[:, 11:, :11], covariances[:, 11:, 11:])
    np.testing.assert_allclose(dynamic.mean, full.mean, atol=1e-12)
    np.testing.assert_allclose(dynamic.covariance, full.covariance, atol=1e-12)
    np.testing.assert_allclose(dynamic.probabilities(), full.probabilities(), atol=1e-12)
    np.testing.assert_array_equal(dynamic.weights, before_weights)
    assert dynamic.observed == full.observed == {0, 5, 9}
    assert len(dynamic.sites) == 3
    full.observe(14, False)
    dynamic.observe(14, False)
    np.testing.assert_allclose(dynamic.covariance, full.covariance, atol=1e-12)


def test_discarding_unqueried_proposals_preserves_future_paid_updates():
    means, covariances = full_prior()
    full = CollisionPosterior(means, covariances, [0.5, 0.5])
    dynamic = CollisionPosterior(means, covariances, [0.5, 0.5])
    for index, label in ((1, True), (9, False)):
        full.observe(index, label)
        dynamic.observe(index, label)
    keep = [1, 5, 9, 12]
    dynamic.retain(keep)
    assert dynamic.observed == {0, 2}
    full.observe(5, True)
    dynamic.observe(1, True)
    np.testing.assert_allclose(dynamic.mean, full.mean[:, keep], atol=1e-12)
    np.testing.assert_allclose(dynamic.covariance, full.covariance[:, keep][:, :, keep], atol=1e-12)
    np.testing.assert_allclose(dynamic.probabilities(), full.probabilities(keep), atol=1e-12)
