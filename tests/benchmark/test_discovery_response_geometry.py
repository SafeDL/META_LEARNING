import numpy as np

from benchmark.discovery_response_geometry import geometry_kernels


def test_geometry_views_preserve_covariance_validity_and_signal_variance():
    random = np.random.default_rng(1301)
    factors = [random.normal(size=(7, 4)) for _ in range(3)]
    original, collision, response = [x @ x.T + .01 * np.eye(7) for x in factors]
    views = geometry_kernels(original, collision, response)
    assert views.shape == (4, 7, 7)
    np.testing.assert_allclose(views, views.transpose(0, 2, 1))
    assert np.linalg.eigvalsh(views).min() > 0
    np.testing.assert_allclose(views.diagonal(axis1=-2, axis2=-1).mean(1), np.trace(original) / 7)
    np.testing.assert_array_equal(views[0], original)
    np.testing.assert_allclose(views[3], .5 * (views[0] + views[2]))


def test_geometry_comparison_removes_only_frozen_signal_scale():
    random = np.random.default_rng(1409)
    factors = [random.normal(size=(5, 3)) for _ in range(3)]
    matrices = [x @ x.T + .01 * np.eye(5) for x in factors]
    original = geometry_kernels(*matrices)
    rescaled = geometry_kernels(matrices[0], 9 * matrices[1], .04 * matrices[2])
    np.testing.assert_allclose(original, rescaled, atol=1e-14)
