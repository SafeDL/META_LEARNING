import numpy as np
import pytest
import torch
from scipy.linalg import cho_solve

from methods.srd_tnp_bqd.common import DEFAULT_CONFIG, config_at
from methods.srd_tnp_bqd.nonstationary_kernel import ResidualKernel
from methods.srd_tnp_bqd.discrepancy import DiscrepancyGP, PoolDiscrepancyGP, conditional_distribution, gaussian_nll


@pytest.fixture
def sample():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    x = rng.random((12, 4))
    h = rng.normal(size=(12, 32))
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    kernel = ResidualKernel(config_at(DEFAULT_CONFIG)["residual_gp"])
    yield kernel, x, h, rng.normal(size=(12, 1))
    torch.set_num_threads(previous)


def test_kernel_psd_and_symmetry(sample):
    k, x, h, _ = sample
    matrix = k.matrix(x, h, x, h).detach().numpy()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-10


def test_kernel_diagonal_positive(sample):
    k, x, h, _ = sample
    a = k.matrix(x, h, x, h).detach().numpy()
    assert np.diag(a).min() > 0
    np.testing.assert_allclose(np.diag(a), k.diagonal(x, h).detach().numpy(), atol=1e-10)


def test_embedding_changes_cross_covariance(sample):
    k, x, h, _ = sample
    assert not torch.allclose(k.matrix(x, h, x, h), k.matrix(x, np.roll(h, 1, 0), x, h))


def test_scale_gate_depends_on_h_and_x(sample):
    k, x, h, _ = sample
    tx, th = torch.tensor(x, requires_grad=True), torch.tensor(h, requires_grad=True)
    weights = k.scale_weights(tx, th)
    gx, gh = torch.autograd.grad(weights[:, 0].sum(), [tx, th])
    assert gx.abs().max() > 1e-8 and gh.abs().max() > 1e-8
    response = torch.exp(-torch.cdist(th, th).square() / (2 * torch.nn.functional.softplus(k.raw_h_length).square()))
    assert torch.autograd.grad(response.sum(), th)[0].abs().max() > 1e-8


def test_local_path_survives_unrelated_embeddings(sample):
    k, x, h, _ = sample
    h = h * 1e6
    assert k.matrix(x[:1], h[:1], x[1:2], h[1:2]).item() > 0
    assert k.sigma_local.item() >= .1


def test_h_permutation_negative_control(sample):
    k, x, h, _ = sample
    assert not torch.allclose(k.matrix(x, h, x, h), k.matrix(x, h[::-1].copy(), x, h[::-1].copy()))


def test_zero_target_data_recovers_prior(sample):
    k, x, h, m = sample
    p = DiscrepancyGP(k).predict(x, h, m)
    np.testing.assert_array_equal(p.mean, m)
    assert p.latent_var.min() > 0


def test_gp_reduces_variance_under_fixed_kernel(sample):
    k, x, h, m = sample
    gp = DiscrepancyGP(k)
    before = gp.predict(x, h, m)
    gp.observe(0, x[0], h[0], m[0], .7)
    after = gp.predict(x, h, m)
    assert np.all(after.latent_var <= before.latent_var + 1e-10)
    assert after.latent_var[0] < .002


def test_posterior_matches_reference_cholesky(sample):
    k, x, h, m = sample
    gp = DiscrepancyGP(k)
    for i, r in enumerate([.2, .7, .9]):
        gp.observe(i, x[i], h[i], m[i], r)
    p = gp.predict(x, h, m)
    mat = k.matrix(x, h, x, h).detach().numpy()
    factor = np.linalg.cholesky(mat[:3, :3] + (gp.noise + gp.used_jitter) * np.eye(3))
    expected_m = m + mat[:, :3] @ cho_solve((factor, True), np.array(gp.e)[:, None])
    expected_v = np.diag(mat) - np.sum(mat[:, :3] * cho_solve((factor, True), mat[:3]).T, axis=1)
    np.testing.assert_allclose(p.mean, expected_m, atol=1e-9)
    np.testing.assert_allclose(p.latent_var[:, 0], expected_v, atol=1e-9)


def test_sequential_equals_batch_conditioning(sample):
    test_posterior_matches_reference_cholesky(sample)


def test_fixed_history_after_every_observation(sample):
    k, x, h, m = sample
    saved = {name: v.clone() for name, v in k.state_dict().items()}
    gp = DiscrepancyGP(k)
    original = m.copy(), h.copy()
    for i in range(5):
        gp.observe(i, x[i], h[i], m[i], .6)
        assert all(torch.equal(saved[n], v) for n, v in k.state_dict().items())
        np.testing.assert_array_equal(m, original[0])
        np.testing.assert_array_equal(h, original[1])


def test_no_nan_near_duplicate_inputs(sample):
    k, x, h, m = sample
    x[1], h[1] = x[0] + 1e-14, h[0]
    gp = DiscrepancyGP(k)
    gp.observe(0, x[0], h[0], m[0], .4)
    gp.observe(1, x[1], h[1], m[1], .4)
    assert np.isfinite(gp.predict(x, h, m).latent_var).all()
    with pytest.raises(ValueError):
        gp.observe(1, x[1], h[1], m[1], .5)


def test_joint_conditional_nll_backpropagates_to_h(sample):
    k, x, h, m = sample
    th = torch.tensor(h, requires_grad=True)
    tm = torch.tensor(m, requires_grad=True)
    mean, cov = conditional_distribution(k, x, th, tm, torch.ones(3, 1), 3)
    loss, _ = gaussian_nll(mean, cov, torch.zeros(9, 1))
    loss.backward()
    assert th.grad.abs().max() > 1e-8
    assert k.gate[0].weight.grad.abs().max() > 1e-8


def test_pool_cholesky_matches_reference_at_every_query(sample):
    k, x, h, m = sample
    pool = PoolDiscrepancyGP(k, x, h, m)
    full = DiscrepancyGP(k)
    for i, r in enumerate([.2, .7, .9, .4, .5]):
        pool.observe(i, r)
        full.observe(i, x[i], h[i], m[i], r)
        a, b = pool.predict(), full.predict(x, h, m)
        np.testing.assert_allclose(a.mean, b.mean, atol=1e-9)
        np.testing.assert_allclose(a.latent_var, b.latent_var, atol=1e-9)


def test_uncertain_mean_matches_joint_gaussian_conditioning(sample):
    k, x, h, m = sample
    config = {"offset_variance": 1., "scale_variance": .25}
    pool = PoolDiscrepancyGP(k, x, h, m, mean_calibration=config)
    basis = np.column_stack((np.ones(len(x)), m[:, 0]))
    prior_covariance = np.diag([1., .25])
    covariance = k.matrix(x, h, x, h).detach().numpy() + basis @ prior_covariance @ basis.T
    prior_cross = prior_covariance @ basis.T
    indices, values = [], []
    for index, risk in zip((0, 3, 8, 2, 11), (.2, .7, .9, .4, .5)):
        pool.observe(index, risk)
        indices.append(index)
        values.append(np.log(risk / (1 - risk)))
        observed = covariance[np.ix_(indices, indices)] + (pool.noise + pool.jitter) * np.eye(len(indices))
        factor = np.linalg.cholesky(observed)
        residual = np.array(values) - m[indices, 0]
        cross = covariance[:, indices]
        expected_mean = m[:, 0] + cross @ cho_solve((factor, True), residual)
        expected_var = np.diag(covariance) - np.sum(cross * cho_solve((factor, True), cross.T).T, axis=1)
        coefficient_cross = prior_cross[:, indices]
        expected_coefficients = np.array([0., 1.]) + coefficient_cross @ cho_solve((factor, True), residual)
        expected_coefficient_covariance = prior_covariance - coefficient_cross @ cho_solve((factor, True), coefficient_cross.T)
        np.testing.assert_allclose(pool.predict().mean[:, 0], expected_mean, atol=1e-9)
        np.testing.assert_allclose(pool.predict().latent_var[:, 0], expected_var, atol=1e-9)
        np.testing.assert_allclose(pool.coefficient_mean, expected_coefficients, atol=1e-9)
        np.testing.assert_allclose(pool.coefficient_covariance, expected_coefficient_covariance, atol=1e-9)


def test_zero_mean_uncertainty_recovers_original_pool(sample):
    k, x, h, m = sample
    calibrated = PoolDiscrepancyGP(k, x, h, m, mean_calibration={"offset_variance": 0., "scale_variance": 0.})
    original = PoolDiscrepancyGP(k, x, h, m)
    for index, risk in enumerate((.2, .7, .9)):
        calibrated.observe(index, risk)
        original.observe(index, risk)
        np.testing.assert_array_equal(calibrated.predict().mean, original.predict().mean)
        np.testing.assert_array_equal(calibrated.predict().latent_var, original.predict().latent_var)


def test_target_feedback_can_withdraw_historical_mean(sample):
    k, x, h, _ = sample
    m = np.linspace(-6., 6., len(x))[:, None]
    pool = PoolDiscrepancyGP(k, x, h, m, mean_calibration={"offset_variance": 1., "scale_variance": .25})
    for index in range(len(x)):
        pool.observe(index, .5)
    assert abs(pool.coefficient_mean[1]) < .1
    assert pool.coefficient_covariance[1, 1] < .25
