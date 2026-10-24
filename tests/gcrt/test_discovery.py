import copy

import numpy as np
import pytest
import torch

from methods.common.classification import site_derivatives
from methods.gcrt.discovery import LocalCollisionPosterior, discovery_posterior
from methods.gcrt.posterior import CollisionPosterior
from methods.gcrt.selection import acquisition_values, batched_acquisition


def states(bandwidth=0.3):
    x = np.linspace(0, 1, 14)[:, None]
    means = np.stack((3 * x[:, 0] - 1.5, 1.5 - 3 * x[:, 0], np.zeros(len(x))))
    kernel = np.exp(-((x - x.T) / 0.2) ** 2) + 1e-6 * np.eye(len(x))
    covariances = np.stack((kernel, 0.8 * kernel, 1.2 * kernel))
    weights = np.array([0.375, 0.375, 0.25])
    local = LocalCollisionPosterior(means, covariances, weights, x=x, bandwidth=bandwidth,
                                     component_names=['left', 'right', 'target'])
    return local, CollisionPosterior(means, covariances, weights)


def test_local_evidence_is_prequery_and_spatially_weighted():
    state, _ = states()
    expected = site_derivatives(state.mean[:, 0], state.covariance[:, 0, 0], True)[0]
    state.observe(0, True)
    np.testing.assert_array_equal(state.predictive_evidence[0], expected)
    logits = state.log_prior[:, None] + expected[:, None] * state.reliability_influence([0], [0, 13])
    weights = np.exp(state.log_weights_at([0, 13]))
    np.testing.assert_allclose(weights, np.exp(logits) / np.exp(logits).sum(0))
    np.testing.assert_allclose(weights.sum(0), 1)
    assert abs(weights[0, 1] - 0.375) < abs(weights[0, 0] - 0.375)
    with pytest.raises(ValueError):
        state.observe(0, False)


def test_constant_influence_matches_global_for_observations_and_fantasies():
    local, global_state = states(np.inf)
    for index, label in ((0, True), (4, False), (10, True)):
        assert local.observe(index, label) == pytest.approx(global_state.observe(index, label))
        np.testing.assert_allclose(local.probabilities(), global_state.probabilities(), atol=1e-13)
    np.testing.assert_allclose(acquisition_values(local, [1, 2, 3, 5], 4, device='cpu'),
                               acquisition_values(global_state, [1, 2, 3, 5], 4, device='cpu'))


def test_local_virtual_update_matches_actual_without_mutating_evidence():
    state, _ = states()
    state.observe(0, True)
    before = copy.deepcopy(state)
    for label in (False, True):
        expected = copy.deepcopy(state)
        expected.observe(5, label)
        np.testing.assert_allclose(state.conditional_probabilities(5, label, [3, 7, 13]),
                                   expected.probabilities([3, 7, 13]), atol=1e-13)
    for name in ('mean', 'covariance', 'local_log_weight', 'log_weight'):
        np.testing.assert_array_equal(getattr(state, name), getattr(before, name))
    assert state.evidence_indices == before.evidence_indices == [0]
    np.testing.assert_array_equal(state.predictive_evidence, before.predictive_evidence)


def test_source_permutation_preserves_local_predictions():
    state, _ = states()
    order = [2, 0, 1]
    permuted = LocalCollisionPosterior(state.mean[order], state.covariance[order],
                                      state.weights[order], x=state.x, bandwidth=state.bandwidth,
                                      component_names=[state.component_names[i] for i in order])
    for index, label in ((0, True), (4, False), (10, True)):
        state.observe(index, label)
        permuted.observe(index, label)
    np.testing.assert_allclose(state.probabilities(), permuted.probabilities())


@pytest.mark.parametrize('horizon', [1, 5, 11])
def test_batched_local_and_reference_acquisition_agree(horizon):
    state, _ = states()
    state.observe(0, True)
    state.observe(4, False)
    available = [i for i in range(14) if i not in state.observed]
    expected = acquisition_values(state, available, horizon, device='cpu')
    if horizon == 1:
        np.testing.assert_allclose(expected, state.probabilities(available))
    else:
        np.testing.assert_allclose(batched_acquisition(state, available, horizon, device='cpu'),
                                   expected, atol=1e-12)
        if torch.cuda.is_available():
            np.testing.assert_allclose(batched_acquisition(state, available, horizon, device='cuda'),
                                       expected, atol=1e-11)


def test_source_specific_kernels_match_own_encodings_and_are_psd():
    from benchmark.scenarios import sample
    from methods.common.response import ResponseArchive, ResponseEncoder
    scenes = sample('cut_in', 24, 'history')
    contexts = [ResponseArchive(str(i), 'rule', str(i), 'history', scenes,
                               np.arange(24) % (i + 2) == 0, np.ones((24, 2))) for i in range(2)]
    encoder = ResponseEncoder(contexts[0].x.shape[1]).eval().requires_grad_(False)
    x = contexts[0].x[:8]
    state = discovery_posterior(encoder, x, contexts, independent=True)
    with torch.no_grad():
        phi = encoder.encode(x, contexts).double()
        expected = encoder.physical_kernel(x).double()[None] + encoder.log_response_scale.exp().double().square() * (phi @ phi.transpose(-1, -2))
    np.testing.assert_allclose(state.covariance[:2], expected.numpy() + 1e-6 * np.eye(8), atol=1e-12)
    assert np.linalg.eigvalsh(state.covariance).min() > 0
    assert state.weights[-1] == pytest.approx(0.25)


def test_independent_reconstruction_detects_corrupted_predictions_and_weights():
    from benchmark.discovery_audit import reconstruct_evidence
    initial, _ = states()
    online = copy.deepcopy(initial)
    queries = []
    for i,label in ((0,True),(4,False),(10,True)):
        probability = float(online.probabilities([i])[0])
        evidence = online.observe(i,label)
        queries.append({'query':len(queries)+1,'index':i,'feedback':{'collision':label},
                        'prequery_probability':probability,'log_evidence':evidence,
                        'source_weights':np.exp(online.log_weights_at([i])[:,0]).tolist()})
    assert len(reconstruct_evidence(queries,initial,initial.x)) == 3
    broken = copy.deepcopy(queries)
    broken[0]['prequery_probability'] += 0.1
    with pytest.raises(AssertionError):
        reconstruct_evidence(broken,initial,initial.x)
    broken = copy.deepcopy(queries)
    broken[0]['source_weights'][0] += 0.1
    with pytest.raises(AssertionError):
        reconstruct_evidence(broken,initial,initial.x)
