import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from benchmark.scenarios import sample
from methods.common.response import ResponseArchive, ResponseEncoder
from methods.gcrt.config import CONFIG
from methods.gcrt.consistency import (MODES, episode_loss, load_encoder,
                                       query_probabilities, train)
from methods.gcrt.discovery import discovery_posterior
from methods.gcrt.train import episode_loss as legacy_episode_loss


def archive(name, role='history'):
    scenes = sample('cut_in', 48, 'history' if role == 'history' else 'development')
    labels = np.arange(48) % 3 == (1 if name == 'b' else 0)
    return ResponseArchive(name, 'rule', name, role, scenes, labels,
                           np.column_stack((np.linspace(-1, 60, 48), np.ones(48))))


def test_shared_global_training_control_matches_existing_loss_and_gradients():
    history = [archive('a'), archive('b')]
    target = archive('dev', 'development')
    encoder = ResponseEncoder(history[0].x.shape[1])
    legacy = copy.deepcopy(encoder)
    indices = np.arange(16)
    loss, _ = episode_loss(encoder, history, target, indices, 4, 'shared_global')
    expected, _ = legacy_episode_loss(legacy, history, target, indices, 4)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-11)
    loss.backward()
    expected.backward()
    for p, reference in zip(encoder.parameters(), legacy.parameters()):
        np.testing.assert_allclose(p.grad, reference.grad, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize('mode', list(MODES))
def test_training_query_prediction_matches_deployed_posterior(mode):
    history = [archive('a'), archive('b')]
    target = archive('dev', 'development')
    encoder = ResponseEncoder(history[0].x.shape[1]).eval().requires_grad_(False)
    x, labels = target.x[:16], target.collision[:16]
    independent, bandwidth = MODES[mode]
    with torch.no_grad():
        expected = query_probabilities(encoder, x, history, labels, 4, mode).numpy()
    state = discovery_posterior(encoder, x, history, independent=independent, bandwidth=bandwidth)
    for i in range(4):
        state.observe(i, bool(labels[i]))
    np.testing.assert_allclose(expected, state.probabilities(np.arange(4,16)), atol=2e-12)


@pytest.mark.parametrize('mode', ['source_global', 'source_local'])
def test_source_kernels_train_with_finite_gradients_and_same_capacity(mode):
    history = [archive('a'), archive('b')]
    target = archive('dev', 'development')
    encoder = ResponseEncoder(history[0].x.shape[1])
    loss, _ = episode_loss(encoder, history, target, np.arange(16), 4, mode)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert encoder.log_response_scale.grad.abs() > 0
    assert sum(p.numel() for p in encoder.parameters()) == sum(
        p.numel() for p in ResponseEncoder(history[0].x.shape[1]).parameters())


def test_small_training_checkpoint_is_role_checked_frozen_and_reusable(tmp_path):
    history = [archive('a'), archive('b')]
    development = [archive('dev', 'development')]
    config = replace(CONFIG, steps=2, validation_interval=1, support_counts=(4,), query_count=6)
    path = tmp_path/'source_local.pt'
    report = train(history, development, path, 'source_local', config)
    assert report['status'] == 'complete'
    model, meta = load_encoder(path)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert meta['specification']['mode'] == 'source_local'
    assert train(history, development, path, 'source_local', config) == report
    with pytest.raises(ValueError, match='another contract'):
        train(history, development, path, 'source_global', config)
    with pytest.raises(ValueError, match='disjoint'):
        train(history, history, tmp_path/'bad.pt', 'source_global', config)


def test_128_support_points_have_finite_gradients_and_match_online_updates():
    scenes = sample('cut_in', 192, 'history')
    history = [ResponseArchive(str(i), 'rule', str(i), 'history', scenes,
                               np.arange(192) % (i+2) == 0, np.ones((192,2))) for i in range(2)]
    encoder = ResponseEncoder(history[0].x.shape[1])
    x, labels = history[0].x[:134], history[0].collision[:134]
    predictions = query_probabilities(encoder, x, history, labels, 128, 'source_local')
    assert predictions.shape == (6,)
    predictions.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    encoder.eval().requires_grad_(False)
    state = discovery_posterior(encoder, x, history, independent=True, bandwidth=0.30)
    for i in range(128):
        state.observe(i, bool(labels[i]))
    np.testing.assert_allclose(predictions.detach(), state.probabilities(np.arange(128,134)), atol=1e-10)
