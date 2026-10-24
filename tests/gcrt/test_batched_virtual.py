"""Full fixed-prior equivalence for batched virtual Boolean observations."""

import numpy as np
import pytest
import torch

from benchmark.discovery_converged_search_audit import probabilities_numpy
from methods.gcrt.batched_virtual import virtual_probabilities
from methods.gcrt.converged_response import component_probabilities


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('paid', [[], [0, 3, 6]])
def test_virtual_batches_match_independent_conditioning_and_preserve_inputs(device, paid):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    torch.set_num_threads(1)
    generator = np.random.default_rng(1327)
    a = generator.normal(size=(3, 8, 8))
    covariance = a @ a.transpose(0, 2, 1) / 8 + .05 * np.eye(8)
    means = generator.normal(size=(3, 8))
    m, c = torch.tensor(means, device=device), torch.tensor(covariance, device=device)
    saved = (m.clone(), c.clone())
    labels = [True, False, False][:len(paid)]
    candidates = [7, 1, 2]
    probability, info = virtual_probabilities(m, c, paid, labels, candidates, chunk_size=2)
    assert probability.shape == (3, 2, 3, 8)
    assert float(info['gradients'].max()) <= 1e-10
    for k, candidate in enumerate(candidates):
        for label in (False, True):
            expected = probabilities_numpy(means, covariance, [*paid, candidate], [*labels, label])
            np.testing.assert_allclose(probability[k, int(label)].cpu(), expected, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(m, saved[0], atol=0, rtol=0)
    torch.testing.assert_close(c, saved[1], atol=0, rtol=0)


def test_source_candidate_permutation_and_chunking_preserve_virtual_predictions():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(1331)
    means = torch.randn((3, 6), generator=generator, dtype=torch.float64)
    a = torch.randn((3, 6, 6), generator=generator, dtype=torch.float64)
    covariance = a @ a.transpose(-1, -2) + .1 * torch.eye(6)
    expected, _ = virtual_probabilities(means, covariance, [0], [True], [1, 2, 4], chunk_size=1)
    order = [2, 0, 1]
    actual, _ = virtual_probabilities(means[order], covariance[order], [0], [True], [4, 1, 2], chunk_size=3)
    torch.testing.assert_close(actual, expected[[2, 0, 1]][:, :, order], atol=2e-11, rtol=2e-11)


def test_virtual_batch_solves_large_variance_modes_for_both_labels():
    means = torch.tensor([[-5., 2., 8.]], dtype=torch.float64)
    covariance = torch.tensor([[[100., 9., .1], [9., 20., .3], [.1, .3, 1.]]], dtype=torch.float64)
    actual, info = virtual_probabilities(means, covariance, [1], [False], [0, 2])
    for k, candidate in enumerate([0, 2]):
        for label in (False, True):
            expected, _ = component_probabilities(means, covariance, [1, candidate], [False, label])
            torch.testing.assert_close(actual[k, int(label)], expected, atol=2e-11, rtol=2e-11)
    assert float(info['gradients'].max()) <= 1e-10
