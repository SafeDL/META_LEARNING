import numpy as np
import pytest
import torch

from methods.gcrt.feature_reliability import log_weights, squared_distance


def test_current_feature_geometry_moves_evidence_without_rescoring_labels():
    prior = torch.tensor([.5, .5], dtype=torch.float64)
    likelihood = torch.tensor([[np.log(.9)], [np.log(.1)]], dtype=torch.float64)
    before = torch.tensor([[0.], [.05], [2.]], dtype=torch.float64)
    after = before[[0, 2, 1]]
    a = log_weights(prior, likelihood, [0], squared_distance(before), .3).exp()
    b = log_weights(prior, likelihood, [0], squared_distance(after), .3).exp()
    assert a[0, 1] > .89 and a[0, 2] == pytest.approx(.5)
    assert b[0, 2] > .89 and b[0, 1] == pytest.approx(.5)
    torch.testing.assert_close(likelihood, torch.tensor([[np.log(.9)], [np.log(.1)]]), rtol=0, atol=0)


def test_global_equivalence_empty_history_and_source_permutation():
    prior = torch.tensor([.25, .75], dtype=torch.float64)
    ll = torch.tensor([[-.2, -.8], [-.6, -.3]], dtype=torch.float64)
    encoded = torch.tensor([[0., 1.], [2., 1.], [1., 2.]], dtype=torch.float64)
    distance = squared_distance(encoded, 2.)
    empty = log_weights(prior, ll[:, :0], [], distance, .3).exp()
    torch.testing.assert_close(empty, prior[:, None].expand(-1, 3))
    global_weights = log_weights(prior, ll, [0, 1], distance, None).exp()
    expected = (prior.log()+ll.sum(1)).softmax(0)[:, None].expand(-1, 3)
    torch.testing.assert_close(global_weights, expected)
    local = log_weights(prior, ll, [0, 1], distance, .3).exp()
    reordered = log_weights(prior.flip(0), ll.flip(0), [0, 1], distance, .3).exp().flip(0)
    torch.testing.assert_close(local, reordered)
    torch.testing.assert_close(local.sum(0), torch.ones(3, dtype=torch.float64))


def test_rigid_feature_change_and_common_scale_leave_distance_unchanged():
    encoded = torch.tensor([[1., 2.], [3., 5.], [7., 11.]], dtype=torch.float64)
    distance = squared_distance(encoded, 2.)
    changed = torch.stack((-encoded[:, 1], encoded[:, 0]), 1)*3+17
    torch.testing.assert_close(squared_distance(changed, 6.), distance)


def test_noncausal_shape_or_duplicate_paid_observation_is_rejected():
    prior = torch.tensor([.5, .5], dtype=torch.float64)
    distance = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match='stored prequery likelihood'):
        log_weights(prior, torch.zeros(2, 3), [0, 1], distance, .3)
    with pytest.raises(ValueError, match='stored prequery likelihood'):
        log_weights(prior, torch.zeros(2, 2), [0, 0], distance, .3)
