import numpy as np
import pytest
import torch

from methods.gcrt.residual_scale import update_reliability
from methods.gcrt.response_locality import influence_matrix, squared_distance


def fixture():
    rng = np.random.default_rng(1303)
    return (torch.tensor(rng.uniform(-1, 1, (9, 4))),
            torch.tensor(rng.uniform(.01, .99, (3, 9))))


@pytest.mark.parametrize('bandwidth', [None, .15, .3, .6])
def test_physical_endpoint_matches_registered_causal_update(bandwidth):
    x, response = fixture()
    kernel = influence_matrix(squared_distance(x, response, 0), bandwidth)
    prior = torch.tensor([.2, .3, .5], dtype=x.dtype)
    score = prior.log()[:, None].expand(-1, len(x)).clone()
    independent = score.clone()
    for index, label in ((1, True), (4, False), (8, True)):
        score = update_reliability(score, response, index, label, x, bandwidth)
        log_likelihood = (response[:, index] if label else 1-response[:, index]).log()
        independent += log_likelihood[:, None]*kernel[index]
        independent -= torch.logsumexp(independent, 0)
        torch.testing.assert_close(score, independent, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize('fraction', [0., .5, 1.])
def test_geometry_is_source_order_invariant_and_has_unit_self_influence(fraction):
    x, response = fixture()
    distance = squared_distance(x, response, fraction)
    torch.testing.assert_close(distance, squared_distance(x, response[[2, 0, 1]], fraction))
    np.testing.assert_allclose(distance.numpy(), distance.numpy().T, atol=1e-15)
    torch.testing.assert_close(distance.diag(), torch.zeros(len(x), dtype=x.dtype))
    kernel = influence_matrix(distance, .3)
    assert torch.all((kernel > 0) & (kernel <= 1))
    torch.testing.assert_close(kernel.diag(), torch.ones(len(x), dtype=x.dtype))


def test_equal_historical_responses_share_evidence_despite_parameter_distance():
    x = torch.tensor([[0., 0.], [1., 1.], [.01, .01]], dtype=torch.float64)
    response = torch.tensor([[.1, .1, .9], [.2, .2, .8]], dtype=torch.float64)
    physical = influence_matrix(squared_distance(x, response, 0), .15)
    historical = influence_matrix(squared_distance(x, response, 1), .15)
    assert physical[0, 1] < physical[0, 2]
    assert historical[0, 1] == 1 and historical[0, 1] > historical[0, 2]
    halfway = squared_distance(x, response, .5)
    torch.testing.assert_close(halfway, .5*(squared_distance(x, response, 0)
                                         + squared_distance(x, response, 1)))


@pytest.mark.parametrize('fraction', [-.1, 1.1])
def test_invalid_fraction_is_rejected(fraction):
    x, response = fixture()
    with pytest.raises(ValueError, match='fraction'):
        squared_distance(x, response, fraction)
