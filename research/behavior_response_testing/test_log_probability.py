"""Ensure numerical certainty does not erase Bernoulli probability ordering."""
import numpy as np
import torch

from .log_probability import LogProbabilityTestingSession
from .session import rmse_discrepancy


def test_log_survival_preserves_ranking_beyond_double_precision_saturation():
    x = np.array([[0., 0., 0., 0., 0.], [0.1, 0.2, 0., 0., 0.]])
    logits = np.array([[60., 100.], [60., 100.]])
    session = LogProbabilityTestingSession(x,
                                           np.full((2, 2), 0.5),
                                           logits,
                                           rmse_discrepancy([0.01, 0.02]),
                                           budget=1,
                                           device="cpu")
    assert torch.all(session.collision == 1)
    assert session.next_index() == 1
    torch.testing.assert_close(session.log_safe_probabilities(),
                               torch.tensor([-60., -100.],
                                            dtype=torch.float64),
                               atol=1e-12,
                               rtol=0)


def test_log_survival_matches_probability_mixture_at_nonsaturated_values():
    x = np.array([[0., 0., 0., 0., 0.], [0.1, 0.2, 0., 0., 0.]])
    session = LogProbabilityTestingSession(x,
                                           np.full((2, 2), 0.5),
                                           [[-2., 1.], [3., 4.]],
                                           rmse_discrepancy([0.01, 0.02]),
                                           budget=1,
                                           device="cpu")
    session.log_weights = torch.tensor([0.2, 0.8], dtype=torch.float64).log()
    expected = session.log_weights.exp() @ session.collision
    torch.testing.assert_close(-session.log_safe_probabilities().expm1(),
                               expected,
                               atol=1e-12,
                               rtol=0)
    assert session.next_index() == int(expected.argmax())
