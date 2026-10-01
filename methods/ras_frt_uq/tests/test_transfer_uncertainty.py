"""Check that transfer queries probe historical safety and use feedback."""

import numpy as np

from methods.ras_frt_uq.transfer_uncertainty import TransferUncertainty


def test_initial_information_can_override_high_historical_risk():
    model = TransferUncertainty(np.array([0.99, 0.01]), np.eye(2))
    assert model.choose() == 1


def test_target_feedback_updates_nearby_candidate_and_avoids_requery():
    prior = np.array([0.05, 0.05, 0.05])
    kernel = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    model = TransferUncertainty(prior, kernel)
    model.observe(0, 1)
    assert model.risk()[1] > prior[1]
    assert model.risk()[2] == prior[2]
    assert not np.isfinite(model.acquisition()[0])
