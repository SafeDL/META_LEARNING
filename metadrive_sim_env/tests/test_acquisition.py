from __future__ import annotations

import numpy as np

from metadrive_sim_env.mining.acquisition import (
    diagnostic_scores,
    mining_scores,
    novelty_weight,
    score_level_set_weight,
)
from metadrive_sim_env.mining.posterior import LatentVulnerabilityPosterior


def test_diagnostic_acquisition_rewards_information_boundary_and_evaluability() -> None:
    posterior = LatentVulnerabilityPosterior.standard_normal(1)
    basis = np.asarray(((1.0, ), (1.0, ), (0.1, )))
    mean = np.asarray((0.5, 0.5, 0.5))
    noise = np.asarray((0.1, 0.1, 0.1))
    values = diagnostic_scores(
        posterior,
        basis,
        mean,
        noise,
        np.asarray((1.0, 0.1, 1.0)),
        level_set_threshold=0.75,
    )
    assert values[0] > values[1]
    assert values[0] > values[2]


def test_mining_acquisition_novelty_and_empty_archive_contract() -> None:
    features = np.asarray(((0.0, ) * 7, (1.0, ) * 7))
    novelty = novelty_weight(features, np.empty((0, 7)))
    np.testing.assert_allclose(novelty, (1.0, 1.0))
    nonempty = novelty_weight(features, features[:1])
    assert nonempty[1] > nonempty[0]
    scores = mining_scores(np.asarray((0.5, 0.5)), np.asarray((0.0, 0.0)), np.asarray((1.0, 1.0)),
                           nonempty)
    assert scores[1] > scores[0]


def test_level_set_boundary_uses_the_explicit_vulnerability_threshold() -> None:
    weight = score_level_set_weight(np.asarray((0.5, 0.75, 0.9)), np.asarray((0.01, 0.01, 0.01)),
                                    0.75)
    assert weight[1] > weight[0]
    assert weight[1] > weight[2]
