from __future__ import annotations

import numpy as np

from mvr.diva.acquisition import diagnostic_scores, mining_scores, novelty_weight
from mvr.diva.posterior import LatentVulnerabilityPosterior


def test_diagnostic_acquisition_rewards_information_boundary_and_evaluability() -> None:
    posterior = LatentVulnerabilityPosterior.standard_normal(1)
    basis = np.asarray(((1.0,), (1.0,), (0.1,)))
    mean = np.asarray((0.5, 0.5, 0.5))
    noise = np.asarray((0.1, 0.1, 0.1))
    values = diagnostic_scores(posterior, basis, mean, noise, np.asarray((1.0, 0.1, 1.0)))
    assert values[0] > values[1]
    assert values[0] > values[2]


def test_mining_acquisition_novelty_and_empty_archive_contract() -> None:
    features = np.asarray(((0.0,) * 7, (1.0,) * 7))
    novelty = novelty_weight(features, np.empty((0, 7)))
    np.testing.assert_allclose(novelty, (1.0, 1.0))
    nonempty = novelty_weight(features, features[:1])
    assert nonempty[1] > nonempty[0]
    scores = mining_scores(np.asarray((0.5, 0.5)), np.asarray((0.0, 0.0)), np.asarray((1.0, 1.0)), nonempty)
    assert scores[1] > scores[0]
