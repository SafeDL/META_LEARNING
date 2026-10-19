from types import SimpleNamespace

import numpy as np
import pytest

from methods.baselines.hem.posterior import EvidenceMixture
from methods.baselines.hem.selection import select


def test_sequential_selector_uses_only_queried_risk_and_counts_initialization():
    rng = np.random.default_rng(19)
    x = np.column_stack([rng.random((256, 4)), np.repeat([0, 1], 128)])
    priors = np.column_stack([np.full(256, .5), x[:, 0], 1 - x[:, 0]])
    covariance = np.eye(256) * .01

    class Oracle:
        def __init__(self, collision):
            self.collision = collision
            self.queried = []

        def query(self, index):
            assert index not in self.queried
            assert len(self.queried) < 200
            self.queried.append(index)
            return SimpleNamespace(risk=float(x[index, 0]), collision=self.collision)

    runs = []
    for collision in (False, True):
        oracle = Oracle(collision)
        run = select(x, covariance, priors, [.2, .4, .4], priors[:, 0], oracle)
        assert run["selected_indices"] == oracle.queried
        assert len(run["evidence"]) == len(set(oracle.queried)) == 200
        runs.append(run)
    assert runs[0]["selected_indices"] == runs[1]["selected_indices"]


def test_invalid_feedback_is_atomic():
    model = EvidenceMixture(np.eye(3) * .2, np.full((3, 2), .5), [.5, .5])
    before = model.mean.copy(), model.log_probability.copy(), model.variance.copy()
    with pytest.raises(ValueError):
        model.observe(1, np.nan)
    for actual, original in zip((model.mean, model.log_probability, model.variance), before):
        np.testing.assert_array_equal(actual, original)
    assert model.indices == model.evidence == []

