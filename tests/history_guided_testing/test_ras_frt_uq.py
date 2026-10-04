"""Check categorical scenarios and binary-only feedback in the unified RAS run."""
import numpy as np

from methods.ras_frt_uq.transfer_uncertainty import OBSERVATION_NOISE, TransferUncertainty
from methods.ras_frt_uq.unified import family_similarities, select_from_responses
from methods.history_guided_testing.scenarios import parameter_cells


def test_ras_feedback_does_not_propagate_between_scenario_families():
    x = np.array([[.2, .3, .4, .5, 0], [.2, .3, .4, .5, 1], [.21, .3, .4, .5, 0]])
    similarity, kernel = family_similarities(x, np.full((3, 6), .3))
    assert similarity[0, 1] == kernel[0, 1] == 0
    assert similarity[0, 2] > .9 and kernel[0, 2] > .9
    model = TransferUncertainty(np.full(3, .3), kernel)
    model.observe(0, 1.)
    assert model.risk()[1] == .3
    assert model.risk()[2] > .3


def test_ras_reads_only_queried_collision_and_charges_every_query():
    x = np.random.default_rng(13).random((12, 5))
    x[:, 4] = np.arange(12) >= 6
    responses = np.random.default_rng(14).random((12, 6))

    class Observation:
        def __init__(self, index, count):
            self.collision = index % 3 == 0
            self.query_number = count
            self.scenario_id = str(index)

        @property
        def risk(self):
            raise AssertionError("RAS must not read continuous target risk")

    class Oracle:
        def __init__(self):
            self.selected = []

        def query(self, index):
            assert index not in self.selected
            self.selected.append(index)
            return Observation(index, len(self.selected))

    oracle = Oracle()
    result = select_from_responses(x, parameter_cells(x), responses, oracle, 11, budget=6)
    assert result["selected_indices"] == oracle.selected
    assert len(result["queries"]) == len(set(oracle.selected)) == 6
    assert all("risk" not in row for row in result["queries"])


def test_ras_binary_residual_updates_match_joint_gaussian_conditioning():
    x = np.random.default_rng(15).random((8, 5))
    x[:, 4] = np.arange(8) >= 4
    _, kernel = family_similarities(x, np.full((8, 6), .2))
    prior = np.linspace(.1, .7, len(x))
    model = TransferUncertainty(prior, kernel)
    selected, labels = [0, 6, 2], np.array([1., 0., 1.])
    for index, label in zip(selected, labels):
        model.observe(index, label)
    cross = kernel[:, selected]
    observed = kernel[np.ix_(selected, selected)] + OBSERVATION_NOISE * np.eye(len(selected))
    expected = prior + cross @ np.linalg.solve(observed, labels - prior[selected])
    covariance = kernel - cross @ np.linalg.solve(observed, cross.T)
    np.testing.assert_allclose(model.risk(), expected.clip(0, 1), atol=1e-12)
    np.testing.assert_allclose(model.covariance, covariance, atol=1e-12)
