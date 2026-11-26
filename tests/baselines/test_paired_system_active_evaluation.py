import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.baselines.paired_system_active_evaluation.method import (
    Config,
    TargetSurrogate,
    control_variate,
    run,
    support_information,
    support_model,
)


def test_local_control_variate_has_paper_shrinkage_and_keeps_negative_correlation():
    value, beta = control_variate([2, 4], [1, 2], [3, 4])
    assert beta == pytest.approx(0.3)
    assert value == pytest.approx(3.6)
    value, beta = control_variate([4, 2], [1, 2], [3, 4])
    assert beta == pytest.approx(-0.3)
    assert value == pytest.approx(2.4)
    assert control_variate([0.1, 0.3], [5, 5], [5, 5]) == pytest.approx((0.2, 0))


def test_support_information_matches_independent_joint_distribution_calculation():
    prior, likelihood = np.array([0.2, 0.3, 0.5]), np.array([0.1, 0.2, 0.9])
    joint = np.column_stack((prior * (1 - likelihood), prior * likelihood))
    independent = prior[:, None] * joint.sum(axis=0)
    reference = np.sum(joint * np.log(joint / independent))
    assert float(support_information(prior, likelihood)) == pytest.approx(reference)
    assert float(support_information([0.5, 0.5], [0, 1])) == pytest.approx(math.log(2))
    assert float(support_information([1, 0], [0.5, 0])) == pytest.approx(0)


def test_support_geometry_handles_repeated_public_coordinates_without_empty_clusters():
    x = np.array([[0, 0], [0, 0], [0, 0], [1, 1], [2, 2]], dtype=float)
    information, metadata = support_model(x, [0, 1, 2])
    assert metadata["clusters"] == 1
    assert np.isfinite(information).all()
    assert np.all((information >= 0) & (information <= math.log(2)))


def test_mc_dropout_statistics_do_not_modify_fitted_surrogate_parameters():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    x = np.random.default_rng(11).random((8, 3))
    model = TargetSurrogate(3, config=Config(training_epochs=3, mc_draws=4))
    model.fit(x, [False] * 8)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mean, sigma, gradient, draw = model.statistics(x)
    assert mean.shape == sigma.shape == gradient.shape == draw.shape == (8,)
    assert np.isfinite(np.r_[mean, sigma, gradient, draw]).all()
    assert np.all(sigma >= 0)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("label, repeated_geometry", [(False, False), (True, True)])
def test_paired_batches_respect_paid_prefix_partial_last_batch_and_duplicate_geometry(
    label, repeated_geometry
):
    scenes = sample("cut_in", 10, "development", replicate=17)
    if repeated_geometry:
        scenes = [replace(scenes[0], scenario_id=f"same:{i}") for i in range(10)]

    def execute(scene):
        return PaidExecution(
            QueryFeedback(scene.scenario_id, label, SafetyMeasurements(0, 0), 1), []
        )

    ledger = QueryLedger(scenes, 7, execute)
    result = run_isolated(
        run,
        {
            "candidates": scenes,
            "initial_indices": [7, 3, 2, 5],
            "proxy": np.linspace(0.1, 0.9, 10),
            "budget": 7,
            "config": Config(
                batch_size=2,
                shortlist_size=4,
                training_epochs=3,
                mc_draws=4,
                local_samples=8,
                paired_samples=3,
            ),
        },
        ledger,
    )
    assert result["selected_indices"][:4] == [7, 3, 2, 5]
    assert len(set(result["selected_indices"])) == result["actual_queries"] == 7
    assert result["target_fit_sample_counts"] == [4, 6]
    assert [row["batch"] for row in result["queries"]] == [0, 0, 0, 0, 1, 1, 2]
