import copy
import math

import numpy as np
import pytest
import torch

from experiments.inference_reference import logistic_moments, posterior_numpy
from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.inference import (
    TARGET_PARAMETERS,
    JointPrior,
    adapted_model,
    fit_joint,
    predict_components,
    source_tensors,
)
from testing_methods.history_transfer.model import MARGIN_NOISE
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import Config, load_model, train


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(1301)


def sources(count):
    rng = np.random.default_rng(19)
    return {
        "names": ["a", "b"],
        "weights": np.array([0.5, 0.5]),
        "means": rng.normal(size=(2, count, 2)),
        "blocks": np.tile(np.array([[0.4, -0.1], [-0.1, 0.2]]), (2, count, 1, 1)),
        "low_rank": rng.normal(size=(2, count, 2, 3)) * 0.1,
    }


def test_covariance_matches_independent_three_latent_construction():
    x = np.array([[0, 0.1], [0.2, 0.3], [0.8, 0.7], [0.9, 0.95]])
    model = SharedPrivateKernel(2, deep=False).double()
    with torch.no_grad():
        model.log_lengthscale.fill_(math.log(0.3))
        model.log_collision_lengthscale.fill_(math.log(0.8))
        model.log_rss_lengthscale.fill_(math.log(0.1))
        model.log_scale.fill_(math.log(1.4))
        model.log_margin_scale.fill_(math.log(0.6))
        model.raw_correlation.fill_(math.atanh(0.4))
    loadings = np.array(
        [[1.4 * np.sqrt(0.5), 1.4 * np.sqrt(0.5), 0], [0.6 * 0.4, 0, 0.6 * np.sqrt(1 - 0.4**2)]]
    )
    distance = np.sqrt(np.sum((x[:, None] - x[None]) ** 2, axis=2))
    covariance = np.zeros((len(x), 2, len(x), 2))
    for column, length in enumerate((0.3, 0.8, 0.1)):
        scaled = np.sqrt(5) * distance / length
        kernel = (1 + scaled + scaled**2 / 3) * np.exp(-scaled)
        covariance += np.einsum("ij,a,b->iajb", kernel, loadings[:, column], loadings[:, column])
    actual = model.kernel(x).detach().numpy()
    np.testing.assert_allclose(actual, covariance.reshape(8, 8), atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() > -1e-10
    point_blocks = np.stack([actual[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] for i in range(len(x))])
    np.testing.assert_allclose(model.point_covariance(len(x)).detach(), point_blocks, atol=1e-12)


def test_source_contraction_preserves_marginals_psd_and_zero_cross_output():
    model = SharedPrivateKernel(2, deep=False).double()
    matrix = torch.randn(10, 13, dtype=torch.float64)
    covariance = matrix @ matrix.T
    for loading in (0.0, -0.4, 0.8):
        with torch.no_grad():
            model.raw_correlation.fill_(math.atanh(loading))
        actual = model.source_covariance(covariance).detach()
        torch.testing.assert_close(actual[::2, ::2], covariance[::2, ::2])
        torch.testing.assert_close(actual[1::2, 1::2], covariance[1::2, 1::2])
        torch.testing.assert_close(actual[::2, 1::2], abs(loading) * covariance[::2, 1::2])
        assert torch.linalg.eigvalsh(actual).min() > -1e-10


def test_zero_sharing_disconnects_rss_even_with_correlated_sources_and_raw_component():
    x = features(sample("car_following", 11, "history"))
    model = SharedPrivateKernel(x.shape[1]).double()
    physical = SharedPrivateKernel(x.shape[1], deep=False).double()
    tensors = source_tensors(sources(len(x)), model)
    reference = model.point_mean(1)[0].detach()
    with torch.no_grad():
        predictions = [
            predict_components(model, physical, x, tensors, reference, [0, 4], [False, True], rss)[
                0
            ]
            for rss in ([-0.8, -0.4], [0.6, 0.9])
        ]
        assert not torch.allclose(predictions[0][:, 1:], predictions[1][:, 1:])
        model.raw_correlation.zero_()
        physical.raw_correlation.zero_()
        predictions = [
            predict_components(model, physical, x, tensors, reference, [0, 4], [False, True], rss)[
                0
            ]
            for rss in ([-0.8, -0.4], [0.6, 0.9], None)
        ]
    for prediction in predictions[1:]:
        torch.testing.assert_close(predictions[0], prediction, atol=1e-10, rtol=1e-10)


def test_zero_loading_separates_rss_from_collision_hyperparameter_optimization():
    x = features(sample("car_following", 8, "history"))
    base = SharedPrivateKernel(x.shape[1]).double()
    with torch.no_grad():
        base.raw_correlation.zero_()
    models = [copy.deepcopy(base) for _ in range(2)]
    for model, rss in zip(models, ([-0.8] * 4, [0.6] * 4)):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in TARGET_PARAMETERS and name != "raw_correlation")
        fit_joint(model, x[:4], [False, True, False, True], rss, steps=10)
    for name in ("mean", "log_lengthscale", "log_scale"):
        torch.testing.assert_close(
            getattr(models[0], name), getattr(models[1], name), atol=1e-10, rtol=1e-10
        )
    assert not torch.allclose(models[0].margin_mean, models[1].margin_mean)


def test_block_prediction_matches_independent_dense_gaussian_then_bernoulli_reference():
    x = features(sample("car_following", 11, "history"))
    model = SharedPrivateKernel(x.shape[1]).double()
    physical = SharedPrivateKernel(x.shape[1], deep=False).double()
    reference = model.point_mean(1)[0]
    tensors = source_tensors(sources(len(x)), model)
    paid, labels, margins = [0, 4, 7], [False, True, False], [-0.2, -0.8, 0.4]
    actual, _ = predict_components(
        model, physical, x, tensors, reference, paid, labels, margins, batch_size=3
    )
    priors = [JointPrior(model, x, reference, s) for s in tensors]
    priors += [JointPrior(model, x, reference), JointPrior(physical, x, reference)]
    ids = list(range(len(x)))
    for predicted, prior in zip(actual, priors):
        mean = prior.mean(ids).detach().numpy().reshape(-1)
        covariance = prior.covariance(ids, ids).detach().numpy()
        rss_ids = 2 * np.array(paid) + 1
        cross = covariance[:, rss_ids]
        rr = covariance[np.ix_(rss_ids, rss_ids)] + MARGIN_NOISE**2 * np.eye(len(paid))
        mean = mean + cross @ np.linalg.solve(rr, np.asarray(margins) - mean[rss_ids])
        covariance = covariance - cross @ np.linalg.solve(rr, cross.T)
        pm, pc = posterior_numpy(mean, covariance, [2 * i for i in paid], labels)
        expected = logistic_moments(pm[::2], np.diag(pc)[::2])[0]
        np.testing.assert_allclose(predicted.detach(), expected, atol=2e-7)


def test_only_six_online_scalars_change_and_private_structure_stays_frozen():
    x = features(sample("car_following", 9, "history"))
    model = SharedPrivateKernel(x.shape[1]).double()
    result = adapted_model(model, x[:4], [False, True, False, False], [-0.2, -0.8, 0.3, 0.5])
    changed = [
        name
        for name, value in result.state_dict().items()
        if not torch.equal(value, model.state_dict()[name])
    ]
    assert changed and set(changed).issubset(TARGET_PARAMETERS)


def test_two_stage_training_and_strict_loading_use_the_registered_kernel(tmp_path):
    archives = []
    for i in range(2):
        scenes = sample("car_following", 6, "history", replicate=i)
        labels = np.arange(6) % 3 == i
        archives.append(
            ResponseArchive(
                f"source:{i}",
                "rule",
                f"source:{i}",
                "history",
                scenes,
                labels,
                np.column_stack((np.ones(6), np.ones(6))),
                rss_margin=np.where(labels, -0.8, 0.4),
                rss_contract=measurement_contract(),
            )
        )
    config = Config(steps=2, initial_count=2, budget=4, historical_context=3, diagnostic_interval=1)
    report = train(archives, tmp_path, config=config, kernel_type=SharedPrivateKernel)
    assert report["encoder"]["status"] == "complete"
    for name in ("target_kernel.pt", "historical_encoder.pt"):
        model, saved = load_model(tmp_path / name, kernel_type=SharedPrivateKernel)
        assert isinstance(model, SharedPrivateKernel)
        assert SharedPrivateKernel.source_path in saved["specification"]["training_contract"]
        with pytest.raises(ValueError, match="registered source"):
            load_model(tmp_path / name)


def test_selector_diagnostics_use_the_same_paid_prefix_and_do_not_affect_chunk_selection():
    scenes = sample("car_following", 12, "history")
    model = SharedPrivateKernel(features(scenes).shape[1]).double()

    class Oracle:
        def __init__(self, changed=None):
            self.changed = changed

        def query(self, index, **kwargs):
            margin = 0.9 if index == self.changed else -0.2
            return QueryFeedback(
                scenes[index].scenario_id,
                index % 3 == 0,
                SafetyMeasurements(5, 1, rss_margin=margin),
                0,
                measurement_contract(),
            )

    options = dict(
        candidates=scenes,
        pretrained=model,
        sources=sources(len(scenes)),
        initial_indices=[0, 1],
        budget=6,
        snapshot_counts=(2, 5, 6),
        diagnose_rss=True,
    )
    first = select(oracle=Oracle(), batch_size=3, **options)
    second = select(oracle=Oracle(), batch_size=8, **options)
    assert first["selected_indices"] == second["selected_indices"]
    for a, b in zip(first["prediction_snapshots"], second["prediction_snapshots"]):
        for key in ("probabilities", "collision_conditioned_probabilities"):
            np.testing.assert_allclose(a[key], b[key], atol=1e-10)
    changed = select(oracle=Oracle(first["selected_indices"][4]), batch_size=3, **options)
    assert changed["selected_indices"][:5] == first["selected_indices"][:5]
    assert changed["prediction_snapshots"][0] == first["prediction_snapshots"][0]
