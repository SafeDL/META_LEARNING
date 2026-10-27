from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.domain_main_validation import reconstruct
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.common.gp_classifier import ClassificationKernel
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.domain_build import historical_posterior
from testing_methods.history_transfer.domain_consistency import (
    adaptation_states,
    historical_prior,
    query_probabilities,
)
from testing_methods.history_transfer.domain_discovery import select
from testing_methods.history_transfer.domain_training import (
    Config,
    cache_episode,
    contexts_for_episode,
    episode_probabilities,
    load_model,
    train,
)
from testing_methods.history_transfer.joint_response import JointResponseKernel


def test_training_historical_posterior_matches_deployment_and_retains_finite_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    scenes = sample("cut_in", 8, "history")
    archive = ResponseArchive(
        "a",
        "rule",
        "a",
        "history",
        scenes,
        np.arange(8) % 3 == 0,
        np.column_stack((np.linspace(0, 20, 8), np.ones(8))),
    )
    model = JointResponseKernel(features(scenes).shape[1]).double()
    x = features(sample("cut_in", 4, "development"))
    expected_mean, expected_covariance, _ = historical_posterior(model, archive, x)
    mean, covariance = historical_prior(model, archive, x)
    assert mean.detach().numpy() == pytest.approx(expected_mean, abs=1e-10)
    assert covariance.detach().numpy() == pytest.approx(expected_covariance, abs=1e-10)
    (mean.square().sum() + covariance.diagonal().sum()).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max() > 1e-8 for g in gradients)


def test_episode_predictions_match_paid_online_updates_and_hide_future_labels():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    scenes = sample("cut_in", 10, "development")
    x = features(scenes)
    base = ClassificationKernel(x.shape[1], deep=True).double()
    source = {
        "means": np.stack((np.linspace(-2, 1, 10), np.linspace(0.5, -0.5, 10))),
        "covariances": np.stack((np.eye(10), 2 * np.eye(10))),
        "weights": np.array([0.4, 0.6]),
        "names": ["a", "b"],
    }
    labels = np.arange(6) % 3 == 0

    class Oracle:
        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id, bool(labels[index]), SafetyMeasurements(0, 0), 0
            )

    actual = select(
        scenes,
        Oracle(),
        pretrained=base,
        sources=source,
        mode="H1F1",
        initial_indices=list(range(6)),
        budget=6,
        device="cpu",
    )
    states = adaptation_states(base, x, labels)
    args = [
        torch.tensor(source[k], dtype=torch.float64) for k in ("means", "covariances", "weights")
    ]
    predictions = query_probabilities(*args, states, labels, (2, 4, 6), 4, covariance_mode="full")
    for count in (2, 4, 6):
        _, expected, _, _ = reconstruct(
            scenes, base, source, actual["queries"][:count], "H1F1", list(range(6))
        )
        assert predictions[count].detach().numpy() == pytest.approx(expected[-4:], abs=1e-10)
    flipped = labels.copy()
    flipped[2:] = ~flipped[2:]
    changed = query_probabilities(
        *args, adaptation_states(base, x, flipped), flipped, (2, 4, 6), 4, covariance_mode="full"
    )
    assert changed[2].detach().numpy() == pytest.approx(predictions[2].detach().numpy(), abs=1e-12)


def test_episode_predictions_and_gradients_do_not_depend_on_ads_family():
    torch.set_num_threads(1)
    torch.manual_seed(37)
    scenes = sample("cut_in", 12, "history")
    history = []
    for offset, family in enumerate(("rule", "rule", "search", "optimization")):
        labels = (np.arange(len(scenes)) + offset) % 3 == 0
        history.append(
            ResponseArchive(
                str(offset),
                family,
                str(offset),
                "history",
                scenes,
                labels,
                np.column_stack((np.where(labels, 0, 5 + offset), np.ones(len(scenes)))),
            )
        )
    config = Config(support_counts=(2,), query_count=2, historical_context=4)
    base = ClassificationKernel(history[0].x.shape[1], deep=True).double()
    episode = cache_episode(base, history[-1], config, config.seed)
    contexts = contexts_for_episode(history, episode, np.random.default_rng(config.seed), config)
    assert [a.source_id for a in contexts] == ["0", "1", "2"]
    relabeled = [replace(a, family="learning") for a in contexts]
    encoder = JointResponseKernel(history[0].x.shape[1]).double()
    results = []
    for archives in (contexts, relabeled):
        probability = episode_probabilities(encoder, archives, episode, (2,), config, "diagonal")[2]
        gradients = torch.autograd.grad(
            probability.sum(), tuple(encoder.parameters()), allow_unused=True
        )
        results.append((probability.detach(), gradients))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-12, rtol=1e-12)
    assert any(g is not None and g.abs().max() > 1e-8 for g in results[0][1])
    for original, relabeled in zip(results[0][1], results[1][1]):
        if original is None:
            assert relabeled is None
        else:
            assert torch.isfinite(original).all()
            torch.testing.assert_close(original, relabeled, atol=1e-12, rtol=1e-12)


def test_complete_consistent_construction_starts_from_history_and_masks_target_context(tmp_path):
    history_scenes = sample("cut_in", 24, "history")
    development_scenes = sample("cut_in", 24, "development")

    def archive(name, family, role, scenes, offset):
        labels = (np.arange(len(scenes)) + offset) % 4 == 0
        return ResponseArchive(
            name,
            family,
            name,
            role,
            scenes,
            labels,
            np.column_stack((np.where(labels, 0, 5 + offset), np.ones(len(scenes)))),
        )

    history = [
        archive("a", "rule", "history", history_scenes, 0),
        archive("b", "rule", "history", history_scenes, 1),
        archive("c", "search", "history", history_scenes, 2),
    ]
    development = [archive("d", "rule", "development", development_scenes, 3)]
    config = Config(
        steps=1, support_counts=(2, 4), query_count=2, historical_context=4, validation_interval=1
    )
    result = train(
        history, development, tmp_path / "construction", covariance_mode="full", config=config
    )
    assert result["target"]["status"] == result["encoder"]["status"] == "complete"
    model, metadata = load_model(tmp_path / "construction/historical_encoder.pt")
    assert next(model.parameters()).dtype == torch.float64
    assert set(metadata["specification"]["history_fingerprints"]) == {"a", "b", "c"}
    assert metadata["specification"]["target_feedback"] == "collision Boolean only"
    assert result["encoder"]["curve"][0]["validation"]["collision_nll"] >= 0
    with pytest.raises(ValueError, match="Held family"):
        train(
            history,
            development,
            tmp_path / "held",
            covariance_mode="full",
            held_family="rule",
            config=config,
        )
