from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from experiments.verify_selection import reconstruct
from testing_methods.common.gp_classifier import ClassificationKernel
from testing_methods.common.response import ResponseArchive, archive_fingerprints
from testing_methods.history_transfer.inference import (
    adaptation_states,
    historical_prior,
    query_probabilities,
)
from testing_methods.history_transfer.model import JointResponseKernel
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.training import (
    Config,
    cache_episode,
    contexts_for_episode,
    episode_probabilities,
    load_model,
    train,
)


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
    predictions = query_probabilities(*args, states, labels, (2, 4, 6), 4)
    for count in (2, 4, 6):
        _, expected, _, _ = reconstruct(
            scenes, base, source, actual["queries"][:count], "H1F1", list(range(6))
        )
        assert predictions[count].detach().numpy() == pytest.approx(expected[-4:], abs=1e-10)
    flipped = labels.copy()
    flipped[2:] = ~flipped[2:]
    changed = query_probabilities(*args, adaptation_states(base, x, flipped), flipped, (2, 4, 6), 4)
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
                rss_margin=np.where(labels, -0.9, -0.2 + offset * 0.1),
                rss_contract=measurement_contract(),
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
        probability = episode_probabilities(encoder, archives, episode, (2,), config)[2]
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
            rss_margin=np.where(labels, -0.9, -0.2 + offset * 0.1),
            rss_contract=measurement_contract(),
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
    result = train(history, development, tmp_path / "construction", config=config)
    assert result["target"]["status"] == result["encoder"]["status"] == "complete"
    model, metadata = load_model(tmp_path / "construction/historical_encoder.pt")
    assert next(model.parameters()).dtype == torch.float64
    assert set(metadata["specification"]["history_fingerprints"]) == {"a", "b", "c"}
    assert metadata["specification"]["target_feedback"] == "collision Boolean only"
    assert metadata["specification"]["continuous_response"] == measurement_contract()
    assert result["encoder"]["curve"][0]["validation"]["collision_nll"] >= 0
    with pytest.raises(ValueError, match="Held family"):
        train(
            history,
            development,
            tmp_path / "held",
            held_family="rule",
            config=config,
        )


def raw_archive():
    scenes = sample("cut_in", 8, "history")
    return ResponseArchive(
        "a",
        "rule",
        "a",
        "history",
        scenes,
        np.arange(8) % 3 == 0,
        np.column_stack((np.ones(8), np.ones(8))),
    )


def test_rss_context_is_explicit_preserves_raw_measurements_and_changes_fingerprint():
    raw = raw_archive()
    with pytest.raises(ValueError, match="requires derived"):
        raw.continuous_margin
    values = np.linspace(-0.9, 0.3, 8)
    rss = replace(raw, rss_margin=values, rss_contract=measurement_contract())
    np.testing.assert_array_equal(rss.safety, raw.safety)
    np.testing.assert_array_equal(rss.continuous_margin, values)
    assert archive_fingerprints([raw]) != archive_fingerprints([rss])
    changed = measurement_contract()
    changed["parameters"]["ego_response_s"] = 0.2
    with pytest.raises(ValueError, match="same global RSS"):
        replace(rss, rss_contract=changed)


def test_rss_posterior_uses_signed_margin_without_clearance_log_or_npc_labels():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    archive = replace(
        raw_archive(), rss_margin=np.linspace(-0.9, 0.3, 8), rss_contract=measurement_contract()
    )
    model = JointResponseKernel(archive.x.shape[1]).double()
    x = features(sample("cut_in", 4, "development"))
    actual = historical_prior(model, archive, x)
    # Original geometric clearance is retained, but cannot affect the RSS posterior.
    altered = replace(archive, safety=100 * archive.safety)
    for a, b in zip(actual, historical_prior(model, altered, x)):
        torch.testing.assert_close(a, b)
    (actual[0].square().sum() + actual[1].diagonal().sum()).backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_rss_loader_rejects_an_old_clearance_checkpoint(tmp_path):
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format": "domain-consistent-collision",
            "specification": {"historical_feedback": "collision and clearance"},
        },
        path,
    )
    with pytest.raises(ValueError, match="registered source"):
        load_model(path)


def test_rss_prior_builder_rejects_an_encoder_from_another_history_scope(monkeypatch):
    from experiments import train as workflow

    archive = replace(
        raw_archive(), rss_margin=np.linspace(-0.9, 0.3, 8), rss_contract=measurement_contract()
    )
    monkeypatch.setattr(workflow, "load_training_data", lambda *args: ([archive], []))
    monkeypatch.setattr(
        workflow,
        "load_model",
        lambda path: (None, {"specification": {"history_fingerprints": {}, "held_family": None}}),
    )
    with pytest.raises(ValueError, match="another history scope"):
        workflow.build_priors("cut_in", None)


def test_rss_source_export_round_trip_and_changed_encoder_rejection(tmp_path, monkeypatch):
    from experiments import train as workflow

    archive = replace(
        raw_archive(), rss_margin=np.linspace(-0.9, 0.3, 8), rss_contract=measurement_contract()
    )
    candidates = sample("cut_in", 4, "new")
    model = JointResponseKernel(archive.x.shape[1]).double()
    specification = {
        "history_fingerprints": archive_fingerprints([archive]),
        "held_family": None,
    }
    encoder = tmp_path / "historical_encoder.pt"
    encoder.write_bytes(b"fixed encoder")
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "model_directory", lambda *args: tmp_path)
    monkeypatch.setattr(workflow, "load_training_data", lambda *args: ([archive], []))
    monkeypatch.setattr(
        workflow, "load_model", lambda path: (model, {"specification": specification})
    )
    monkeypatch.setattr(workflow, "sample", lambda *args: candidates)
    expected = workflow.source_predictions(model, [archive], candidates)
    workflow.build_priors("cut_in", None)
    loaded = workflow.load_sources("cut_in")
    np.testing.assert_array_equal(loaded["means"], expected["means"])
    np.testing.assert_array_equal(
        loaded["covariances"], np.stack([np.diag(v) for v in expected["variances"]])
    )
    np.testing.assert_array_equal(loaded["weights"], expected["weights"])
    assert loaded["names"] == expected["names"]
    encoder.write_bytes(b"different encoder")
    with pytest.raises(ValueError, match="current global measure and encoder"):
        workflow.load_sources("cut_in")


def test_joint_kernel_is_positive_semidefinite_and_zero_correlation_disconnects_safety():
    torch.manual_seed(19)
    model = JointResponseKernel(3).double()
    x = np.random.default_rng(19).random((9, 3))
    covariance = model.joint_covariance(x)
    torch.testing.assert_close(covariance, covariance.T)
    assert torch.linalg.eigvalsh(covariance).min() > -1e-9
    with torch.no_grad():
        model.raw_correlation.zero_()
    covariance = model.joint_covariance(x)
    assert torch.count_nonzero(covariance[:9, 9:]) == 0


def test_signed_rss_context_influences_collision_predictions_only_through_cross_correlation():
    torch.manual_seed(23)
    raw = raw_archive()
    low = replace(raw, rss_margin=np.full(8, -0.9), rss_contract=measurement_contract())
    high = replace(raw, rss_margin=np.full(8, 0.5), rss_contract=measurement_contract())
    model = JointResponseKernel(raw.x.shape[1]).double()
    x = features(sample("cut_in", 4, "development"))
    low_mean, _ = historical_prior(model, low, x)
    high_mean, _ = historical_prior(model, high, x)
    assert (low_mean > high_mean).all()
    with torch.no_grad():
        model.raw_correlation.zero_()
    low_mean, low_covariance = historical_prior(model, low, x)
    high_mean, high_covariance = historical_prior(model, high, x)
    torch.testing.assert_close(low_mean, high_mean)
    torch.testing.assert_close(low_covariance, high_covariance)
