from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.rss import measurement_contract
from experiments.scenarios import sample
from testing_methods.common.response import ResponseArchive, archive_fingerprints
from testing_methods.history_transfer.model import JointResponseKernel
from testing_methods.history_transfer.training import (
    Config,
    cache_episode,
    contexts_for_episode,
    episode_probability,
    load_model,
    train,
)


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
    config = Config(initial_count=2, budget=4, historical_context=4)
    base = JointResponseKernel(history[0].x.shape[1]).double()
    episode = cache_episode(base, history[-1], config, config.seed)
    contexts = contexts_for_episode(history, episode, np.random.default_rng(config.seed), config)
    assert [a.source_id for a in contexts] == ["0", "1", "2"]
    relabeled = [replace(a, family="learning") for a in contexts]
    encoder = JointResponseKernel(history[0].x.shape[1]).double()
    results = []
    for archives in (contexts, relabeled):
        probability = episode_probability(encoder, archives, episode, 2)
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
    config = Config(steps=2, initial_count=2, budget=5, historical_context=4, diagnostic_interval=1)
    result = train(history, tmp_path / "construction", config=config)
    assert result["target"]["status"] == result["encoder"]["status"] == "complete"
    model, metadata = load_model(tmp_path / "construction/historical_encoder.pt")
    assert next(model.parameters()).dtype == torch.float64
    assert set(metadata["specification"]["history_fingerprints"]) == {"a", "b", "c"}
    assert metadata["specification"]["target_feedback"] == "paid collision and RSS"
    assert metadata["specification"]["continuous_response"] == measurement_contract()
    assert result["encoder"]["curve"][0]["training_nll"] >= 0
    assert metadata["step"] == config.steps
    assert "development_fingerprints" not in metadata["specification"]
    with pytest.raises(ValueError, match="Held family"):
        train(
            history,
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


def test_single_record_sources_remain_context_but_do_not_supervise_base_training(
    tmp_path, monkeypatch
):
    from testing_methods.history_transfer import training

    complete = replace(
        raw_archive(), rss_margin=np.linspace(-0.9, 0.3, 8), rss_contract=measurement_contract()
    )
    singleton = replace(
        complete,
        source_id="single",
        scenarios=complete.scenarios[:1],
        collision=complete.collision[:1],
        safety=complete.safety[:1],
        rss_margin=complete.rss_margin[:1],
    )
    scheduled = training.scheduled_item

    def select_supervised(items, new_ids, step):
        assert all(len(item.scenarios) >= 2 for item in items)
        return scheduled(items, new_ids, step)

    monkeypatch.setattr(training, "scheduled_item", select_supervised)
    config = Config(steps=1, initial_count=2, budget=4, historical_context=4, diagnostic_interval=1)
    report = train([singleton, complete], tmp_path, config=config)
    assert report["encoder"]["status"] == "complete"
    assert "single" in report["encoder"]["specification"]["history_fingerprints"]


def test_joint_kernel_is_positive_semidefinite_and_zero_correlation_disconnects_safety():
    torch.manual_seed(19)
    model = JointResponseKernel(3).double()
    x = np.random.default_rng(19).random((9, 3))
    covariance = model.kernel(x)
    torch.testing.assert_close(covariance, covariance.T)
    assert torch.linalg.eigvalsh(covariance).min() > -1e-9
    with torch.no_grad():
        model.raw_correlation.zero_()
    covariance = model.kernel(x)
    assert torch.count_nonzero(covariance[::2, 1::2]) == 0
