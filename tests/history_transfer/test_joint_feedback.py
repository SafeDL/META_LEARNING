import copy

import numpy as np
import pytest
import torch

from experiments.inference_reference import logistic_moments, posterior_numpy
from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.inference import (
    JointPrior,
    adapted_model,
    historical_prior,
    predict_components,
    sequential_probability,
    source_tensors,
)
from testing_methods.history_transfer.model import MARGIN_NOISE, JointResponseKernel
from testing_methods.history_transfer.selection import aggregate_history, select
from testing_methods.history_transfer.training import Config, episode_layout, train


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(1301)


def archive(name="source", n=12, role="history", offset=0):
    scenes = sample(
        "car_following", n, "development" if role == "development" else "history", replicate=offset
    )
    y = (np.arange(n) + offset) % 3 == 0
    return ResponseArchive(
        name,
        "rule",
        name,
        role,
        scenes,
        y,
        np.column_stack((np.where(y, 0, 10), np.ones(n))),
        rss_margin=np.where(y, -0.8, 0.2),
        rss_contract=measurement_contract(),
    )


def sources(n):
    return {
        "names": ["a", "b"],
        "weights": np.array([0.4, 0.6]),
        "means": np.random.default_rng(1).normal(size=(2, n, 2)),
        "blocks": np.tile(np.array([[0.4, -0.1], [-0.1, 0.2]]), (2, n, 1, 1)),
    }


def adaptation_states(pretrained, x, labels, margins):
    pretrained = adapted_model(pretrained, x[:0], [], None if margins is None else [])
    physical = JointResponseKernel(x.shape[1], deep=False).double().to(pretrained.mean.device)
    return [
        (
            adapted_model(pretrained, x[:n], labels[:n], None if margins is None else margins[:n]),
            adapted_model(
                physical,
                x[:n],
                labels[:n],
                None if margins is None else margins[:n],
                learning_rate=0.01,
            ),
        )
        for n in range(len(labels) + 1)
    ]


def dense_reference(prior, paid, labels, margins):
    ids = list(range(len(prior.x)))
    mean = prior.mean(ids).detach().numpy().reshape(-1)
    covariance = prior.covariance(ids, ids).detach().numpy()
    r = 2 * np.array(paid) + 1
    cross = covariance[:, r]
    rr = covariance[np.ix_(r, r)] + MARGIN_NOISE**2 * np.eye(len(r))
    mean = mean + cross @ np.linalg.solve(rr, np.asarray(margins) - mean[r])
    covariance = covariance - cross @ np.linalg.solve(rr, cross.T)
    pm, pc = posterior_numpy(mean, covariance, [2 * i for i in paid], labels)
    return logistic_moments(pm[::2], np.diag(pc)[::2])[0]


def test_joint_block_inference_matches_independent_dense_numpy_reference():
    scenes = sample("car_following", 11, "history")
    x = features(scenes)
    model = JointResponseKernel(x.shape[1]).double()
    physical = JointResponseKernel(x.shape[1], deep=False).double()
    reference = model.point_mean(1)[0]
    raw = aggregate_history(sources(len(x)))
    tensors = source_tensors(raw, model)
    paid, y, r = [0, 4, 7], [False, True, False], [-0.2, -0.8, 0.4]
    actual, _ = predict_components(model, physical, x, tensors, reference, paid, y, r, batch_size=3)
    priors = [
        JointPrior(model, x, reference, tensors[0]),
        JointPrior(model, x, reference),
        JointPrior(physical, x, reference),
    ]
    for prediction, prior in zip(actual, priors):
        np.testing.assert_allclose(
            prediction.detach(), dense_reference(prior, paid, y, r), atol=2e-7
        )


def test_paid_rss_changes_future_collision_predictions_and_zero_correlation_disconnects():
    x = features(sample("car_following", 8, "history"))
    model = JointResponseKernel(x.shape[1]).double()
    for correlated in (True, False):
        if not correlated:
            with torch.no_grad():
                model.raw_correlation.zero_()
        probabilities = [
            predict_components(model, model, x, [], model.point_mean(1)[0], [0], [False], [r])[0]
            for r in (-0.9, 0.5)
        ]
        if correlated:
            assert not torch.allclose(probabilities[0][:, 1:], probabilities[1][:, 1:])
        else:
            torch.testing.assert_close(*probabilities, atol=1e-12, rtol=1e-12)


def test_prescribed_feedback_matches_online_prefix_and_never_reads_later_feedback():
    scenes = sample("car_following", 12, "history")
    model = JointResponseKernel(features(scenes).shape[1]).double()

    class Oracle:
        def __init__(self, change_after=None):
            self.calls = 0
            self.change_after = change_after

        def query(self, index, *, purpose):
            self.calls += 1
            changed = self.change_after is not None and self.calls > self.change_after
            return QueryFeedback(
                scenes[index].scenario_id,
                changed or index % 3 == 0,
                SafetyMeasurements(5, 1, rss_margin=0.8 if changed else -0.2),
                0,
                measurement_contract(),
            )

    options = dict(
        pretrained=model,
        sources=sources(12),
        initial_indices=[0, 1],
        budget=5,
        snapshot_counts=(2, 4, 5),
    )
    online = select(scenes, Oracle(), **options)
    order = online["selected_indices"]
    replay = select(scenes, Oracle(), **options, query_order=order)
    later = select(scenes, Oracle(4), **options, query_order=order)
    assert replay["selected_indices"] == order
    for left, right in zip(online["queries"], replay["queries"]):
        np.testing.assert_allclose(
            left["prequery_component_probabilities"],
            right["prequery_component_probabilities"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            left["prequery_component_weights"], right["prequery_component_weights"], atol=1e-12
        )
    for left, right in zip(replay["prediction_snapshots"][:2], later["prediction_snapshots"][:2]):
        assert left == right
    with pytest.raises(ValueError, match="preserve initialization"):
        select(scenes, Oracle(), **options, query_order=order[::-1])


def test_only_six_scalars_adapt_and_shared_embeddings_and_base_stay_frozen():
    a = archive()
    model = JointResponseKernel(a.x.shape[1]).double()
    original = copy.deepcopy(model.state_dict())
    result = adapted_model(model, a.x[:3], a.collision[:3], a.continuous_margin[:3])
    changed = []
    for name, value in result.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], original[name], rtol=0, atol=0)
        if not torch.equal(value, original[name]):
            changed.append(name)
    assert changed and len(changed) <= 6
    assert all(not name.startswith("embedding") for name in changed)
    assert not any(p.requires_grad for p in result.parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    shared = adapted_model(model, a.x[:3], a.collision[:3], a.continuous_margin[:3])
    assert shared.embedding is model.embedding


def test_historical_export_preserves_joint_blocks_without_candidate_matrix(monkeypatch):
    a = archive()
    x = features(sample("car_following", 257, "development"))
    model = JointResponseKernel(a.x.shape[1]).double()
    kernel = model.kernel_from_features
    calls = []

    def checked(left, right=None):
        calls.append((len(left), len(left) if right is None else len(right)))
        assert right is not None or len(left) == len(a.x)
        return kernel(left, right)

    monkeypatch.setattr(model, "kernel_from_features", checked)
    mean, blocks = historical_prior(model, a, x, batch_size=32)
    assert mean.shape == (257, 2) and blocks.shape == (257, 2, 2)
    assert torch.linalg.eigvalsh(blocks).min() >= -1e-8
    assert blocks[:, 0, 1].abs().max() > 1e-8
    assert max(n for n, m in calls if n != len(a.x)) <= 32


def test_large_pool_uses_only_paid_matrices_and_reuses_kernel_blocks_across_sources(monkeypatch):
    x = features(sample("car_following", 4096, "development"))
    model = JointResponseKernel(x.shape[1]).double()
    kernel = model.kernel_from_features
    calls = []

    def checked(left, right=None):
        width = len(left) if right is None else len(right)
        calls.append((len(left), width))
        assert width <= 4
        assert len(left) <= 128
        return kernel(left, right)

    monkeypatch.setattr(model, "kernel_from_features", checked)
    with torch.no_grad():
        result, _ = predict_components(
            model,
            model,
            x,
            source_tensors(sources(len(x)), model),
            model.point_mean(1)[0],
            [0, 1, 2, 3],
            [False, True, False, False],
            [-0.2, -0.8, 0.2, 0.3],
        )
    assert result.shape == (4, 4096) and torch.isfinite(result).all()
    assert len(calls) == 1 + 4096 // 128


def test_batch_size_does_not_change_predictions_or_selection_and_future_feedback_is_hidden():
    scenes = sample("car_following", 12, "development")
    model = JointResponseKernel(features(scenes).shape[1]).double()

    class Oracle:
        def __init__(self, later=0):
            self.calls = 0
            self.later = later

        def query(self, index, *, purpose):
            self.calls += 1
            margin = -0.2 if self.calls <= 2 else self.later
            return QueryFeedback(
                scenes[index].scenario_id,
                index % 3 == 0,
                SafetyMeasurements(5, 1, rss_margin=margin),
                0,
                measurement_contract(),
            )

    outputs = [
        select(
            scenes,
            Oracle(),
            pretrained=model,
            sources=sources(12),
            initial_indices=[0, 1],
            budget=4,
            batch_size=b,
        )
        for b in (1, 7, 128)
    ]
    assert all(o["selected_indices"] == outputs[0]["selected_indices"] for o in outputs)
    for output in outputs[1:]:
        np.testing.assert_allclose(
            [q["prequery_probability"] for q in output["queries"]],
            [q["prequery_probability"] for q in outputs[0]["queries"]],
            atol=1e-12,
        )
    altered = select(
        scenes,
        Oracle(later=0.8),
        pretrained=model,
        sources=sources(12),
        initial_indices=[0, 1],
        budget=4,
    )
    assert altered["selected_indices"][:3] == outputs[0]["selected_indices"][:3]
    for first, second in zip(outputs[0]["queries"][:3], altered["queries"][:3]):
        assert first["prequery_probability"] == pytest.approx(
            second["prequery_probability"], abs=1e-12
        )


def test_episode_prefix_predictions_match_online_and_do_not_use_future_rss():
    scenes = sample("car_following", 10, "development")
    x = features(scenes)
    model = JointResponseKernel(x.shape[1]).double()
    raw = sources(10)
    y = np.arange(6) % 3 == 0
    margins = np.linspace(-0.8, 0.2, 6)

    class Oracle:
        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id,
                bool(y[index]),
                SafetyMeasurements(5, 1, rss_margin=float(margins[index])),
                0,
                measurement_contract(),
            )

    actual = select(
        scenes, Oracle(), pretrained=model, sources=raw, initial_indices=list(range(6)), budget=6
    )
    states = adaptation_states(model, x, y, margins)
    assert states[0][0].embedding is states[-1][0].embedding
    assert states[0][0].embedding is not model.embedding
    result = {}
    for count in (2, 4):
        result[count] = sequential_probability(
            raw, states, model.point_mean(1)[0].detach(), x, y, margins, count
        )
        assert float(result[count]) == pytest.approx(
            actual["queries"][count]["prequery_probability"], abs=1e-8
        )
    changed = margins.copy()
    changed[2:] = 0.9
    flipped = y.copy()
    flipped[2:] = ~flipped[2:]
    altered = sequential_probability(
        raw,
        adaptation_states(model, x, flipped, changed),
        model.point_mean(1)[0].detach(),
        x,
        flipped,
        changed,
        2,
    )
    torch.testing.assert_close(result[2], altered, atol=1e-12, rtol=1e-12)


def test_fixed_source_weights_keep_paid_joint_adaptation_enabled():
    scenes = sample("car_following", 7, "development")
    model = JointResponseKernel(3).double()

    class Oracle:
        def __init__(self, margin):
            self.margin = margin

        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id,
                False,
                SafetyMeasurements(5, 1, rss_margin=self.margin),
                0,
                measurement_contract(),
            )

    results = [
        select(
            scenes,
            Oracle(margin),
            pretrained=model,
            sources=sources(7),
            initial_indices=[0, 1],
            budget=4,
            update_source_weights=False,
            snapshot_counts=(0, 4),
        )
        for margin in (-0.8, 0.6)
    ]
    for result in results:
        assert result["target_feedback_updates"]
        assert not result["source_weight_updates"]
        for i, query in enumerate(result["queries"]):
            assert query["conditioning_queries"] == i
            assert query["prequery_log_likelihood"] is None
            np.testing.assert_allclose(
                query["component_weights_after_query"],
                result["queries"][0]["prequery_component_weights"],
                atol=1e-15,
            )
        assert (
            result["prediction_snapshots"][0]["target_parameters"]
            != (result["prediction_snapshots"][-1]["target_parameters"])
        )
    assert results[0]["queries"][1]["prequery_probability"] != pytest.approx(
        results[1]["queries"][1]["prequery_probability"],
        abs=1e-8,
    )


@pytest.mark.parametrize("n", [2, 8, 32, 128, 200])
def test_partial_archives_use_every_time_after_initialization(n):
    a = archive(n=n)
    initial, length = episode_layout(a, Config())
    assert length == n
    assert initial == min(20, n - 1)
    assert list(range(initial, length))[-1] == n - 1


def test_reduced_joint_training_with_non_aligned_partial_archives(tmp_path):
    history = [archive("a", 8), archive("b", 10, offset=1)]
    config = Config(steps=1, initial_count=2, budget=4, historical_context=4, diagnostic_interval=1)
    report = train(history, tmp_path, config=config)
    assert report["encoder"]["status"] == "complete"
    assert report["encoder"]["specification"]["target_feedback"] == "paid collision and RSS"
