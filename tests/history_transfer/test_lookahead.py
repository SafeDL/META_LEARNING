import copy

import numpy as np
import torch

from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.common.classification import torch_moments
from testing_methods.history_transfer.inference import (
    initial_weights,
    mixture_weights,
    predict_components,
    source_tensors,
)
from testing_methods.history_transfer.lookahead import JointLookahead, sample_responses, shortlist
from testing_methods.history_transfer.model import MARGIN_NOISE, JointResponseKernel
from testing_methods.history_transfer.selection import select


def setup():
    torch.set_num_threads(1)
    torch.manual_seed(1301)
    scenes = sample("car_following", 9, "history")
    x = features(scenes)
    base = JointResponseKernel(x.shape[1]).double().eval()
    physical = JointResponseKernel(x.shape[1], deep=False).double().eval()
    for model in (base, physical):
        model.requires_grad_(False)
    sources = {
        "names": ["a", "b"],
        "weights": np.array([0.4, 0.6]),
        "means": np.random.default_rng(11).normal(size=(2, len(x), 2)),
        "blocks": np.tile(np.array([[0.4, -0.1], [-0.1, 0.2]]), (2, len(x), 1, 1)),
    }
    tensors = source_tensors(sources, base)
    reference = base.point_mean(1)[0].detach()
    lookahead = JointLookahead(
        base,
        physical,
        x,
        tensors,
        reference,
        base.features(x).detach(),
        physical.features(x).detach(),
        [s.scenario_id for s in scenes],
        shortlist_count=4,
        fantasy_count=4,
        seed=11,
        batch_size=3,
    )
    return scenes, sources, lookahead


def test_joint_fantasies_match_mixture_moments_and_preserve_collision_rss_dependence():
    torch.set_num_threads(1)
    means = torch.tensor([[-0.7, 0.2], [1.2, -0.8]], dtype=torch.float64)
    blocks = torch.tensor([[[0.8, -0.2], [-0.2, 0.3]], [[0.4, 0.15], [0.15, 0.2]]])
    blocks = blocks.double()
    weights = torch.tensor([0.3, 0.7], dtype=torch.float64)
    uniforms = torch.quasirandom.SobolEngine(4, scramble=True, seed=11).draw_base2(
        16, dtype=torch.float64
    )
    labels, margins = sample_responses(means, blocks, weights, uniforms)
    y, r = np.asarray(labels), np.asarray(margins)
    p, derivative, _ = torch_moments(means[:, 0], blocks[:, 0, 0])
    expected_r = float(weights @ means[:, 1])
    expected_variance = (
        float(weights @ (blocks[:, 1, 1] + MARGIN_NOISE**2 + means[:, 1] ** 2)) - expected_r**2
    )
    expected_yr = float(weights @ (means[:, 1] * p + blocks[:, 0, 1] * derivative))
    np.testing.assert_allclose(y.mean(), float(weights @ p), atol=0.003)
    np.testing.assert_allclose(r.mean(), expected_r, atol=0.003)
    np.testing.assert_allclose(r.var(), expected_variance, atol=0.004)
    np.testing.assert_allclose((y * r).mean(), expected_yr, atol=0.004)
    assert (r < -1).any()  # These are model fantasies, never physical feedback.


def test_point_posterior_uses_the_existing_component_predictive_distribution():
    _, _, state = setup()
    paid, labels, margins = [0, 4], [False, True], [0.3, -0.8]
    indices = [1, 3, 7]
    means, blocks = state.point_posteriors(
        state.base, state.physical_base, paid, labels, margins, indices
    )
    actual, _ = predict_components(
        state.base,
        state.physical_base,
        state.x,
        state.sources,
        state.reference,
        paid,
        labels,
        margins,
        indices=indices,
        batch_size=2,
    )
    expected = torch_moments(means[:, :, 0], blocks[:, :, 0, 0].clamp_min(1e-10))[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.linalg.eigvalsh(blocks).min() >= -1e-12


def test_fantasy_update_matches_actual_paid_update_without_mutating_inputs():
    scenes, sources, state = setup()

    class Oracle:
        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id,
                index == 3,
                SafetyMeasurements(5, 1, rss_margin=-0.8 if index == 3 else 0.3),
                0,
                measurement_contract(),
            )

    real = select(
        scenes,
        Oracle(),
        pretrained=state.base,
        sources=sources,
        initial_indices=[0],
        query_order=[0, 3],
        budget=2,
        snapshot_counts=(1, 2),
    )
    first, final = real["prediction_snapshots"]
    query = real["queries"][1]
    scores = torch.tensor(
        real["queries"][0]["component_weights_after_query"][:-1], dtype=torch.float64
    )
    scores = (scores / scores.sum()).log()
    before = copy.deepcopy(state.base.state_dict())
    paid, labels, margins = [0], [False], [0.3]
    saved = copy.deepcopy((paid, labels, margins, scores))
    actual = state.fantasy_prediction(
        scores,
        paid,
        labels,
        margins,
        3,
        True,
        -0.8,
        torch.tensor(query["prequery_component_probabilities"], dtype=torch.float64),
    )
    np.testing.assert_allclose(actual, final["probabilities"], atol=2e-9)
    assert first["paid_queries"] == 1
    assert (paid, labels, margins) == saved[:3]
    torch.testing.assert_close(scores, saved[3], atol=0, rtol=0)
    for name, value in state.base.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_last_query_is_greedy_and_shortlist_keeps_probability_and_disagreement_choices():
    scenes, sources, state = setup()
    components = torch.full((4, len(scenes)), 0.2, dtype=torch.float64)
    components[:, 4] = 0.8
    components[0, 5], components[1:, 5] = 0.01, 0.7
    scores = initial_weights(sources, state.reference).log()
    weights = mixture_weights(scores)
    choices = shortlist(set(range(len(scenes))), components, weights, state.candidate_ids, 4)
    assert 4 in choices and 5 in choices and len(set(choices)) == 4
    index, diagnostic = state.choose(
        state.base, state.physical_base, scores, [], [], [], components, set(range(len(scenes))), 1
    )
    assert index == 4 and diagnostic["fantasy_branches"] == 0


def test_full_lookahead_is_reproducible_and_uses_only_virtual_branches(monkeypatch):
    scenes, sources, state = setup()
    paid, labels, margins = [0, 4], [False, True], [0.3, -0.8]
    scores = initial_weights(sources, state.reference).log()
    components, _ = predict_components(
        state.base,
        state.physical_base,
        state.x,
        state.sources,
        state.reference,
        paid,
        labels,
        margins,
    )
    remaining = set(range(len(scenes))) - set(paid)
    saved = copy.deepcopy((paid, labels, margins, remaining))
    kernel = JointResponseKernel.kernel_from_features

    def no_candidate_matrix(self, z, other=None):
        assert not (len(z) == len(scenes) and (other is None or len(other) == len(scenes)))
        return kernel(self, z, other)

    monkeypatch.setattr(JointResponseKernel, "kernel_from_features", no_candidate_matrix)
    options = (
        state.base,
        state.physical_base,
        scores,
        paid,
        labels,
        margins,
        components,
        remaining,
        3,
    )
    first = state.choose(*options)
    assert first == state.choose(*options)
    index, diagnostic = first
    assert index in remaining and diagnostic["fantasy_branches"] == 16
    assert all(np.isfinite(row["utility"]) for row in diagnostic["shortlist"])
    assert (paid, labels, margins, remaining) == saved
    state.batch_size = 128
    chunked_index, chunked_diagnostic = state.choose(*options)
    assert index == chunked_index
    np.testing.assert_allclose(
        [r["utility"] for r in diagnostic["shortlist"]],
        [r["utility"] for r in chunked_diagnostic["shortlist"]],
        atol=1e-12,
        rtol=0,
    )


class ToyOracle:
    def __init__(self, scenes, *, change_after=None):
        self.scenes, self.change_after = scenes, change_after
        self.calls = []

    def query(self, index, *, purpose):
        self.calls.append(index)
        collision = index % 3 == 0
        if self.change_after is not None and len(self.calls) > self.change_after:
            collision = not collision
        return QueryFeedback(
            self.scenes[index].scenario_id,
            collision,
            SafetyMeasurements(5, 1, rss_margin=-0.8 if collision else 0.3),
            0,
            measurement_contract(),
        )


def selector_options(state, sources):
    return {
        "pretrained": state.base,
        "sources": sources,
        "initial_indices": [0],
        "budget": 4,
        "snapshot_counts": (1, 2, 4),
        "lookahead": {"shortlist_count": 2, "fantasy_count": 2, "seed": 11},
    }


def test_selector_fantasies_do_not_pay_queries_or_change_the_real_update():
    scenes, sources, state = setup()
    oracle = ToyOracle(scenes)
    options = selector_options(state, sources)
    result = select(scenes, oracle, **options)
    assert oracle.calls == result["selected_indices"]
    assert len(set(oracle.calls)) == result["actual_queries"] == 4
    assert result["queries"][-1]["lookahead"]["fantasy_branches"] == 0
    assert sum(q["lookahead"]["fantasy_branches"] for q in result["queries"][1:]) == 8
    replay = select(
        scenes,
        ToyOracle(scenes),
        **{**options, "lookahead": None, "query_order": result["selected_indices"]},
    )
    for left, right in zip(result["queries"], replay["queries"], strict=True):
        np.testing.assert_allclose(
            left["prequery_component_probabilities"],
            right["prequery_component_probabilities"],
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            left["component_weights_after_query"],
            right["component_weights_after_query"],
            atol=1e-12,
            rtol=0,
        )


def test_selector_batch_size_and_later_feedback_do_not_change_current_choices():
    scenes, sources, state = setup()
    options = selector_options(state, sources)
    small = select(scenes, ToyOracle(scenes), **options, batch_size=2)
    large = select(scenes, ToyOracle(scenes), **options, batch_size=128)
    later = select(scenes, ToyOracle(scenes, change_after=3), **options, batch_size=2)
    assert small["selected_indices"] == large["selected_indices"] == later["selected_indices"]
    for result in (large, later):
        np.testing.assert_allclose(
            [q["prequery_probability"] for q in small["queries"]],
            [q["prequery_probability"] for q in result["queries"]],
            atol=1e-12,
            rtol=0,
        )
    assert (
        small["queries"][-1]["feedback"]["collision"]
        != later["queries"][-1]["feedback"]["collision"]
    )


def test_common_feedback_lookahead_records_choices_without_changing_prefix_predictions():
    scenes, sources, state = setup()
    options = {**selector_options(state, sources), "query_order": [0, 1, 2, 3]}
    with_diagnostics = select(scenes, ToyOracle(scenes), **options)
    original = select(scenes, ToyOracle(scenes), **{**options, "lookahead": None})
    assert with_diagnostics["selected_indices"] == original["selected_indices"]
    for left, right in zip(
        with_diagnostics["prediction_snapshots"], original["prediction_snapshots"], strict=True
    ):
        np.testing.assert_allclose(
            left["probabilities"], right["probabilities"], atol=1e-12, rtol=0
        )
        diagnostic = left["lookahead"]
        if left["paid_queries"] == 4:
            assert diagnostic is None
        else:
            assert diagnostic["remaining_budget"] == 4 - left["paid_queries"]
            assert diagnostic["selected_index"] not in left["observed_indices"]
