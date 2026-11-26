import copy
import math

import numpy as np
import pytest
import torch

from experiments.k_shot import prediction_metrics, split_candidates, validate_result
from experiments.scenarios import sample
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from tests.fixtures.feedback import Oracle


def test_split_uses_canonical_ids_and_is_response_independent():
    scenes = sample("car_following", 2000, "new", replicate=90)
    split = split_candidates(scenes, 11)
    reversed_scenes = scenes[::-1]
    reverse = split_candidates(reversed_scenes, 11)
    for key in ("query_order", "evaluation_indices"):
        assert [scenes[i].scenario_id for i in split[key]] == [
            reversed_scenes[i].scenario_id for i in reverse[key]
        ]
    assert len(split["query_order"]) == 200
    assert len(split["evaluation_indices"]) == 1800
    assert not set(split["query_order"]) & set(split["evaluation_indices"])


def test_metrics_match_hand_example_and_id_tie_breaking():
    p, y, ids = [0.5, 0.5, 1, 0, 0.99], [0, 1, 1, 0, 0], ["b", "a", "c", "d", "e"]
    metrics, bins = prediction_metrics(p, y, ids, [0, 1, 2, 3], top=(1, 2))
    assert metrics["nll"] == pytest.approx(math.log(2) / 2, abs=1e-12)
    assert metrics["brier"] == 0.125
    assert metrics["auroc"] == 0.875
    assert metrics["auprc_ap"] == pytest.approx(5 / 6)
    assert metrics["recall_2"] == 1
    assert metrics["top_2_hits"] == 2
    assert metrics["ece"] == 0
    assert bins[9]["count"] == 1
    assert sum(b["count"] for b in bins) == 4
    tied, _ = prediction_metrics([0.5, 0.5], [0, 1], ["b", "a"], [0, 1], top=(1,))
    assert tied["recall_1"] == 1


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_single_class_metrics_are_explicitly_undefined(labels):
    metrics, _ = prediction_metrics([0, 1], labels, ["a", "b"], [0, 1], top=(1, 5))
    assert metrics["auroc"] is None
    assert metrics["auroc_undefined"] == "single_class"
    assert math.isfinite(metrics["nll"])
    assert metrics["top_5_count"] == 2
    if not any(labels):
        assert metrics["auprc_ap"] is None and metrics["recall_1"] is None
    else:
        assert metrics["auprc_ap"] == 1 and metrics["recall_1"] == 0.5


@pytest.mark.parametrize("probabilities", [[float("nan"), 0.2], [-0.1, 0.2], [0.1, 1.1]])
def test_invalid_probabilities_fail_before_metric_clipping(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        prediction_metrics(probabilities, [0, 1], ["a", "b"], [0, 1])


def test_saved_result_requires_complete_snapshots_and_the_exact_prefix():
    split = {"candidate_ids": ["a", "b", "c"], "query_order": [1], "evaluation_indices": [0, 2]}
    result = {
        "selected_indices": [1],
        "queries": [{}],
        "prediction_snapshots": [
            {
                "paid_queries": k,
                "observed_indices": [1][:k],
                "probabilities": [0.5] * 3,
                "component_weights": [1],
                "target_parameters": {"mean": 0.0},
            }
            for k in (0, 1)
        ],
    }
    validate_result(result, split, checkpoints=(0, 1))
    for bad in ("missing", "duplicate", "prefix", "probabilities"):
        changed = copy.deepcopy(result)
        if bad == "missing":
            changed["prediction_snapshots"].pop()
        elif bad == "duplicate":
            changed["prediction_snapshots"].append(changed["prediction_snapshots"][-1])
        elif bad == "prefix":
            changed["prediction_snapshots"][-1]["observed_indices"] = [2]
        else:
            changed["prediction_snapshots"][-1]["probabilities"] = [1.2] * 3
        with pytest.raises(ValueError):
            validate_result(changed, split, checkpoints=(0, 1))


def transfer_inputs(scenes):
    base = SharedPrivateKernel(3, deep=False).double()
    return base, {
        "names": ["history_a", "history_b"],
        "weights": np.array([0.5, 0.5]),
        "means": np.stack(
            (np.tile([-0.5, -0.2], (len(scenes), 1)), np.tile([0.7, 0.3], (len(scenes), 1)))
        ),
        "blocks": np.tile(np.eye(2)[None, None] * 0.2, (2, len(scenes), 1, 1)),
    }


def test_transfer_zero_feedback_weight_updates_and_rss_ablation_are_causal():
    torch.set_num_threads(1)
    scenes = sample("car_following", 8, "new", replicate=90)
    labels, margins = np.arange(8) % 2 == 0, np.linspace(-0.7, 0.6, 8)
    base, sources = transfer_inputs(scenes)
    order = [3, 1, 6]
    common = dict(
        candidates=scenes,
        pretrained=base,
        sources=sources,
        initial_indices=order[:1],
        budget=3,
        query_order=order,
        snapshot_counts=(0, 1, 3),
    )
    results = {}
    for mode, updates in (("collision", False), ("collision", True), ("joint", True)):
        first = select(
            oracle=Oracle(scenes, labels, margins),
            feedback_mode=mode,
            update_source_weights=updates,
            **common,
        )
        changed = select(
            oracle=Oracle(scenes, labels, margins + 0.4),
            feedback_mode=mode,
            update_source_weights=updates,
            **common,
        )
        if mode == "collision":
            assert first["prediction_snapshots"] == changed["prediction_snapshots"]
        else:
            assert (
                first["prediction_snapshots"][-1]["probabilities"]
                != changed["prediction_snapshots"][-1]["probabilities"]
            )
        short_args = {**common, "budget": 1, "query_order": order[:1], "snapshot_counts": (0, 1)}
        changed_labels, changed_margins = labels.copy(), margins.copy()
        changed_labels[[1, 6, 7]] = ~changed_labels[[1, 6, 7]]
        changed_margins[[1, 6, 7]] += 1
        short = select(
            oracle=Oracle(scenes, changed_labels, changed_margins),
            feedback_mode=mode,
            update_source_weights=updates,
            **short_args,
        )
        assert first["prediction_snapshots"][:2] == short["prediction_snapshots"]
        results[mode, updates] = first
    zero = [r["prediction_snapshots"][0]["probabilities"] for r in results.values()]
    assert zero[0] == zero[1] == zero[2]
    fixed = results["collision", False]
    assert all(
        s["component_weights"] == fixed["prediction_snapshots"][0]["component_weights"]
        for s in fixed["prediction_snapshots"]
    )
    dynamic = results["collision", True]
    scores = np.log(np.r_[0.675 * sources["weights"], 0.225, 0.1])
    for query in dynamic["queries"]:
        probability = np.array(query["prequery_component_probabilities"])
        likelihood = probability if query["feedback"]["collision"] else 1 - probability
        scores += np.log(np.maximum(likelihood, 1e-15))
        scores -= np.logaddexp.reduce(scores)
        np.testing.assert_allclose(
            query["component_weights_after_query"], np.exp(scores), atol=1e-12
        )


def test_joint_target_only_responds_to_prefix_rss_without_using_future_responses():
    torch.set_num_threads(1)
    scenes = sample("car_following", 8, "new", replicate=90)
    labels, margins = np.arange(8) % 2 == 0, np.linspace(-0.7, 0.6, 8)
    common = dict(
        candidates=scenes,
        initial_indices=[3],
        budget=1,
        query_order=[3],
        snapshot_counts=(0, 1),
        raw_kernel_type=SharedPrivateKernel,
    )
    first = select(oracle=Oracle(scenes, labels, margins), **common)
    changed = select(oracle=Oracle(scenes, ~labels, margins + 0.4), **common)
    assert first["prediction_snapshots"][0] == changed["prediction_snapshots"][0]
    future_labels, future_margins = labels.copy(), margins.copy()
    unqueried = [i for i in range(len(scenes)) if i != 3]
    future_labels[unqueried] = ~future_labels[unqueried]
    future_margins[unqueried] += 1
    future = select(oracle=Oracle(scenes, future_labels, future_margins), **common)
    assert first["prediction_snapshots"] == future["prediction_snapshots"]
    rss_only = select(oracle=Oracle(scenes, labels, margins + 0.4), **common)
    assert (
        first["prediction_snapshots"][-1]["probabilities"]
        != rss_only["prediction_snapshots"][-1]["probabilities"]
    )
    assert first["representation"] == "normalized_parameters"
