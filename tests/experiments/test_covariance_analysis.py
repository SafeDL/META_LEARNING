import copy

import numpy as np
import pytest

from experiments.analyze_covariance import choose_covariance, compare_full_replay, diagonal_sources


def test_covariance_control_preserves_mean_marginal_variance_and_psd():
    rng = np.random.default_rng(11)
    factors = rng.normal(size=(3, 8, 8))
    covariance = factors @ factors.transpose(0, 2, 1)
    source = {
        "means": rng.normal(size=(3, 8)),
        "covariances": covariance,
        "weights": np.ones(3) / 3,
        "names": ["a", "b", "c"],
    }
    diagonal = diagonal_sources(source)
    assert diagonal["means"] is source["means"]
    assert diagonal["weights"] is source["weights"]
    for original, changed in zip(covariance, diagonal["covariances"]):
        assert changed.diagonal() == pytest.approx(original.diagonal())
        assert np.linalg.eigvalsh(changed).min() >= 0
        assert changed == pytest.approx(np.diag(original.diagonal()))
    assert source["covariances"] is covariance


def test_covariance_adoption_uses_paired_primary_metrics_and_only_then_auc():
    def summary(count, recall, auc):
        return {
            "unique_collisions": {"mean": count},
            "collision_recall": {"mean": recall},
            "discovery_auc": {"mean": auc},
        }

    full = summary(37, 0.98, 24)
    assert choose_covariance({"full": full, "diagonal": summary(37, 0.98, 24)}) == "full"
    assert choose_covariance({"full": full, "diagonal": summary(37, 0.98, 25)}) == "diagonal"
    assert choose_covariance({"full": full, "diagonal": summary(38, 0.99, 23)}) == "diagonal"
    assert choose_covariance({"full": full, "diagonal": summary(36, 0.99, 26)}) == "full"


def test_full_replay_rejects_changed_predictions_likelihoods_weights_or_selection():
    query = {
        "prequery_probability": 0.5,
        "prequery_component_probabilities": [0.4, 0.6, 0.5],
        "prequery_component_weights": [0.45, 0.45, 0.1],
        "prequery_group_log_likelihood": [-0.2, -0.3],
        "group_weights_after_query": [0.6, 0.4],
    }
    actual = {"selected_indices": [3], "queries": [query]}
    expected = {
        "selected_indices": [3],
        "queries": [
            {
                **{
                    k: query[k]
                    for k in (
                        "prequery_probability",
                        "prequery_component_probabilities",
                        "prequery_component_weights",
                    )
                },
                "prequery_log_likelihood": query["prequery_group_log_likelihood"],
                "component_weights_after_query": [0.54, 0.36, 0.1],
            }
        ],
    }
    assert compare_full_replay(actual, expected) == 0
    changed = copy.deepcopy(actual)
    changed["queries"][0]["prequery_probability"] += 1e-4
    with pytest.raises(ValueError, match="differs"):
        compare_full_replay(changed, expected)
    changed = copy.deepcopy(actual)
    changed["selected_indices"] = [4]
    with pytest.raises(ValueError, match="sequence"):
        compare_full_replay(changed, expected)
