import numpy as np
import pytest

from experiments.analyze_gains import failure_decomposition, oracle_limits, weight_evidence


def auc(positions, budget):
    return np.mean([sum(q <= t for q in positions.values()) for t in range(1, budget + 1)])


def test_manual_replacement_and_timing_reconstruct_discovery_auc():
    old, new = {0: 1, 1: 4}, {0: 1, 2: 2, 3: 4}
    row = failure_decomposition(old, new, 4)
    assert row["common_failures"] == 1
    assert row["added_failures"] == 2 and row["lost_failures"] == 1
    assert row["delta_failures"] == 1
    assert row["common_timing_auc"] == 0
    assert row["added_auc"] == 1 and row["lost_auc"] == -0.25
    assert auc(new, 4) - auc(old, 4) == 0.75


def test_shared_failures_can_be_slower_even_when_final_count_increases():
    old, new = {0: 1}, {0: 3, 1: 4}
    row = failure_decomposition(old, new, 4)
    assert row["delta_failures"] == 1 and row["common_timing_auc"] == -0.5
    assert row["added_auc"] == 0.25
    assert row["common_timing_auc"] + row["added_auc"] + row["lost_auc"] == auc(new, 4) - auc(
        old, 4
    )


@pytest.mark.parametrize(
    "truth,initial,expected", [(0, [0], (0, 0)), (2, [0], (2, 1.25)), (5, [1], (4, 2.5))]
)
def test_oracle_bound_preserves_actual_initialization_and_stops_at_available_truth(
    truth, initial, expected
):
    result = oracle_limits(truth, initial, 4)
    assert (result["oracle_failures"], result["oracle_auc"]) == expected


def weight_record():
    return {
        "group_prior_weights": [0.5, 0.5],
        "queries": [
            {
                "feedback": {"collision": True},
                "prequery_component_probabilities": [0.8, 0.2, 0.5],
                "prequery_component_weights": [0.45, 0.45, 0.1],
                "prequery_probability": 0.5,
            },
            {
                "feedback": {"collision": False},
                "prequery_component_probabilities": [0.8, 0.2, 0.5],
                "prequery_component_weights": [0.72, 0.18, 0.1],
                "prequery_probability": 0.662,
            },
        ],
    }


def test_prequery_weight_evidence_uses_the_realized_sequence_and_likelihood_identity():
    evidence = weight_evidence(weight_record())
    assert evidence["sequence_dynamic_nll"] == pytest.approx(-np.log(0.5 * 0.338))
    assert evidence["sequence_fixed_prior_nll"] == pytest.approx(-np.log(0.25))
    assert evidence["sequence_nll_change"] > 0
    assert evidence["mixture_identity_error"] < 1e-12
    assert evidence["likelihood_floor_activations"] == 0


def test_prequery_weight_evidence_rejects_weights_contaminated_by_current_feedback():
    record = weight_record()
    record["queries"][1]["prequery_component_weights"] = [0.45, 0.45, 0.1]
    with pytest.raises(ValueError, match="sequential updating"):
        weight_evidence(record)
