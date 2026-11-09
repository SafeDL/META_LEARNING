import pytest

from experiments.paper_report import aggregate, paired_distances


def test_macro_coverage_and_pooled_coverage_handle_zero_failure_task_differently():
    rows = []
    for failures, truth in ((1, 2), (1, 10), (0, 0)):
        rows.append(
            {
                "method": "joint",
                "metrics": {
                    "unique_collisions": failures,
                    "pool_unique_collisions": truth,
                    "post_initial_collisions": failures,
                    "discovery_auc": failures,
                    "collision_recall": failures / truth if truth else None,
                    "pair_distance": None,
                    "nearest_neighbor_distance": None,
                    "discovery_curve": [failures] * 200,
                },
            }
        )
    result = aggregate(rows, ("joint",))[0]
    assert result["failures"] == 2 and result["tasks"] == 3
    assert result["pooled_coverage"] == pytest.approx(1 / 6)
    assert result["macro_coverage"] == pytest.approx(0.3)
    assert result["coverage_defined_tasks"] == 2
    assert result["pair_distance"] is None


def test_matched_distance_uses_the_earliest_common_prefix_instead_of_final_distance():
    rows = []
    for method, count, pairs, nearest in (
        ("joint", 3, [None, 0.2, 0.4], [None, 0.1, 0.15]),
        ("random", 2, [None, 0.6], [None, 0.3]),
    ):
        rows.append(
            {
                "sut_id": "ads",
                "scenario": "car_following",
                "method": method,
                "metrics": {
                    "unique_collisions": count,
                    "discovery_auc": count,
                    "distances_by_discovery_count": {
                        "pair_distance": pairs,
                        "nearest_neighbor_distance": nearest,
                    },
                },
            }
        )
    pair = paired_distances(rows, ("joint", "random"))[0]
    assert pair["matched_count"] == 2 and pair["delta_failures"] == 1
    assert pair["joint_pair_distance"] == 0.2
    assert pair["control_pair_distance"] == 0.6
