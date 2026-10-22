import pytest

from benchmark.audit_collision_statistics import paired_summary, survival_summary


def record(lineage, seed, method, value=0, *, time=128, observed=False):
    return {"lineage": lineage, "target_id": f"target_{lineage}", "scenario": "fixture", "condition": "fixture",
            "environment_seed": 81001, "algorithm_seed": seed, "method": method,
            "unique_collisions": value, "first_collision": {"time": time, "observed": observed}}


def test_hierarchical_intervals_weight_unequal_lineages_not_seed_counts():
    rows = [record("a", 0, "ours", 10), record("a", 0, "reference", 0)]
    for seed in range(100):
        rows.extend([record("b", seed, "ours", 0), record("b", seed, "reference", 0)])
    result = paired_summary(rows, "ours", "reference", "unique_collisions")
    assert result["difference"] == 5
    assert result["difference"] != 10 / 101
    assert result["independent_lineages"] == 2 and result["paired_runs"] == 101
    assert result["lineage_differences"] == {"a": 10, "b": 0}
    assert result["interval_95"] == [0, 10]


@pytest.mark.parametrize("corruption", ["missing", "duplicate"])
def test_statistical_pairs_cannot_omit_or_duplicate_an_algorithm_run(corruption):
    rows = [record("a", 11, "ours", 4), record("a", 11, "reference", 1)]
    if corruption == "missing":
        rows.pop()
    else:
        rows.append(rows[0].copy())
    with pytest.raises(ValueError, match="counterpart"):
        paired_summary(rows, "ours", "reference", "unique_collisions")


def test_censored_discoveries_remain_in_equal_lineage_survival_and_rmst():
    rows = [record("a", 11, "ours", time=1, observed=True), record("a", 23, "ours"),
            record("b", 11, "ours", time=12, observed=True)]
    result = survival_summary(rows, restriction=128)
    assert result["restricted_mean_queries"] == 38.25
    assert result["censored_runs"] == 1 and result["independent_lineages"] == 2
    assert result["curve"] == [{"time": 0, "survival": 1}, {"time": 1, "survival": .75},
                               {"time": 12, "survival": .25}, {"time": 128, "survival": .25}]
    rows[1]["first_collision"]["time"] = 64
    with pytest.raises(ValueError, match="common administrative"):
        survival_summary(rows, restriction=128)
