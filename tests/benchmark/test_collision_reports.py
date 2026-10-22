import numpy as np
import pytest

from benchmark.report_collision import (METRICS, aggregate, equal_coverage, plot_curves, plot_survival,
                                        restricted_records, source_weight_curve)


def fixture_rows():
    rows = []
    for lineage, repeats, value in (("a", 4, 2), ("b", 1, 6)):
        for seed in range(repeats):
            for method, delta in (("history_response", 0), ("random", -1)):
                collision = value + delta
                rows.append({"lineage": lineage, "target_id": lineage, "algorithm_seed": seed,
                             "environment_seed": 81001, "scenario": "front_brake", "condition": "new",
                             "method": method, **{m: collision for m in METRICS},
                             "first_collision": {"time": 128, "observed": False},
                             "discovery_curve": [min(i + 1, collision) for i in range(128)],
                             "behavior_curve": [min(i + 1, 2) for i in range(128)]})
    return rows


def test_reports_weight_lineages_and_keep_censored_runs():
    result = aggregate(fixture_rows())
    main = result["methods"]["history_response"]
    assert main["metrics"]["unique_collisions"] == 4
    assert main["independent_lineages"] == 2
    assert main["first_collision"]["censored_runs"] == 5
    assert main["first_collision"]["restricted_mean_queries"] == 128
    assert all(row["difference"] == (0 if row["metric"] == "restricted_first_queries" else 1)
               for row in result["paired_lineage_intervals"])
    assert all(row["paired_runs"] == 5 for row in result["paired_lineage_intervals"])
    coverage = equal_coverage(fixture_rows())
    assert len(coverage) == 5
    assert {row["common_maximum"] for row in coverage} == {1, 5}
    with pytest.raises(ValueError, match="counterpart"):
        aggregate(fixture_rows()[:-1])


def test_paper_plot_exports_standalone_vector_and_raster_artifacts(tmp_path):
    plot_curves(fixture_rows(), "fixture", "Numerical verification fixture", destination=tmp_path)
    assert {p.suffix for p in tmp_path.iterdir()} == {".pdf", ".svg", ".png"}
    assert all(p.stat().st_size > 1000 for p in tmp_path.iterdir())
    plot_survival(fixture_rows(), "survival_fixture", "Censoring verification fixture", destination=tmp_path)
    assert len(list(tmp_path.iterdir())) == 6


def test_adaptation_report_preserves_family_mass_and_independent_target_mass():
    queries = [{"source_names": ["history_idm_mobil", "history_fvdm_lane", "history_ppo", "target_only"],
                "source_weights": [0.1, 0.2, 0.3, 0.4]}]
    values = source_weight_curve(queries)
    np.testing.assert_allclose(values, [[0.3, 0, 0, 0.3, 0.4]])
    queries[0]["source_weights"] = [0.1, 0.2, -0.1, 0.8]
    with pytest.raises(ValueError, match="Invalid mixture"):
        source_weight_curve(queries)


def test_restricted_mean_keeps_the_censoring_flags_and_uses_a_common_window():
    rows = fixture_rows()
    rows[0]["first_collision"] = {"time": 2, "observed": True}
    converted = restricted_records(rows)
    assert converted[0]["restricted_first_queries"] == 2
    assert converted[1]["restricted_first_queries"] == 128
    assert converted[1]["first_collision"]["observed"] is False
    rows[1]["first_collision"] = {"time": 32, "observed": False}
    with pytest.raises(ValueError, match="common administrative"):
        restricted_records(rows)


def test_survival_summary_weights_lineages_equally_when_repeat_counts_differ():
    rows = fixture_rows()
    for row in rows:
        row["first_collision"] = ({"time": 2, "observed": True} if row["lineage"] == "a"
                                   else {"time": 128, "observed": False})
    result = aggregate(rows)["methods"]["history_response"]["first_collision"]
    assert result["restricted_mean_queries"] == 65
    assert result["independent_lineages"] == 2
    assert result["curve"][1] == {"time": 2, "survival": 0.5}


def test_paired_first_discovery_contrast_retains_runs_without_any_collision():
    rows = fixture_rows()
    for row in rows:
        row["first_collision"] = ({"time": 2, "observed": True} if row["method"] == "history_response"
                                   else {"time": 128, "observed": False})
    result = aggregate(rows)
    contrast = next(r for r in result["paired_lineage_intervals"] if r["metric"] == "restricted_first_queries")
    assert contrast["difference"] == -126
    assert contrast["interval_95"] == [-126, -126]
    assert contrast["lower_is_earlier"]
    assert result["methods"]["random"]["first_collision"]["censored_runs"] == 5
