"""Paired-family inference must not count selector repeats as samples."""

from methods.failure_memory_regression.nl_statistics import _area_by_family, _signflip_p


def test_exact_signflip_resolution_is_family_limited():
    assert _signflip_p([1.0, 1.0, 1.0]) == 0.25
    assert _signflip_p([0.0, 0.0, 0.0]) == 1.0


def test_family_areas_sum_to_global_early_area():
    queries = [{"rank": 2 * index + 1, "context_id": f"{family}:context",
                "discovery": hit} for index, (family, hit) in enumerate(
                    (("S01", 1), ("S08", 0), ("S08", 1), ("S01", 0)))]
    global_area = 2 * sum(sum(row["discovery"] for row in queries[:j])
                          for j in range(1, len(queries) + 1)) / (len(queries) *
                                                                    (len(queries) + 1))
    assert abs(_area_by_family(queries, "S01") +
               _area_by_family(queries, "S08") - global_area) < 1e-12
