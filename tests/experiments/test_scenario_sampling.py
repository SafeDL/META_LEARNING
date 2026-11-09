import json

import numpy as np
import pytest
from scipy.stats import qmc

from experiments.domain import Domain
from experiments.scenario_sampling import sample_candidates


def domain(dimension):
    return Domain(
        {f"x{i}": (0, 1) for i in range(dimension)},
        {f"x{i}": 0.1 for i in range(dimension)},
        {},
        fixed_categories={"side": "left"},
    )


@pytest.mark.parametrize("dimension", (2, 3, 4, 7))
def test_candidate_count_is_2000_at_every_dimension_and_fixed_fields_are_not_sampled(dimension):
    definition = domain(dimension)
    scenes, report = sample_candidates(
        "example",
        definition,
        count=2000,
        seed=62011,
        namespace="target:10",
        feasible=lambda s: True,
        max_draws=2000,
    )
    assert len(scenes) == len({s.coordinate() for s in scenes}) == 2000
    assert report["examined_points"] == 2000 and report["accepted_count"] == 2000
    assert definition.features(scenes).shape == (2000, dimension)
    assert all(not s.categories for s in scenes)
    assert all(definition.execution_values(s)[1] == {"side": "left"} for s in scenes)


def test_sobol_mapping_matches_independent_ranges_and_does_not_quantize_continuous_values():
    definition = Domain(
        {"gap": (10, 50), "deceleration": (2, 6)},
        {"gap": 5, "deceleration": 1},
        {"weather": ("dry", "wet", "snow")},
    )
    scenes, report = sample_candidates(
        "example",
        definition,
        count=16,
        seed=11,
        namespace="check",
        feasible=lambda s: True,
        max_draws=16,
    )
    reference = qmc.Sobol(3, scramble=True, seed=11).random_base2(4)
    actual = np.array([[s.parameters["gap"], s.parameters["deceleration"]] for s in scenes])
    np.testing.assert_allclose(actual, [10, 2] + reference[:, :2] * [40, 4], atol=1e-14, rtol=0)
    assert [s.categories["weather"] for s in scenes] == [
        definition.categories["weather"][int(v * 3)] for v in reference[:, 2]
    ]
    assert not np.allclose((actual[:, 0] - 10) / 5, np.round((actual[:, 0] - 10) / 5))
    assert report["geometry_rejections"] == report["duplicate_coordinates"] == 0


def test_coordinate_only_filter_is_deterministic_audited_and_retains_original_point_ids():
    definition = domain(2)

    def feasible(scene):
        return scene.parameters["x0"] + scene.parameters["x1"] < 0.8

    options = dict(count=128, seed=11, namespace="history", feasible=feasible, max_draws=1024)
    scenes, report = sample_candidates("example", definition, **options)
    restored = Domain.from_dict(json.loads(json.dumps(definition.to_dict(), sort_keys=True)))
    repeated, repeated_report = sample_candidates("example", restored, **options)
    assert scenes == repeated and report == repeated_report and all(map(feasible, scenes))
    assert report["geometry_rejections"] > 0
    assert report["examined_points"] == report["accepted_count"] + report["geometry_rejections"]
    ids = [int(s.scenario_id.rsplit(":", 1)[1]) for s in scenes]
    assert (
        ids == sorted(ids) and max(ids) + 1 == report["examined_points"] and ids != list(range(128))
    )


def test_duplicate_coordinates_are_not_rechecked_and_incomplete_libraries_are_rejected():
    definition = Domain({}, {}, {"side": ("left", "right")})
    checked = []

    def feasible(scene):
        checked.append(scene.coordinate())
        return True

    with pytest.raises(ValueError, match="accepted=2/3.*duplicates=30"):
        sample_candidates(
            "example",
            definition,
            count=3,
            seed=11,
            namespace="check",
            feasible=feasible,
            max_draws=32,
        )
    assert len(checked) == len(set(checked)) == 2


def test_geometry_errors_stop_sampling_instead_of_becoming_safe_or_rejected_results():
    def broken(scene):
        raise RuntimeError("geometry failure")

    with pytest.raises(RuntimeError, match="geometry failure"):
        sample_candidates(
            "example", domain(2), count=4, seed=11, namespace="check", feasible=broken, max_draws=16
        )


@pytest.mark.parametrize(
    "change",
    [
        {"count": 0},
        {"count": 2.5},
        {"max_draws": 3},
        {"seed": None},
        {"namespace": ""},
    ],
)
def test_preparation_requires_fixed_count_seed_namespace_and_draw_limit(change):
    with pytest.raises(ValueError):
        sample_candidates(
            "example",
            domain(2),
            **{
                "count": 4,
                "seed": 11,
                "namespace": "check",
                "feasible": lambda s: True,
                "max_draws": 16,
                **change,
            },
        )


def test_history_and_target_designs_use_independent_seeds_without_ads_information():
    definition = domain(3)
    arguments = dict(count=64, feasible=lambda s: True, max_draws=64)
    history, _ = sample_candidates(
        "example", definition, seed=31013, namespace="history", **arguments
    )
    target, _ = sample_candidates(
        "example", definition, seed=62011, namespace="target:10", **arguments
    )
    assert [s.coordinate() for s in history] != [s.coordinate() for s in target]


def test_nested_supplement_skips_existing_coordinates_without_reassigning_point_ids():
    definition = domain(3)
    options = dict(seed=62011, feasible=lambda scene: True, max_draws=32)
    original, _ = sample_candidates("example", definition, count=8, namespace="original", **options)
    extra, report = sample_candidates(
        "example", definition, count=8, namespace="supplement", existing=original, **options
    )
    assert not {s.coordinate() for s in original} & {s.coordinate() for s in extra}
    assert report["duplicate_coordinates"] == 8 and report["examined_points"] == 16
    assert [int(s.scenario_id.rsplit(":", 1)[1]) for s in extra] == list(range(8, 16))
