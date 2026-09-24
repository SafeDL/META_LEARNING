from __future__ import annotations

import csv
import json
from dataclasses import fields

import numpy as np

from highway_env_benchmark.data.response_bank import ResponseBank
from sut_algorithms.highway_env.idm_profiles import SUTProfile
from method_chains.function_posterior_search.benchmark import (
    CONFIRMATION_SEEDS,
    HETEROGENEITY_LEVELS,
    MODES,
    REGIME_COUNTS,
    SCENARIO_COUNT,
    SOURCE_NAMES,
    TARGET_COVERAGES,
    generate_confirmation_scenarios,
    physical_episode_count,
    source_releases,
    target_releases,
)
from method_chains.function_posterior_search.confirmation import (
    ADAPTIVE_METHOD,
    BANK_DIR,
    BUDGETS,
    METHODS,
    OUTPUT,
    RANDOM_REPEATS,
)


def _signature(profile: SUTProfile) -> tuple[object, ...]:
    return tuple(
        getattr(profile, field.name)
        for field in fields(profile)
        if field.name != "name"
    )


def test_confirmation_release_factorial_and_coverage_are_frozen() -> None:
    sources, targets = source_releases(), target_releases()
    assert len(sources) == len(SOURCE_NAMES) == 6
    assert len(targets) == 3 * 3 * 2 == 18
    assert physical_episode_count() == 8200
    assert len({release.name for release in (*sources, *targets)}) == 24
    cells = [(release.coverage, release.heterogeneity) for release in targets]
    assert {cell: cells.count(cell) for cell in set(cells)} == {
        (coverage, heterogeneity): 2
        for coverage in TARGET_COVERAGES
        for heterogeneity in HETEROGENEITY_LEVELS
    }
    source_signatures = {
        mode: {_signature(release.modules[index]) for release in sources}
        for index, mode in enumerate(MODES)
    }
    for release in targets:
        for index, mode in enumerate(MODES):
            present = _signature(release.modules[index]) in source_signatures[mode]
            assert present == (release.coverage == "exact")


def test_confirmation_scenarios_are_balanced_without_outcome_filtering() -> None:
    for seed in CONFIRMATION_SEEDS:
        anchors, modes, controls, regimes = generate_confirmation_scenarios(seed)
        assert anchors.shape == (SCENARIO_COUNT, 2)
        assert controls.shape == (SCENARIO_COUNT, 2)
        assert np.all((controls >= 0.0) & (controls <= 1.0))
        for mode in MODES:
            mask = modes == mode
            assert int(mask.sum()) == 100
            assert {
                regime: int(np.sum(regimes[mask] == regime))
                for regime in REGIME_COUNTS
            } == REGIME_COUNTS


def test_confirmation_artifacts_recompute_queries_and_metrics() -> None:
    with (OUTPUT / "records.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for seed in CONFIRMATION_SEEDS:
        bank = ResponseBank.load(BANK_DIR / f"response_bank_{seed}.npz")
        for release in target_releases():
            target_index = bank.index_of(release.name)
            truth = bank.collisions[target_index] | bank.near_misses[target_index]
            selected_rows = [
                row for row in rows
                if int(row["seed"]) == seed and row["target_sut"] == release.name
            ]
            for method in METHODS:
                method_rows = [row for row in selected_rows if row["method"] == method]
                expected_repeats = RANDOM_REPEATS if method == "Random" else 1
                assert len(method_rows) == len(BUDGETS) * expected_repeats
                for repeat in range(expected_repeats):
                    by_budget = {
                        int(row["budget"]): row for row in method_rows
                        if int(row["repeat"]) == repeat
                    }
                    previous: list[int] = []
                    for budget in BUDGETS:
                        row = by_budget[budget]
                        queried = [
                            int(value) for value in row["queried_indices"].split(";")
                        ]
                        assert len(queried) == len(set(queried)) == budget
                        assert queried[:len(previous)] == previous
                        assert int(row["critical_events_found"]) == int(
                            truth[queried].sum()
                        )
                        expected = truth[queried].sum() / max(1, truth.sum())
                        assert np.isclose(float(row["critical_recall"]), expected)
                        previous = queried
                    if method in {ADAPTIVE_METHOD, "Function-Conditioned Mining"}:
                        support = [
                            int(value)
                            for value in by_budget[10]["queried_indices"].split(";")
                        ]
                        assert set(bank.modes[support]) == set(MODES)


def test_confirmation_selection_and_summary_use_independent_units() -> None:
    with (OUTPUT / "source_selection.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        tuning = list(csv.DictReader(stream))
    assert len(tuning) == len(CONFIRMATION_SEEDS) * len(SOURCE_NAMES) * 3
    assert all(row["target_data_used"] == "False" for row in tuning)
    with (OUTPUT / "summary.csv").open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    overall = [
        row for row in summary
        if row["scope"] == "overall" and int(row["budget"]) == 50
    ]
    assert {row["method"] for row in overall} == set(METHODS)
    assert {int(row["seed_target_units"]) for row in overall} == {5 * 18}
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen independent confirmation"
    assert manifest["target_count"] == 18
    assert manifest["new_simulator_banks"] == 5
    assert manifest["total_new_physical_episodes"] == 41000
    with (OUTPUT / "subgroup_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        subgroups = list(csv.DictReader(stream))
    assert len(subgroups) == (len(MODES) + len(REGIME_COUNTS)) * len(METHODS) * len(BUDGETS)
    assert {int(row["seed_target_units"]) for row in subgroups} == {5 * 18}
