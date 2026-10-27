import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.baselines.av_bo_rf_ei.method import (
    Config,
    RandomForestEI,
    expected_improvement,
    jackknife_variance,
    risk,
    run,
)


def test_grouped_jackknife_has_paper_pseudovalue_scale():
    actual = jackknife_variance([4, 5], [[1, 3], [2, 3], [3, 3]])
    assert actual == pytest.approx([4 / 3, 0])
    # Ordinary fold sample variance is 1, so it would fail this check.
    with pytest.raises(ValueError, match="two fitted folds"):
        jackknife_variance([4], [[1]])


def test_ei_matches_gaussian_integral_and_zero_variance_limit():
    for mean, variance in [(-2, 0.3), (0, 1), (1, 4)]:
        reference = quad(lambda y: y * norm.pdf(y, loc=mean, scale=math.sqrt(variance)), 0, np.inf)[
            0
        ]
        assert float(expected_improvement(mean, variance, 0)) == pytest.approx(reference, abs=1e-10)
    assert expected_improvement([-1, 0, 2], [0, 0, 0], 0) == pytest.approx([0, 0, 2])


def test_risk_keeps_impact_severity_and_negative_ttc_instead_of_binary_labels():
    safe = QueryFeedback("safe", False, SafetyMeasurements(3, 2), 1)
    collision = QueryFeedback("crash", True, SafetyMeasurements(0, 0, impact_delta_v_mps=7), 1)
    assert risk(safe) == -2
    assert risk(collision) == 7
    assert risk(replace(safe, safety=SafetyMeasurements(3, math.inf))) == -10
    with pytest.raises(ValueError, match="measured impact"):
        risk(replace(collision, safety=SafetyMeasurements(0, 0)))


def test_rf_jackknife_is_reproducible_and_not_between_tree_spread():
    x = np.arange(8).reshape(-1, 1) / 7
    y = np.array([-10, -5, -2, -1, 0, 1, 3, 7])
    config = Config(n_estimators=10, max_depth=3)
    a = RandomForestEI(config=config).fit(x, y)
    b = RandomForestEI(config=config).fit(x, y)
    for left, right in zip(a.predict(x), b.predict(x)):
        assert left == pytest.approx(right)
    assert len(a.fold_models) == 8
    tree_variance = np.var([tree.predict(x) for tree in a.model.estimators_], axis=0, ddof=1)
    assert not np.allclose(a.predict(x)[1], tree_variance)


def test_tpe_tunes_only_the_paper_table_parameters_using_paid_sample_count():
    config = Config(n_estimators=10, max_depth=3, tune_every=10, tuning_trials=2)
    model = RandomForestEI(config=config).fit(np.arange(10).reshape(-1, 1), np.arange(10))
    assert model.tuning[0]["paid_observations"] == 10
    assert model.tuning[0]["trials"] == 2
    assert 10 <= model.parameters["n_estimators"] <= 300
    assert 3 <= model.parameters["max_depth"] <= 30
    assert 2 <= model.parameters["min_samples_split"] <= 10


def test_isolated_rf_ei_pays_for_common_initialization_and_never_repeats():
    scenes = sample("cut_in", 12, "development", replicate=61)

    def execute(scene):
        value = float(scene.parameters["gap_m"])
        label = value < 20
        safety = SafetyMeasurements(value, value / 5, impact_delta_v_mps=3.0 if label else None)
        return PaidExecution(QueryFeedback(scene.scenario_id, label, safety, 1), [])

    ledger = QueryLedger(scenes, 8, execute)
    result = run_isolated(
        run,
        {
            "candidates": scenes,
            "initial_indices": [5, 2, 8, 1],
            "budget": 8,
            "config": Config(n_estimators=10, max_depth=3),
        },
        ledger,
    )
    assert result["selected_indices"][:4] == [5, 2, 8, 1]
    assert len(set(result["selected_indices"])) == result["actual_queries"] == 8
    assert [r["purpose"] for r in ledger.records] == ["initial"] * 4 + ["target"] * 4
    assert all("expected_improvement" not in r for r in result["queries"][:4])
    assert all(r["expected_improvement"] >= 0 for r in result["queries"][4:])
    assert all(r["prequery_probability"] is None for r in result["queries"])
