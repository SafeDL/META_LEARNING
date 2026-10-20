import numpy as np
import pytest

from methods.gcrt.calibration_effectiveness import (
    BIAS_GROUPS, CALIBRATION_SUTS, METHODS, MCTS, bias_group, calibrate,
    configuration_metrics, paired_metrics, probability_bin, reliability_bins,
    stopping_metrics, summarize_predictions, summarize_stopping,
    typical_configurations,
)


def test_paired_scores_use_exact_count_and_retention_event():
    base = np.array([0.5625, 0.375, 0.0625])
    corrected = np.array([0.25, 0.5, 0.25])
    check = {
        "query": 20, "found": 19, "allowed_misses": 1, "deterministic": False,
        "tilt": float(np.log(3)), "completed_segments": 4, "segment_rate_bias": 0.15,
        "base_remaining_log_probability": np.log(base).tolist(),
        "remaining_log_probability": np.log(corrected).tolist(),
        "base_remaining_mean": 0.5, "remaining_mean": 1.0,
        "base_score": 0.9375, "score": 0.75,
    }
    result = paired_metrics(check, 2, budget=22)
    assert result["event"] == 0
    assert result["base_mae"] == 1.5 and result["corrected_mae"] == 1.0
    assert result["base_nll"] == pytest.approx(-np.log(0.0625))
    assert result["corrected_nll"] == pytest.approx(-np.log(0.25))
    assert result["base_brier"] == 0.9375 ** 2
    assert result["corrected_brier"] == 0.75 ** 2
    assert result["delta_mae"] == -0.5
    assert result["eta"] == pytest.approx(np.log(3))
    # When all remaining counts are allowed, the event is certain for both PMFs.
    check.update(allowed_misses=2, deterministic=True, base_score=1.0, score=1.0)
    certain = paired_metrics(check, 2, budget=22)
    assert certain["event"] == 1 and certain["corrected_brier"] == certain["base_brier"] == 0


def test_group_boundaries_and_configuration_equal_weighting():
    assert bias_group(-0.050001) == "underprediction"
    assert bias_group(0.050001) == "overprediction"
    assert bias_group(-0.05) == bias_group(0.05) == "small_bias"
    checks = []
    for name, errors in (("a", [0.1, 0.1, 0.1]), ("b", [-0.1])):
        for error in errors:
            checks.append({
                "configuration_id": name, "family": "cutin", "sut": "idm_ref", "pool": 0,
                "base_rate_error": error, "u": 0.0, "segment_rate_bias": 0.0,
                "mean_change": 0.0, "score_change": 0.0,
                **{f"{prefix}_{metric}": (1.0 if name == "a" else 3.0)
                   for prefix in (*METHODS, "delta") for metric in ("mae", "nll", "brier")},
            })
    configurations = configuration_metrics(checks)
    total = summarize_predictions(configurations)[0]
    assert total["configurations"] == 2 and total["checkpoints"] == 4
    assert total["base_mae_mean"] == 2.0  # Not the checkpoint-pooled mean, 1.5.
    assert total["base_mae_sd"] == pytest.approx(np.sqrt(2))
    assert [r["bias_group"] for r in configurations] == ["overprediction", "underprediction"]
    assert all(r["bias_group"] in BIAS_GROUPS for r in checks)


def test_probability_bins_include_boundaries_and_do_not_fill_empty_bins():
    assert [probability_bin(p) for p in (0, 0.2, 0.4, 0.6, 0.8, 1)] == [0, 1, 2, 3, 4, 4]
    checks = [
        {"configuration_id": "a", "family": "cutin", "base_score": 0.0, "corrected_score": 0.2, "event": 0},
        {"configuration_id": "a", "family": "cutin", "base_score": 1.0, "corrected_score": 1.0, "event": 1},
        {"configuration_id": "b", "family": "cutin", "base_score": 0.1, "corrected_score": 0.6, "event": 1},
    ]
    bins = reliability_bins(checks)
    first = next(r for r in bins if r["family"] == "cutin" and r["method"] == "base" and r["bin"] == 0)
    assert first["checkpoints"] == first["configurations"] == 2
    assert first["probability_mean"] == 0.05 and first["event_frequency"] == 0.5
    empty = next(r for r in bins if r["family"] == "cutin" and r["method"] == "base" and r["bin"] == 1)
    assert empty["checkpoints"] == 0 and empty["probability_mean"] is empty["event_frequency"] is None
    assert sum(r["checkpoints"] for r in bins if r["method"] == "corrected") == 3


def test_both_methods_hold_mcts_out_of_brake_calibration():
    replays = []
    for method, score in (("gcrt", 0.8), ("without_correction", 0.7)):
        for sut in (*CALIBRATION_SUTS["brake"], MCTS, "fvdm_target"):
            for pool in (0, 1, 2):
                replays.append({
                    "family": "brake", "method": method, "sut": sut, "pool": pool,
                    "full_found": 100, "forecast": [
                        {"query": 10, "found": 10, "deterministic": False,
                         "score": score if sut in CALIBRATION_SUTS["brake"] else 1.0},
                    ],
                })
    for method, expected in (("base", 0.7), ("corrected", 0.8)):
        result = calibrate(replays, "brake", method)
        assert result["threshold"] == expected and result["configurations"] == 9
        assert MCTS not in result["controllers"] and "fvdm_target" not in result["controllers"]
        assert result["unsafe_checkpoints"] == 9


def test_stopping_is_first_strict_trigger_and_zero_discoveries_are_undefined():
    trace = {"sut": "idm_ref", "pool": 0, "found": list(range(1, 201))}
    checks = [
        {"configuration_id": "a", "family": "cutin", "query": 10,
         "base_score": 0.8, "corrected_score": 0.8, "deterministic": False},
        {"configuration_id": "a", "family": "cutin", "query": 15,
         "base_score": 0.81, "corrected_score": 0.81, "deterministic": False},
    ]
    positive = stopping_metrics(trace, checks, "base", 0.8)
    assert positive["queries"] == positive["found"] == 15
    assert not positive["target_met"]
    trace["found"] = [0] * 200
    zero = stopping_metrics(trace, checks, "base", 0.8)
    assert zero["retention"] is zero["target_met"] is None
    corrected = {**zero, "method": "corrected"}
    summary = summarize_stopping([positive, zero, corrected])
    base = next(r for r in summary if r["family"] == "all" and r["method"] == "base")
    assert base["valid_configurations"] == base["zero_discovery_configurations"] == 1
    assert base["retention_mean"] == positive["retention"]


def test_typical_configuration_uses_median_bias_and_deterministic_tie_break():
    configurations = [
        {"configuration_id": name, "bias_group": "overprediction", "base_rate_bias": bias}
        for name, bias in (("z", 0.1), ("a", 0.3))
    ]
    checks = [{**r, "query": t} for r in configurations for t in (10, 15, 50)]
    selected = typical_configurations(configurations, checks)
    assert [r["query"] for r in selected] == [10, 50]
    assert all(r["configuration_id"] == "a" for r in selected)
