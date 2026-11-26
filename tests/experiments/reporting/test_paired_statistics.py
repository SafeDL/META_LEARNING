"""Feedback repeats and undefined scene metrics do not become extra ADS samples."""

import copy

import numpy as np
import pytest

from experiments.paired_statistics import json_ready, paired_summary


def summarize(values, **kwargs):
    return paired_summary(
        values,
        ads_ids=[f"ads-{i}" for i in range(len(values))],
        ads_families=["rule" if i < len(values) / 2 else "learning" for i in range(len(values))],
        methods=["target_only", "ours"],
        replicates=300,
        **kwargs,
    )


def test_scenes_are_averaged_within_ads_before_the_equal_ads_mean():
    values = np.array(
        [
            [[[1, 0], [1, 0]], [[1, 0], [1, 0]]],
            [[[3, 2], [3, 2]], [[np.nan, np.nan], [np.nan, np.nan]]],
        ]
    )
    result = summarize(values)
    np.testing.assert_allclose(result["unpaired"]["mean"], [2, 1])
    np.testing.assert_allclose(result["paired"]["gain"], [0, 1])
    np.testing.assert_allclose(result["paired"]["gain_ci_low"], [0, 1])
    np.testing.assert_allclose(result["paired"]["gain_ci_high"], [0, 1])
    assert result["paired"]["valid_tasks"].tolist() == [3, 3]
    assert result["paired"]["valid_scenes_per_ads"].tolist() == [[2, 2], [1, 1]]


def test_pairing_uses_common_seed_cells_before_averaging_either_method():
    values = np.array([[[[1, 0], [100, np.nan]]], [[[3, 2], [300, np.nan]]]])
    result = summarize(values)
    assert result["unpaired"]["mean"][0] == 101
    assert result["paired"]["reference_mean"][1] == 2
    assert result["paired"]["method_mean"][1] == 1
    assert result["paired"]["gain"][1] == 1
    assert result["paired"]["valid_seed_runs"].tolist() == [4, 2]


def test_duplicate_feedback_repeats_do_not_narrow_the_ads_bootstrap():
    values = np.array([[[[2, 1], [8, 7]]], [[[4, 2], [10, 8]]], [[[3, 0], [9, 6]]]])
    original = summarize(values)
    duplicated = summarize(np.repeat(values, 5, axis=2))
    for scope in ("unpaired", "paired"):
        names = (
            ("mean", "ci_low", "ci_high")
            if scope == "unpaired"
            else ("gain", "gain_ci_low", "gain_ci_high")
        )
        for field in names:
            np.testing.assert_allclose(original[scope][field], duplicated[scope][field])
    assert duplicated["design"]["registered_ads"] == original["design"]["registered_ads"] == 3


def test_zero_collision_metric_cells_remain_undefined_with_actual_valid_counts():
    values = np.full((8, 1, 5, 2, 2), np.nan)
    values[..., 0] = 0.0  # Brier is defined even for a zero-collision task.
    values[:2, ..., 1] = [0.2, 0.4]  # Recall is defined for two ADSs only.
    result = summarize(values, lower_better=[True, False])
    np.testing.assert_array_equal(result["paired"]["valid_ads"], [[8, 2], [8, 2]])
    assert result["unpaired"]["mean"][1, 0] == 0.0
    assert result["paired"]["gain"][1, 1] == pytest.approx(0.2)
    assert result["design"]["registered_ads"] == 8
    assert result["paired"]["valid_bootstrap_replicates"][1, 1] < 300


def test_all_undefined_cells_serialize_as_null_while_observed_zero_is_kept():
    values = np.full((2, 1, 2, 2, 2), np.nan)
    values[..., 0] = 0
    result = json_ready(summarize(values))
    assert result["unpaired"]["mean"] == [[0.0, None], [0.0, None]]
    assert result["paired"]["valid_ads"] == [[2, 0], [2, 0]]
    assert result["paired"]["valid_bootstrap_replicates"] == [[300, 0], [300, 0]]


def test_whole_ads_resampling_preserves_curve_pairing_and_metric_directions():
    values = np.zeros((3, 2, 5, 2, 2, 2))
    for i in range(3):
        values[i, :, :, 0, :, :] = [i + 2, 0.1 * i]
        values[i, :, :, 1, :, :] = [i + 1, 0.1 * i + 0.2]
    result = summarize(values, lower_better=[True, False])
    np.testing.assert_allclose(result["paired"]["gain"][1], [[1, 0.2], [1, 0.2]])
    np.testing.assert_allclose(result["paired"]["gain_ci_low"][1], result["paired"]["gain"][1])
    np.testing.assert_allclose(result["paired"]["gain_ci_high"][1], result["paired"]["gain"][1])


def test_leave_ads_and_family_out_are_descriptive_reaggregations():
    values = np.array([[[[5, 4]]], [[[5, 3]]], [[[5, 0]]], [[[5, 1]]]])
    result = summarize(values)
    assert result["paired"]["gain"][1] == 3
    assert result["leave_ads_out"]["ads-0"]["mean"][1] == pytest.approx(11 / 3)
    assert result["leave_family_out"]["rule"]["mean"][1] == 4.5
    assert result["leave_family_out"]["learning"]["mean"][1] == 1.5
    assert not result["design"]["leave_group_out_retrains_models"]


@pytest.mark.parametrize(
    "change", ("shape", "infinity", "duplicates", "missing_baseline", "empty", "replicates")
)
def test_invalid_units_or_inputs_do_not_produce_a_summary(change):
    values = np.zeros((2, 1, 2, 2))
    kwargs = {
        "ads_ids": ["a", "b"],
        "ads_families": ["rule", "learning"],
        "methods": ["target_only", "ours"],
        "replicates": 10,
    }
    if change == "shape":
        values = values[0]
    elif change == "infinity":
        values[0, 0, 0, 0] = np.inf
    elif change == "duplicates":
        kwargs["ads_ids"] = ["a", "a"]
    elif change == "missing_baseline":
        kwargs["methods"] = ["pooled", "ours"]
    elif change == "empty":
        values = values[:, :0]
    else:
        kwargs["replicates"] = 0
    with pytest.raises(ValueError):
        paired_summary(values, **kwargs)


def test_summary_is_deterministic_and_does_not_change_the_input():
    values = np.random.default_rng(53).normal(size=(8, 4, 5, 2, 6))
    before = copy.deepcopy(values)
    left, right = summarize(values), summarize(values)
    np.testing.assert_array_equal(left["paired"]["gain_ci_low"], right["paired"]["gain_ci_low"])
    np.testing.assert_array_equal(values, before)
