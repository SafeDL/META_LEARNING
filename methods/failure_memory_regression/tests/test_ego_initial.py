"""The v3 ego state must reach both the physical case and the selector features."""

from pathlib import Path

import yaml

from methods.failure_memory_regression.archive import read_jsonl
from methods.failure_memory_regression.audit import CONFIG, _new_case
from methods.failure_memory_regression.pattern_memory import (
    active_values, build_dictionaries,
)
from methods.failure_memory_regression.selector import (
    TargetOracle, run_selector,
)


ROOT = Path("results/method_chains/failure_memory_regression/memory_v2")


def test_ego_initial_state_is_a_seven_dimensional_search_input():
    config = yaml.safe_load(CONFIG.read_text(
        encoding="utf-8"))
    old = next(case for case in read_jsonl(ROOT / "compact_bank" / "scenario_cases.jsonl")
               if case["catalogue_id"] == "S05" and case["scenario_id"].endswith(":0"))
    low = _new_case(old, config["initial_state_bounds"], "test_low",
                    {"ego_initial_speed_mps": 20.0})
    high = _new_case(old, config["initial_state_bounds"], "test_high",
                     {"ego_initial_speed_mps": 30.0})
    low_features, high_features = active_values(low), active_values(high)
    assert len(low_features) == len(high_features) == 7
    assert low_features[2] == 0.0 and high_features[2] == 1.0
    assert low_features[:2] == high_features[:2]

    parent = [{"build_id": "mobil_ref_v2", "template_id": case["template_id"],
               "scenario_id": case["scenario_id"], "scenario": case,
               "context_id": case["context_id"], "ego_collision": False,
               "completed": True, "inconclusive": False,
               "execution_id": "parent-" + case["audit_label"]}
              for case in (low, high)]
    dictionary = build_dictionaries(parent, [], [low, high])["fbrt_lane_change_rear"]
    assert dictionary.feature_dim == 7
    assert len(dictionary.features(low)) == 1 + 7 + len(dictionary.coverage_centers)

    oracle = TargetOracle({
        low["scenario_id"]: {"execution_id": "target-low", "ego_collision": True,
                             "completed": False, "inconclusive": False},
        high["scenario_id"]: {"execution_id": "target-high", "ego_collision": False,
                              "completed": True, "inconclusive": False},
    })
    queries, observations, _cards, updates = run_selector(
        "FBRT-Memory", [{**low, "parent_pass": True},
                        {**high, "parent_pass": True}], parent, oracle,
        2, 4179901, "mobil_rear_guard_off_v2", parent_build_id="mobil_ref_v2",
        mode="regression", session_id="ego-state-test")
    assert len(queries) == len(observations) == 2
    refits = [row for row in updates if row.get("event") == "posterior_refit"]
    assert len(refits) == 2
    assert all(row["feature_count"] >= 8 for row in refits)
