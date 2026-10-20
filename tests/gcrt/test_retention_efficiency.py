import copy

import pytest

from methods.gcrt.controller import threshold
from methods.gcrt.efficiency_report import (
    MCTS, discovery_metrics, held_out_calibration, summarize as discovery_summary,
)
from methods.gcrt.retention_efficiency import summarize, task_row


def example_trace():
    return {
        "sut": "test", "pool": 0, "positives": 300,
        "found": list(range(1, 201)), "selected_indices": list(range(200)),
        "execution_s": [2.0] * 200, "state_recording_s": [3.0] * 200,
        "timing": {"task_initialization_s": 1.0, "query_compute_s": [1.0] * 200},
    }


def example_prediction():
    return {
        "prequery_s": [0.5] * 200, "correction_s": [0.2] * 200,
        "forecast": [
            {"query": t, "score": 0.0, "deterministic": t == 190, "compute_s": 0.1}
            for t in range(10, 200, 5)
        ],
    }


def test_matched_costs_include_terminal_check_and_exclude_unused_suffix():
    trace, prediction = example_trace(), example_prediction()
    fixed = task_row("cutin", "full_budget", trace, None, 1.0, 4.0)
    stopped = task_row("cutin", "gcrt", trace, prediction, 1.0, 4.0)
    assert fixed["total_s"] == 605.0
    assert fixed["stopping_overhead_s"] == 0
    assert stopped["queries"] == 190
    assert stopped["retention"] == 0.95 and stopped["target_met"]
    assert stopped["state_construction_s"] == 117
    assert stopped["terminal_decision_s"] == 0.1
    assert stopped["stopping_decision_s"] == pytest.approx(3.7)
    assert stopped["algorithm_s"] == pytest.approx(448.7)
    assert stopped["total_s"] == pytest.approx(828.7)
    assert stopped["total_saved_s"] == pytest.approx(-223.7)
    assert stopped["selected_indices"] == fixed["selected_indices"][:190]
    prediction["prequery_s"][190:] = [1000.0] * 10
    prediction["correction_s"][190:] = [1000.0] * 10
    prediction["forecast"][-1]["compute_s"] = 1000.0
    trace["execution_s"][190:] = [1000.0] * 10
    changed = task_row("cutin", "gcrt", trace, prediction, 1.0, 4.0)
    assert changed["total_s"] == stopped["total_s"]
    assert changed["terminal_decision_s"] == stopped["terminal_decision_s"]


def test_zero_discoveries_excluded_from_retention_denominators():
    trace = example_trace()
    positive = task_row("cutin", "full_budget", trace, None, 1.0, 4.0)
    trace["found"] = [0] * 200
    zero = task_row("cutin", "full_budget", trace, None, 1.0, 4.0)
    assert zero["retention"] is None and zero["target_met"] is None
    aggregate = summarize([positive, zero])
    assert aggregate["tasks"] == 2 and aggregate["valid_tasks"] == 1
    assert aggregate["zero_discovery_tasks"] == 1
    assert aggregate["target_rate"] == aggregate["pooled_retention"] == 1.0
    assert aggregate["total_s"] == positive["total_s"] + zero["total_s"]
    empty = summarize([zero])
    assert empty["minimum_retention"] is None
    assert empty["target_rate"] is None and empty["pooled_retention"] is None


def test_per_task_target_cannot_be_replaced_by_pooled_retention():
    row = task_row("cutin", "full_budget", example_trace(), None, 1.0, 0.0)
    failed = copy.deepcopy(row)
    failed.update(found=0, full_found=1, retention=0.0, target_met=False)
    stats = summarize([row, failed])
    assert stats["target_rate"] == 0.5
    assert stats["minimum_retention"] == 0.0
    assert stats["pooled_retention"] > 0.95
    assert not stats["all_targets_met"]
    assert threshold([], "empirical") == 0.0
    assert threshold([], "strict") == 1.0


def test_risk_and_collision_coverage_use_complete_candidate_set():
    metrics = discovery_metrics([0.7, 0.2, 0.8, 0.1], [True, False, False, False], [2])
    assert metrics["danger_found"] == 1 and metrics["danger_total"] == 2
    assert metrics["danger_coverage"] == 0.5
    assert metrics["collision_found"] == 0 and metrics["collision_total"] == 1
    assert metrics["collision_coverage"] == 0.0
    selected = discovery_metrics([0.7, 0.2, 0.8, 0.1], [True, False, False, False], [0, 2])
    assert selected["danger_coverage"] == selected["collision_coverage"] == 1.0


def test_collision_free_configurations_keep_actual_cost_but_not_undefined_coverage():
    rows = []
    for collision in ([True, False], [False, False]):
        rows.append({"family": "cutin", "method": "gcrt",
                     **discovery_metrics([0.9, 0.1], collision, [0]),
                     "queries": 1, "simulation_s": 2.0, "algorithm_s": 1.0, "total_s": 3.0})
    stats = discovery_summary(rows)
    assert stats["configurations"] == 2 and stats["collision_valid_configurations"] == 1
    assert stats["collision_coverage_mean"] == stats["collision_coverage_pooled"] == 1.0
    assert stats["queries_sum"] == 2 and stats["total_s_sum"] == 6.0
    assert discovery_metrics([0.1], [False], [0])["danger_coverage"] is None


def test_supplementary_controller_is_excluded_from_calibration():
    def replay(sut, score, family="brake", method="gcrt"):
        return {"family": family, "method": method, "sut": sut, "pool": 0,
                "full_found": 100, "forecast": [{"query": 10, "found": 10, "score": score}]}

    records = [replay(MCTS, 0.99), replay("ppo_ref_v2", 0.7), replay("idm_ref", 0.8),
               replay("vi_ttc_ref_audit_v4", 1.0), replay("ppo_ref_v2", 0.98, family="cutin"),
               replay("ppo_ref_v2", 0.97, method="without_correction")]
    calibration = held_out_calibration(records, "brake", MCTS)
    assert calibration["threshold"] == 0.8
    assert set(calibration["calibration_controllers"]) == {"ppo_ref_v2", "idm_ref"}
    records[0]["full_found"] = 1000
    records[0]["forecast"][0]["score"] = 1.0
    assert held_out_calibration(records, "brake", MCTS) == calibration


def test_new_threshold_uses_first_strict_trigger_and_prefix_costs():
    trace, prediction = example_trace(), example_prediction()
    for check in prediction["forecast"]:
        check["score"] = 0.8 if check["query"] < 45 else 0.81
    row = task_row("brake", "gcrt", trace, prediction, 0.8, 4.0)
    assert row["queries"] == 45
    assert row["selected_indices"] == list(range(45))
    assert row["simulation_s"] == 90.0
    assert row["terminal_decision_s"] == 0.1
    prediction["forecast"][-1]["score"] = 1.0
    prediction["prequery_s"][45:] = [1000.0] * 155
    assert task_row("brake", "gcrt", trace, prediction, 0.8, 4.0)["total_s"] == row["total_s"]
