from dataclasses import replace

import pytest

from experiments.collision_metrics import (
    behavior_signature,
    equal_count_coverage,
    kaplan_meier,
    paired_lineage_bootstrap,
    summarize,
)
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements


def trajectory(collision):
    actors = [
        {"actor_id": "ego", "x": 0, "y": 4, "speed": 20, "heading": 0, "length": 5, "width": 2},
        {"actor_id": "lead", "x": 4, "y": 4, "speed": 0, "heading": 0, "length": 5, "width": 2},
    ]
    return [
        {
            "time_s": 0,
            "actors": actors,
            "collision": False,
            "collision_partner": None,
            "control": {"acceleration": 0, "steering": 0},
        },
        {
            "time_s": 1,
            "actors": actors,
            "collision": collision,
            "collision_partner": "lead" if collision else None,
            "control": {"acceleration": -8, "steering": 0},
        },
    ]


def test_fixed_quantization_and_actual_collision_counts_are_separate():
    a = sample("front_brake", 1, "development")[0]
    b = replace(a, scenario_id="duplicate_coordinate")
    scenes = [a, b]
    queries = [
        {
            "index": i,
            "feedback": QueryFeedback(s.scenario_id, True, SafetyMeasurements(0, 0), 2).to_dict(),
            "prequery_probability": 0.8,
        }
        for i, s in enumerate(scenes)
    ]
    result = summarize(scenes, queries, [trajectory(True), trajectory(True)])
    assert result["collision_executions"] == 2
    assert result["unique_collisions"] == 1
    assert result["discovery_curve"] == [1, 1]
    assert result["brier"] == pytest.approx(0.04)
    assert result["behavior_counts"]["collision_object"] == {"lead": 2}
    assert result["behavior_counts"]["ego_action"] == {"lane_keep/brake": 2}
    assert len(result["behavior_signatures"]) == 1
    assert result["equivalent_execution_s"] == 4
    reversed_queries = list(reversed(queries))
    assert summarize(scenes, reversed_queries, [trajectory(True)] * 2)["unique_collisions"] == 1
    assert behavior_signature(a, trajectory(True))[:3] == ("lead", "front", ("lane_keep", "brake"))
    queries[0]["feedback"]["collision"] = False
    with pytest.raises(ValueError, match="independently logged"):
        summarize(scenes, queries, [trajectory(True)] * 2)


def test_no_failure_is_censored_and_equal_counts_do_not_drop_it():
    scene = sample("front_brake", 1, "development")[0]
    query = {
        "index": 0,
        "feedback": QueryFeedback(
            scene.scenario_id, False, SafetyMeasurements(20, float("inf")), 1
        ).to_dict(),
    }
    result = summarize([scene], [query], [trajectory(False)])
    assert result["first_collision_query"] is None
    assert result["first_collision"] == {"time": 1, "observed": False}
    assert equal_count_coverage([result])["common_maximum"] == 0
    restored = QueryFeedback.from_dict(query["feedback"])
    assert restored.safety.min_ttc_s == float("inf")


def test_kaplan_meier_includes_censored_runs_in_risk_sets():
    result = kaplan_meier(
        [
            {"time": 2, "observed": True},
            {"time": 4, "observed": True},
            {"time": 4, "observed": False},
            {"time": 8, "observed": False},
        ]
    )
    assert [r["survival"] for r in result["curve"]] == [1, 0.75, 0.5, 0.5]
    assert result["restricted_mean_queries"] == 5.5
    assert result["censored_runs"] == 2


def test_restricted_mean_uses_one_window_without_extrapolating_unknown_follow_up():
    completed = [{"time": 2, "observed": True}, {"time": 4, "observed": True}]
    result = kaplan_meier(completed, restriction=128)
    assert result["restriction"] == result["curve"][-1]["time"] == 128
    assert result["restricted_mean_queries"] == 3
    assert result["curve"][-1]["survival"] == 0
    assert kaplan_meier(completed, restriction=3)["restricted_mean_queries"] == 2.5
    with pytest.raises(ValueError, match="Follow-up"):
        kaplan_meier([{"time": 4, "observed": False}], restriction=128)


def test_paired_statistics_count_lineages_instead_of_algorithm_seeds():
    records = []
    for lineage, difference in (("idm", 1), ("mpc", 2), ("ppo", 3)):
        for seed in range(5):
            for method, value in (("main", 10 + difference), ("base", 10)):
                records.append(
                    {
                        "lineage": lineage,
                        "target_id": lineage,
                        "scenario": "front_brake",
                        "condition": "new",
                        "environment_seed": 81001,
                        "algorithm_seed": seed,
                        "method": method,
                        "auc": value,
                    }
                )
    result = paired_lineage_bootstrap(records, "main", "base", "auc", repetitions=100)
    assert result["independent_lineages"] == 3
    assert result["paired_runs"] == 15
    assert result["difference"] == 2
    with pytest.raises(ValueError, match="counterpart"):
        paired_lineage_bootstrap(records[:-1], "main", "base", "auc")
