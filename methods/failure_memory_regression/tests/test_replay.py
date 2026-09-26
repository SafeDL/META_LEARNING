from __future__ import annotations

import hashlib
import sys

import numpy as np
import pytest

from methods.failure_memory_regression.bayes_model import (
    align_diagonal_prior, build_source_prior, source_fits, target_posterior,
)
from methods.failure_memory_regression.pattern_memory import (
    build_dictionaries, build_pattern_cards, new_failure_card,
)
from methods.failure_memory_regression.replay_utils import (
    contextual_history, is_parent_pass, is_usable_outcome, task_reward,
)
from methods.failure_memory_regression.selector import (
    TargetOracle, run_selector,
)
from methods.failure_memory_regression.replay import _summaries
from methods.failure_memory_regression.replay import _empirical_effect
from methods.failure_memory_regression.replay import _regression_seed


def _scenario(scenario_id: str, x: float, y: float = 0.5) -> dict:
    return {
        "scenario_id": scenario_id,
        "template_id": "fbrt_cutin",
        "context_id": "test-context",
        "parameterization_version": "research_v2",
        "active_parameters": {"initial_clearance_m": 8.0 + 52.0 * x,
                               "lane_change_time_scale_s": 1.5 + 1.5 * y},
        "research_bounds": {"initial_clearance_m": [8.0, 60.0],
                            "lane_change_time_scale_s": [1.5, 3.0]},
        "fixed_context": {"duration_s": 12, "ego_speed_mps": 25,
                          "event_start_s": 1, "lane_count": 2, "lead_speed_mps": 20},
    }


def _record(execution_id: str, collision: bool, x: float,
            **extra) -> dict:
    scenario = _scenario(execution_id, x)
    return {"execution_id": execution_id, "scenario_id": execution_id,
            "build_id": extra.pop("build_id", "source-a"),
            "template_id": "fbrt_cutin", "context_id": "test-context",
            "scenario": scenario, "completed": not collision,
            "ego_collision": collision, "inconclusive": False,
            "visibility": "historical", **extra}


def _candidate(sid: str, x: float, parent_pass: bool = True) -> dict:
    return {"scenario_id": sid, "template_id": "fbrt_cutin",
            "context_id": "test-context", "scenario": _scenario(sid, x),
            "parent_pass": parent_pass, "completed": parent_pass,
            "ego_collision": not parent_pass}


def test_failure_history_and_parent_candidate_predicates_are_separate():
    passed = _record("pass", False, 0.15, completed=True)
    failure = _record("collision-stop", True, 0.17, completed=False,
                      collision_partner_role="lead")
    incomplete = _record("background", False, 0.2, completed=False,
                         inconclusive=False)
    evaluator = {**_record("target", True, 0.2), "visibility": "evaluator_only"}
    assert is_usable_outcome(passed)
    assert is_usable_outcome(failure)
    assert not is_usable_outcome(incomplete)
    assert not is_parent_pass(failure)
    assert is_parent_pass(passed)
    assert contextual_history([passed, failure, incomplete, evaluator],
                              [_candidate("c", 0.3)]) == [passed, failure, incomplete]
    cards = build_pattern_cards([passed, failure, incomplete, evaluator])
    assert len(cards) == 1
    assert cards[0].failure_record_ids == ["collision-stop"]
    assert cards[0].observed_partner_roles == ["lead"]
    assert build_pattern_cards([]) == []
    evaluator_target = {**evaluator, "build_id": "target-build"}
    dictionary = build_dictionaries([passed], build_pattern_cards([passed]),
                                    [_candidate("c", 0.3)])['fbrt_cutin']
    assert set(source_fits([passed, evaluator_target], dictionary)) == {"source-a"}


def test_paired_regression_seeds_are_common_across_methods_and_repeats():
    task = "fixed-task"
    assert _regression_seed(task, "FBRT-Memory", 0, 10) == _regression_seed(
        task, "FBRT-NoMemory", 0, 10)
    assert _regression_seed(task, "FBRT-Memory", 0, 10) != _regression_seed(
        task, "FBRT-Memory", 1, 10)
    assert _regression_seed(task, "FBRT-Memory", 0, 1) != _regression_seed(
        task, "FBRT-NoMemory", 0, 1)


def test_feature_ids_keep_dynamic_failure_centers_from_shifting_priors():
    history = [_record("old-fail", True, 0.1)]
    candidates = [_candidate("c0", 0.1), _candidate("c1", 0.5),
                  _candidate("c2", 0.9)]
    cards = build_pattern_cards(history)
    dictionary = build_dictionaries(history, cards, candidates)["fbrt_cutin"]
    old_ids = tuple(dictionary.ordered_feature_ids())
    old_mean = np.arange(1, len(old_ids) + 1, dtype=float) * 1.7
    old_variance = np.arange(1, len(old_ids) + 1, dtype=float) * 2.3
    first, added = new_failure_card(_record("new-fail-1", True, 0.55),
                                    {"fbrt_cutin": dictionary}, "session")
    assert added
    second, added = new_failure_card(_record("new-fail-2", True, 0.95),
                                     {"fbrt_cutin": dictionary}, "session")
    assert added
    new_ids = tuple(dictionary.ordered_feature_ids())
    aligned_mean, aligned_variance = align_diagonal_prior(
        old_ids, old_mean, old_variance, new_ids)
    for feature_id, mean, variance in zip(old_ids, old_mean, old_variance):
        index = new_ids.index(feature_id)
        assert aligned_mean[index] == mean
        assert aligned_variance[index] == variance
    assert aligned_mean[new_ids.index(f"failure:{first.pattern_id}")] == 0.0
    assert aligned_variance[new_ids.index(f"failure:{first.pattern_id}")] == 4.0
    assert aligned_mean[new_ids.index(f"failure:{second.pattern_id}")] == 0.0
    assert aligned_variance[new_ids.index(f"failure:{second.pattern_id}")] == 4.0
    assert dictionary.ordered_feature_ids()[:3] == [
        "bias", "coord:initial_clearance_m", "coord:lane_change_time_scale_s"]

    permutation = list(reversed(range(len(new_ids))))
    theta = np.arange(len(new_ids), dtype=float)
    z = dictionary.features(candidates[1]["scenario"])
    assert np.isclose(z @ theta, z[permutation] @ theta[permutation])


def test_prior_alignment_rejects_ambiguous_or_invalid_schemas():
    with pytest.raises(ValueError, match="unique"):
        align_diagonal_prior(["a", "a"], np.ones(2), np.ones(2), ["a"])
    with pytest.raises(ValueError, match="cannot disappear"):
        align_diagonal_prior(["a", "b"], np.ones(2), np.ones(2), ["a"])
    with pytest.raises(ValueError, match="do not match"):
        align_diagonal_prior(["a"], np.ones(2), np.ones(1), ["a"])
    with pytest.raises(ValueError, match="positive"):
        align_diagonal_prior(["a"], np.ones(1), np.zeros(1), ["a"])


def test_empty_history_prior_matches_full_feature_schema():
    dictionary = build_dictionaries([], [], [_candidate("c", 0.3)])[
        "fbrt_cutin"]
    mean, variance, used = build_source_prior({}, {}, feature_ids=dictionary.ordered_feature_ids())
    assert used == []
    assert mean.shape == variance.shape == (len(dictionary.ordered_feature_ids()),)
    assert np.all(mean == 0.0)
    assert np.all(variance == 4.0)


def test_repeated_posterior_refits_use_the_same_observations_once(monkeypatch):
    candidate = _candidate("c", 0.3)
    dictionary = build_dictionaries([], [], [candidate])["fbrt_cutin"]
    ids = tuple(dictionary.ordered_feature_ids())
    mean, variance, _ = build_source_prior({}, {}, feature_ids=ids)
    observed = _record("target-observation", True, 0.4, template_id="fbrt_cutin")
    captured = []

    class Fit:
        mean = np.zeros(len(ids))
        covariance = np.eye(len(ids))

    def fake_fit(features, labels, prior_mean=None, prior_variance=None,
                 max_iter=25, **kwargs):
        captured.append((features.copy(), labels.copy(), prior_mean.copy(), prior_variance.copy()))
        return Fit()

    monkeypatch.setattr("methods.failure_memory_regression.bayes_model.fit_logistic", fake_fit)
    target_posterior(dictionary, [observed], ids, mean, variance,
                     source_feature_specs=dictionary.ordered_feature_specs(),
                     source_schema_identity=dictionary.schema_identity)
    target_posterior(dictionary, [observed], ids, mean, variance,
                     source_feature_specs=dictionary.ordered_feature_specs(),
                     source_schema_identity=dictionary.schema_identity)
    assert len(captured) == 2
    assert all(len(labels) == 1 for _features, labels, _mean, _var in captured)
    assert np.array_equal(captured[0][2], captured[1][2])
    assert np.array_equal(captured[0][3], captured[1][3])


def test_source_prior_schema_rejects_reused_id_with_changed_semantics():
    history = [_record("old-fail", True, 0.1)]
    candidates = [_candidate("c0", 0.1), _candidate("c1", 0.7)]
    dictionary = build_dictionaries(history, build_pattern_cards(history), candidates)["fbrt_cutin"]
    ids = tuple(dictionary.ordered_feature_ids())
    mean, variance, _ = build_source_prior({}, {}, feature_ids=ids)
    specs = dictionary.ordered_feature_specs()
    dictionary.centers[0]["center"] = [0.8, 0.8]
    with pytest.raises(ValueError, match="semantics changed"):
        target_posterior(dictionary, [], ids, mean, variance,
                         source_feature_specs=specs,
                         source_schema_identity=dictionary.schema_identity)


def test_task_reward_and_cross_agent_ucb_use_valid_collision():
    collision = {"ego_collision": True, "inconclusive": False}
    uncertain = {"ego_collision": True, "inconclusive": True}
    assert task_reward("cross_agent", collision) == 1
    assert task_reward("regression", collision, parent_pass=True) == 1
    assert task_reward("regression", collision, parent_pass=False) == 0
    assert task_reward("cross_agent", uncertain) == 0
    with pytest.raises(ValueError, match="explicit parent-pass"):
        task_reward("regression", collision)

    candidates = [_candidate("a", 0.2),
                  {**_candidate("b", 0.8), "template_id": "fbrt_moving_lead",
                   "scenario": {**_scenario("b", 0.8), "template_id": "fbrt_moving_lead"}}]
    bank = {sid: {"scenario_id": sid, "ego_collision": True, "completed": False,
                  "inconclusive": False, "collision_partner": "lead"}
            for sid in ("a", "b")}
    queries, _observations, _cards, _updates = run_selector(
        "HistoryRank-UCB-v2", candidates, [], TargetOracle(bank), 1, 7,
        target_build_id="agent-x", mode="cross_agent", initial_cards=[])
    assert queries[0]["valid_collision"] is True
    assert queries[0]["regression"] is None
    assert queries[0]["selection_reward"] == 1
    assert queries[0]["ucb_count_after"] == 1
    assert queries[0]["ucb_reward_after"] == 1

    uncertain_bank = {sid: {"scenario_id": sid, "ego_collision": True,
                            "completed": False, "inconclusive": True}
                      for sid in ("a", "b")}
    uncertain_queries, observations, *_ = run_selector(
        "HistoryRank-UCB-v2", candidates, [], TargetOracle(uncertain_bank), 1, 8,
        target_build_id="agent-x", mode="cross_agent", initial_cards=[])
    assert uncertain_queries[0]["selection_reward"] == 0
    assert uncertain_queries[0]["ucb_count_after"] == 1
    assert uncertain_queries[0]["ucb_reward_after"] == 0
    assert observations == []


def test_regression_reward_requires_parent_pass_and_observed_fields_survive():
    candidate = _candidate("candidate", 0.4, parent_pass=True)
    outcome = {"scenario_id": "candidate", "ego_collision": True, "completed": False,
               "inconclusive": False, "collision_partner": "static",
               "collision_time_s": 2.25, "event_times": {"first_exit_s": 0.0},
               "public_signature": {"event": "observed"},
               "execution_contract_version": "contract-v2", "trajectory_path": "trace.jsonl"}
    queries, observations, cards, _updates = run_selector(
        "FBRT-Memory", [candidate], [], TargetOracle({"candidate": outcome}), 1, 9,
        target_build_id="fault", parent_build_id="parent", mode="regression",
        initial_cards=[])
    assert queries[0]["regression"] is True
    assert queries[0]["selection_reward"] == 1
    assert observations[0]["collision_partner_role"] == "static"
    assert observations[0]["collision_time_s"] == 2.25
    assert observations[0]["public_signature"] == {"event": "observed"}
    assert observations[0]["event_times"] == {"first_exit_s": 0.0}
    assert observations[0]["execution_contract_version"] == "contract-v2"
    assert observations[0]["trajectory_path"] == "trace.jsonl"
    assert cards[-1].observed_partner_roles == ["static"]

    no_parent = {key: value for key, value in candidate.items() if key != "parent_pass"}
    with pytest.raises(ValueError, match="explicit parent-pass"):
        run_selector("HistoryRank-UCB-v2", [no_parent], [],
                        TargetOracle({"candidate": outcome}), 1, 10,
                        target_build_id="fault", mode="regression")


def test_zero_target_failures_keep_null_recall():
    candidate = _candidate("c", 0.3)
    queries, *_ = run_selector(
        "Random", [candidate], [], TargetOracle({"c": {
            "scenario_id": "c", "ego_collision": False, "completed": True,
            "inconclusive": False}}), 1, 3,
        target_build_id="target", mode="regression", initial_cards=[])
    rows = _summaries("no-regression", "target", "Random", 0,
                      queries, target_failures=0, candidate_count=1,
                      history_rows=[], initial_patterns=0)
    assert all(row["failure_recall"] is None for row in rows)
    assert all("NO_TARGET_FAILURE_IN_POOL" in row["status"] for row in rows)


def test_cross_agent_memory_comparison_groups_methods_by_shared_task():
    rows = [
        {"task_id": "cross_agent_FBRT-Memory_ppo_ref_after_mobil",
         "method": "FBRT-Memory", "repeat": 0, "budget": 20,
         "failure_count": 13, "failure_pool_count": 19},
        {"task_id": "cross_agent_FBRT-NoMemory_ppo_ref_after_mobil",
         "method": "FBRT-NoMemory", "repeat": 0, "budget": 20,
         "failure_count": 17, "failure_pool_count": 19},
        {"task_id": "cross_agent_Random_ppo_ref_after_mobil",
         "method": "Random", "repeat": 0, "budget": 20,
         "failure_count": 4, "failure_pool_count": 19},
        {"task_id": "cross_agent_Random_ppo_ref_after_mobil",
         "method": "Random", "repeat": 1, "budget": 20,
         "failure_count": 6, "failure_pool_count": 19},
    ]
    effect, comparisons = _empirical_effect(rows)
    assert effect == "no_gain"
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["delta_vs_no_memory"] == -4.0
    assert comparison["baseline_failure_count_means"]["Random"] == 5.0


def test_fake_bank_replay_stays_offline_and_does_not_mutate_inputs(tmp_path):
    from methods.failure_memory_regression import replay

    input_file = tmp_path / "frozen_bank.jsonl"
    input_file.write_text('{"id":1}\n', encoding="utf-8")
    before = hashlib.sha256(input_file.read_bytes()).hexdigest()
    candidates = [_candidate("c", 0.4)]
    bank = {"c": {"scenario_id": "c", "ego_collision": False,
                  "completed": True, "inconclusive": False}}
    imported_before = set(sys.modules)
    task = replay._run_selector_task(
        tmp_path / "out", "fake", "target", candidates, [], bank,
        "cross_agent", None, lambda _method, repeat: 11 + repeat)
    assert task["logical_query_count"] > 0
    assert "methods.failure_memory_regression.experiment" not in (
        set(sys.modules) - imported_before)
    assert hashlib.sha256(input_file.read_bytes()).hexdigest() == before
