"""Checks for the independent six-input interaction contract."""

import json
from pathlib import Path

import numpy as np
import pytest

from highway_sim_env.envs.fbrt_unified_env import run_build_episode
from methods.failure_memory_regression.bayes_model import (
    posterior_failure_probabilities, target_posterior,
)
from methods.failure_memory_regression.catalogue import (
    INTERACTION_CATALOGUE, INTERACTION_HOLDOUT_CATALOGUE,
    _sample_parameters, compile_interaction_scenarios, load_catalogue,
)
from methods.failure_memory_regression import interaction
from methods.failure_memory_regression.prepare_holdout import (
    prepare_holdout,
)
from methods.failure_memory_regression.validate_holdout import (
    analyze, evaluate_seed, run_evaluation,
)
from methods.failure_memory_regression.pattern_memory import (
    active_values, build_dictionaries, semantic_key,
)
from methods.failure_memory_regression.selector import TargetOracle, run_selector


def test_compiler_keeps_legacy_two_dimensional_stream_and_compiles_six_inputs(tmp_path):
    legacy = _sample_parameters(load_catalogue()["scenarios"][0], 4179901)
    assert len(legacy) == 16
    assert legacy[0]["active_parameters"] == {"initial_clearance_m": 21.0,
                                                "lane_change_time_scale_s": 1.875}
    cases = compile_interaction_scenarios(tmp_path)
    assert len(cases) == 128
    assert {case["template_id"] for case in cases} == {
        "fbrt_interaction_front_rear", "fbrt_interaction_cutin_escape"}
    assert all(len(case["active_parameters"]) == 6 for case in cases)
    assert all(len(active_values(case)) == 6 for case in cases)


def test_entire_frozen_legacy_compilation_is_unchanged():
    root = Path("results/method_chains/failure_memory_regression/memory_v2")
    frozen = root / "compact_bank" / "scenario_cases.jsonl"
    if not frozen.is_file():
        pytest.skip("frozen compact bank is unavailable in this checkout")
    recipe = json.loads((root / "selected_recipe.json").read_text(encoding="utf-8"))
    selected = set(recipe["selected_scenario_ids"])
    expected = [case for item in load_catalogue()["scenarios"]
                if item["id"] in selected
                for case in _sample_parameters(item, 4179901)]
    recorded = [json.loads(line) for line in frozen.read_text(encoding="utf-8").splitlines()]
    assert recorded == expected


def test_fixed_v2_development_contract_is_separate_from_v1(tmp_path):
    v1 = compile_interaction_scenarios(
        tmp_path / "v1", 4179901,
        INTERACTION_HOLDOUT_CATALOGUE)
    v2 = compile_interaction_scenarios(
        tmp_path / "v2", 4179902,
        INTERACTION_CATALOGUE)
    assert len(v2) == len(v1) == 128
    assert not ({case["scenario_id"] for case in v1} &
                {case["scenario_id"] for case in v2})
    assert {case["context_id"] for case in v2} == {
        "IA:interaction_v2:fixed_before_execution",
        "IB:interaction_v2:fixed_before_execution"}


def test_proposed_holdout_freezes_all_eight_seeds_without_episodes(tmp_path):
    root = tmp_path / "age080"
    protocol = prepare_holdout(root)
    assert protocol["planned_case_build_episodes"] == 3072
    assert len(protocol["candidate_sha256_by_seed"]) == 8
    assert protocol["target_outcome_filtering"] is False
    assert protocol["build_fingerprints"]["mobil_rear_state_age080"] != (
        protocol["build_fingerprints"]["mobil_ref_v2"])
    assert prepare_holdout(root) == protocol
    assert not (root / "seed4179910" / "measured").exists()


def test_prospective_gate_uses_physical_seed_clusters_not_selector_repeats():
    from methods.failure_memory_regression.prepare_holdout import SEEDS
    from methods.failure_memory_regression.selector import METHODS

    rows = [{"simulator_seed": seed, "repeat": repeat, "method": method,
             "selector_seed": seed + repeat, "target_failure_pool": 4,
             "hits_at_20": 2 if method == "FBRT-Memory" else 1}
            for seed in SEEDS for repeat in range(10) for method in METHODS]
    result = analyze(rows)
    assert result["sign_flip_p_two_sided"] == 1 / 128
    assert result["memory_improvement_gate"] is True
    rows[0]["selector_seed"] += 1
    with pytest.raises(ValueError, match="share a selector seed"):
        analyze(rows)


def test_prospective_selector_repeats_share_seed_and_isolate_target(tmp_path):
    cases = compile_interaction_scenarios(tmp_path)[:2]
    banks = {}
    for build_id in ("mobil_ref_v2", "mobil_rear_guard_off_v2",
                     "mobil_rear_state_age080"):
        banks[build_id] = {
            case["scenario_id"]: {
                "scenario_id": case["scenario_id"], "scenario": case,
                "build_id": build_id, "completed": build_id != "mobil_rear_state_age080",
                "ego_collision": build_id == "mobil_rear_state_age080",
                "inconclusive": False, "min_ttc": 2.0, "min_clearance": 4.0,
            } for case in cases}
    protocol = {"schema": "test-holdout", "candidate_sha256_by_seed": {"4179910": "fixed"}}
    summaries, runs, pool = evaluate_seed(cases, banks, protocol, 4179910,
                                          budget=1, repeats=2)
    assert pool == 2
    assert len(summaries) == len(runs) == 10
    for repeat in range(2):
        assert len({row["selector_seed"] for row in summaries
                    if row["repeat"] == repeat}) == 1
    assert all(row["eligible_candidates"] == 2 for row in summaries)


def test_prospective_evaluation_fails_closed_without_physical_banks(tmp_path):
    root = tmp_path / "unmeasured"
    with pytest.raises(ValueError, match="incomplete or duplicate measured bank"):
        run_evaluation(root)
    assert not (root / "evaluation").exists()


def test_interaction_events_use_measured_timing_and_four_actor_interface(tmp_path):
    cases = compile_interaction_scenarios(tmp_path)
    a, b = cases[0], cases[64]
    a_result, a_trace = run_build_episode("mobil_ref_v2", a, 4179901, with_trace=True)
    b_result, _ = run_build_episode("mobil_ref_v2", b, 4179901)
    assert set(a_result["actor_roles"]) == {"ego", "lead", "rear", "adjacent_front"}
    assert len(a_trace[0]) == 5  # time plus four actors
    rear_events = [event for event in a_result["public_signature"]["background_phases"]
                   if event["phase"] == "TARGET_SPEED_INCREASE"]
    expected = 3.0 + a["active_parameters"]["rear_event_offset_s"]
    assert rear_events and abs(rear_events[0]["time_s"] - expected) <= 0.05
    merge = b_result["event_times"]["measured_merge_complete_s"]
    brake = [event for event in b_result["lead_events"]
             if event["phase"] == "BRAKE_AFTER_MERGE"]
    assert brake and brake[0]["time_s"] >= merge + b["active_parameters"][
        "brake_after_measured_merge_s"] - 1e-6


def test_collision_diagnostics_use_actual_partner_and_phase(tmp_path):
    cases = {case["scenario_id"]: case for case in compile_interaction_scenarios(
        tmp_path, catalogue_path=INTERACTION_HOLDOUT_CATALOGUE)}
    side, _ = run_build_episode(
        "mobil_rear_guard_off_v2", cases["4179901:IB:space_filling:49"], 4179901)
    front, _ = run_build_episode(
        "ppo_ref_v2", cases["4179901:IB:space_filling:44"], 4179901)
    assert (side["ego_collision"], side["collision_partner_role"],
            side["collision_type"]) == (True, "rear", "lane_change_side_rear")
    assert (front["ego_collision"], front["collision_partner_role"],
            front["collision_type"]) == (True, "lead", "current_lane_front")


def test_interaction_dictionary_marks_missing_source_margin(tmp_path):
    case = compile_interaction_scenarios(tmp_path)[0]
    dictionary = build_dictionaries([], [], [case])[case["template_id"]]
    features = dictionary.features(case)
    ids = dictionary.ordered_feature_ids()
    assert len(features) == len(ids)
    assert features[ids.index("relation:historical_margin_missing")] == 1.0
    altered = {**case, "active_parameters": {**case["active_parameters"],
                                              "rear_event_offset_s": 0.75}}
    assert not np.array_equal(features, dictionary.features(altered))
    assert len(semantic_key(case)) == 3
    assert semantic_key({**case, "observed_maneuver_phase": "during_ego_lane_change",
                         "collision_partner_role": "rear"})[-2:] == (
                             "during_ego_lane_change", "rear")


def test_pass_only_history_uses_target_only_branch(tmp_path):
    cases = compile_interaction_scenarios(tmp_path)[:10]
    history = [{"scenario_id": case["scenario_id"], "execution_id": f"source-{index}",
                "build_id": "mobil_ref_v2", "template_id": case["template_id"],
                "context_id": case["context_id"], "scenario": case,
                "completed": True, "ego_collision": False, "inconclusive": False,
                "min_ttc": 2.0, "visibility": "historical"}
               for index, case in enumerate(cases[:3])]
    bank = {case["scenario_id"]: {"completed": index not in (1, 6),
                                  "ego_collision": index in (1, 6),
                                  "inconclusive": False} for index, case in enumerate(cases)}
    memory = run_selector("FBRT-Memory", cases, history, TargetOracle(bank), 10, 23,
                             target_build_id="mobil_rear_state_age")
    no_memory = run_selector("FBRT-NoMemory", cases, history, TargetOracle(bank), 10, 23,
                                target_build_id="mobil_rear_state_age")
    assert [row["scenario_id"] for row in memory[0]] == [
        row["scenario_id"] for row in no_memory[0]]
    assert all(row["history_source_count"] == 0 and
               row["history_weight_before_query"] == 0.0 and
               row["historical_margin_prediction_before_query"][
                   "historical_margin_missing"] == 1.0 for row in memory[0])


def test_interaction_evaluation_excludes_colliding_parent_even_if_completed(
        tmp_path, monkeypatch):
    cases = compile_interaction_scenarios(tmp_path / "cases")[:2]
    parent_rows = [
        {"scenario_id": case["scenario_id"], "scenario": case,
         "completed": True, "ego_collision": index == 0, "inconclusive": False}
        for index, case in enumerate(cases)
    ]
    target_rows = [
        {"scenario_id": case["scenario_id"], "scenario": case,
         "completed": False, "ego_collision": True, "inconclusive": False}
        for case in cases
    ]
    monkeypatch.setattr(interaction, "compile_interaction_scenarios",
                        lambda *_args: cases)
    monkeypatch.setattr(interaction, "read_jsonl",
                        lambda path: parent_rows if path.stem == "mobil_ref_v2"
                        else target_rows if path.stem == "mobil_rear_state_age" else [])
    results = interaction.evaluate(
        tmp_path / "evaluation", "mobil_rear_state_age", "mobil_ref_v2", 1)
    assert all(result["queries"][0]["scenario_id"] == cases[1]["scenario_id"]
               for result in results)


def test_pass_only_parent_does_not_displace_failure_source(tmp_path):
    cases = compile_interaction_scenarios(tmp_path)[:3]
    history = []
    for build_id, failed in (("mobil_ref_v2", False),
                             ("mobil_rear_guard_off_v2", True)):
        case = cases[0]
        history.append({"scenario_id": case["scenario_id"],
                        "execution_id": build_id + "-case", "build_id": build_id,
                        "template_id": case["template_id"], "context_id": case["context_id"],
                        "scenario": case, "completed": not failed,
                        "ego_collision": failed, "inconclusive": False,
                        "min_ttc": 0.5 if failed else 4.0,
                        "visibility": "historical"})
    candidates = [{**case, "parent_pass": True} for case in cases]
    bank = {case["scenario_id"]: {"completed": True, "ego_collision": False,
                                  "inconclusive": False} for case in cases}
    queries, _, _, updates = run_selector(
        "FBRT-Memory", candidates, history, TargetOracle(bank), 1, 41,
        target_build_id="mobil_rear_state_age", parent_build_id="mobil_ref_v2",
        mode="regression")
    assert queries[0]["history_source_count"] == 1
    refit = next(row for row in updates if row["event"] == "posterior_refit")
    assert refit["source_build_ids"] == ["mobil_rear_guard_off_v2"]
    chosen = next(case for case in candidates if case["scenario_id"] == queries[0]["scenario_id"])
    dictionary = build_dictionaries([], [], candidates, seed=7319)[chosen["template_id"]]
    feature_ids = tuple(dictionary.ordered_feature_ids())
    fit = target_posterior(dictionary, [], feature_ids, np.zeros(len(feature_ids)),
                           np.full(len(feature_ids), 4.0))
    expected = posterior_failure_probabilities(
        dictionary, [chosen], fit,
        seed=41 + 1 + sum(chosen["template_id"].encode("utf-8")), samples=32)[0]
    weight_update = next(row for row in updates
                         if row["event"] == "prequery_likelihood_weight")
    assert np.isclose(weight_update["target_probability_before"], expected)
