"""Direction, information-boundary, and release-lineage checks."""

from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from highway_sim_env.envs.fbrt_unified_env import run_build_episode

from methods.failure_memory_regression.bidirectional import (
    EXPLORATORY_METHODS, METHODS, ParentModel, TargetOracle, _regions, _schedule,
    _change_label, _change_prior_offset, _direction_conditioned_observations,
    _update_prequential_weights,
    compile_manifest, freeze, select, transition_truth,
)
from sut_algorithms.highway_env.nl_release import (
    NL2_V1, NL2_V2, NL3_V0, NL3_V1, NL3_V2, NL_EFFICIENCY_CANDIDATES,
    NL_PARENTS, NL_V0, NL_V1, NL_V2,
)
from sut_algorithms.highway_env.registry import build_spec_factory


def _outcome(build: str, case: dict, collision: bool | None) -> dict:
    return {"build_id": build, "scenario_id": case["scenario_id"],
            "scenario": case, "ego_collision": collision,
            "completed": collision is False, "inconclusive": collision is None}


def test_release_is_incremental_same_controller():
    assert NL_PARENTS == {"nl_v0": None, "nl_v1": "nl_v0", "nl_v2": "nl_v1"}
    assert {build_spec_factory(name).adapter_kind for name in NL_PARENTS} == {"legacy_profile"}
    assert {profile.controller for profile in (NL_V0, NL_V1, NL_V2)} == {"IDM"}
    for field in ("time_wanted", "desired_gap", "comfort_acceleration", "max_brake"):
        assert getattr(NL_V2, field) == getattr(NL_V1, field)
    assert asdict(NL_V0) != asdict(NL_V1) != asdict(NL_V2)
    assert build_spec_factory("nl2_v1").parent_build_id == "nl_v0"
    assert build_spec_factory("nl2_v2").parent_build_id == "nl2_v1"
    selected = asdict(NL_EFFICIENCY_CANDIDATES["nl_eff_c2"])
    v1 = asdict(NL2_V1)
    assert {key: value for key, value in v1.items() if key != "name"} == {
        key: value for key, value in selected.items() if key != "name"}
    for field in ("time_wanted", "desired_gap", "comfort_acceleration", "max_brake"):
        assert getattr(NL2_V2, field) == getattr(NL2_V1, field)
    assert NL3_V0.desired_gap > NL3_V1.desired_gap > 5.0
    assert NL3_V2.desired_gap == NL3_V1.desired_gap
    assert build_spec_factory("nl3_v1").parent_build_id == "nl3_v0"
    assert build_spec_factory("nl3_v2").parent_build_id == "nl3_v1"


def test_four_states_and_anomaly_are_evaluator_only():
    cases = compile_manifest("development", resolution=3)[:5]
    pairs = [(0, 1), (1, 0), (1, 1), (0, 0), (0, None)]
    bank = {}
    for case, (old, new) in zip(cases, pairs):
        bank["old", case["scenario_id"]] = _outcome("old", case, bool(old))
        bank["new", case["scenario_id"]] = _outcome(
            "new", case, None if new is None else bool(new))
    truth = transition_truth(cases, bank, "old", "new")
    assert [row["transition"] for row in truth] == [
        "REGRESSION", "IMPROVEMENT", "PERSISTENT_FAILURE", "STABLE_PASS", "UNKNOWN"]
    assert truth[1]["target_label"] == 0


def test_both_pools_and_unqueried_target_do_not_leak():
    cases = compile_manifest("development", resolution=3)[:8]
    parent = {case["scenario_id"]: _outcome("old", case, i % 2 == 1)
              for i, case in enumerate(cases)}
    target_a = {case["scenario_id"]: _outcome("new", case, i % 3 == 0)
                for i, case in enumerate(cases)}
    target_b = {sid: dict(row) for sid, row in target_a.items()}
    for row in target_b.values():
        row["ego_collision"] = not row["ego_collision"]
        row["completed"] = not row["ego_collision"]
    for method in ("directed_residual", "coordinate_context_ucb_pure",
                   "directed_context_ucb_pure", "directed_context_ucb_loose",
                   "coordinate_offset_calibrated_ucb",
                   "directed_offset_calibrated_ucb",
                   "coordinate_bootstrap_offset_ucb",
                   "directed_bootstrap_offset_ucb",
                   "direction_conditioned_offset_ucb",
                   "prequential_bma_ucb", "bidirectional_flip_ucb"):
        a = select(cases, parent, TargetOracle(target_a), method, budget=2)
        b = select(cases, parent, TargetOracle(target_b), method, budget=2)
        assert [row["direction"] for row in a] == ["R", "I"]
        assert a[0]["scenario_id"] == b[0]["scenario_id"]
        assert a[0]["score_before_query"] == b[0]["score_before_query"]
        # Restore the first feedback; every remaining target label may differ,
        # but the next decision must still be identical.
        target_b[a[0]["scenario_id"]] = target_a[a[0]["scenario_id"]]
        c = select(cases, parent, TargetOracle(target_b), method, budget=2)
        assert a[1]["scenario_id"] == c[1]["scenario_id"]
        assert a[1]["score_before_query"] == c[1]["score_before_query"]
    assert _schedule({"a": 0, "b": 1}, 4) == ["R", "I"]


def test_anomaly_is_charged_and_only_then_excluded_from_fit():
    cases = compile_manifest("development", resolution=3)[:4]
    parent = {case["scenario_id"]: _outcome("old", case, i % 2 == 1)
              for i, case in enumerate(cases)}
    target = {case["scenario_id"]: _outcome("new", case, None)
              for case in cases}
    oracle = TargetOracle(target)
    queries = select(cases, parent, oracle, "directed_residual", budget=4)
    assert len(oracle.queried) == len(queries) == 4
    assert all(not row["target_valid"] and row["episode_cost"] == 1 for row in queries)


def test_real_runner_uses_distinct_release_profiles_on_same_scene():
    case = compile_manifest("development", resolution=3)[4]
    outcomes = [run_build_episode(build, case, 4179931)[0]
                for build in ("nl_v0", "nl_v1", "nl_v2")]
    assert {row["scenario_fingerprint"] for row in outcomes} == {
        outcomes[0]["scenario_fingerprint"]}
    assert len({row["build_fingerprint"] for row in outcomes}) == 3
    assert {row["ego_runtime"]["vehicle_class"] for row in outcomes} == {
        "ProfiledIDMVehicle"}
    for field in ("initial_x_m", "initial_y_m", "initial_heading_rad", "initial_speed_mps"):
        assert len({row["ego_runtime"][field] for row in outcomes}) == 1
    assert all(row["episode_cost"] == 1 for row in outcomes)


def test_paired_physical_episode_is_reproducible():
    case = compile_manifest("development2", resolution=3)[2]
    first, trace_a = run_build_episode("nl2_v1", case, 4179943, with_trace=True)
    second, trace_b = run_build_episode("nl2_v1", case, 4179943, with_trace=True)
    assert first == second
    assert trace_a == trace_b


def test_region_coverage_uses_frozen_grid_adjacency():
    cases = compile_manifest("development", resolution=3)[:9]
    marked = {cases[index]["scenario_id"] for index in (0, 1, 8)}
    truth = [{"scenario_id": case["scenario_id"],
              "transition": "IMPROVEMENT" if case["scenario_id"] in marked else "STABLE_PASS"}
             for case in cases]
    assert sorted(map(len, _regions(cases, truth, "IMPROVEMENT"))) == [1, 2]


def test_context_laplace_features_do_not_share_unrelated_edge_slots():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: _outcome("old", case, case["grid_index"][0] > 0)
              for case in cases}
    model = ParentModel(cases, parent)
    first = model.features(cases[0], "directed_context_laplace")
    second = model.features(cases[9], "directed_context_laplace")
    assert len(first) == len(second) == 3 + 8 * 2
    assert any(first[3:11]) and not any(first[11:19])
    assert not any(second[3:11]) and any(second[11:19])


def test_laplace_next_query_ignores_unqueried_target_labels():
    cases = compile_manifest("development2", resolution=3)[:9]
    parent = {case["scenario_id"]: _outcome("old", case, i >= 4)
              for i, case in enumerate(cases)}
    target_a = {case["scenario_id"]: _outcome("new", case, i % 3 == 0)
                for i, case in enumerate(cases)}
    for method in ("directed_context_laplace", "directed_context_ucb",
                   "coordinate_context_ucb_pure", "directed_context_ucb_pure",
                   "directed_context_ucb_loose",
                   "coordinate_offset_calibrated_ucb",
                   "directed_offset_calibrated_ucb",
                   "coordinate_bootstrap_offset_ucb",
                   "directed_bootstrap_offset_ucb"):
        first_two = select(cases, parent, TargetOracle(target_a), method, budget=2)
        target_b = {sid: dict(row) for sid, row in target_a.items()}
        for sid, row in target_b.items():
            if sid != first_two[0]["scenario_id"]:
                row["ego_collision"] = not row["ego_collision"]
                row["completed"] = not row["ego_collision"]
        perturbed = select(cases, parent, TargetOracle(target_b), method, budget=2)
        assert [(row["scenario_id"], row["score_before_query"]) for row in first_two] == [
            (row["scenario_id"], row["score_before_query"]) for row in perturbed]


def test_new_chain_freeze_rejects_changed_protocol(tmp_path):
    builds = ("nl_v0", "nl2_v1", "nl2_v2")
    methods = METHODS + EXPLORATORY_METHODS
    assert len(freeze(tmp_path, "development2", 3, builds=builds, methods=methods)) == 27
    protocol = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["primary_method"] == "directed_residual"
    assert protocol["primary_baselines"] == ["static_risk", "coordinate_residual"]
    assert len(freeze(tmp_path, "development2", 3, builds=builds, methods=methods)) == 27
    with pytest.raises(ValueError, match="protocol differs"):
        freeze(tmp_path, "development2", 3, builds=builds, methods=METHODS)


def test_margin_coverage_reaches_each_parent_pass_context_and_follows_feedback():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {}
    target = {}
    for case in cases:
        sid = case["scenario_id"]
        parent[sid] = _outcome("old", case, False)
        parent[sid]["min_ttc"] = 1.0 + sum(case["grid_index"])
        target[sid] = _outcome("new", case, case["context_id"].endswith("c1")
                               and case["grid_index"] == [0, 0])
    for method in ("static_margin_coverage", "coordinate_margin_frontier",
                   "directed_margin_frontier"):
        queries = select(cases, parent, TargetOracle(target), method, budget=3)
        assert [row["context_id"] for row in queries[:2]] == [
            "S01:confirmation:c0", "S01:confirmation:c1"]
        assert queries[1]["discovery"] == 1
        if method != "static_margin_coverage":
            assert queries[2]["context_id"] == "S01:confirmation:c1"
            assert sum(cases_by_id["grid_index"] == [0, 0]
                       for cases_by_id in (next(case for case in cases
                                                if case["scenario_id"] == row["scenario_id"])
                                           for row in queries[:2])) == 2


def test_margin_frontier_cannot_read_unqueried_target_labels():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: {**_outcome("old", case, False),
                                     "min_ttc": 1.0 + sum(case["grid_index"])}
              for case in cases}
    target_a = {case["scenario_id"]: _outcome(
        "new", case, case["context_id"].endswith("c1")
        and case["grid_index"] == [0, 0])
                for case in cases}
    for method in ("coordinate_margin_frontier", "directed_margin_frontier"):
        first = select(cases, parent, TargetOracle(target_a), method, budget=3)
        target_b = {sid: dict(row) for sid, row in target_a.items()}
        for sid, row in target_b.items():
            if sid not in {first[0]["scenario_id"], first[1]["scenario_id"]}:
                row["ego_collision"] = True
                row["completed"] = False
        second = select(cases, parent, TargetOracle(target_b), method, budget=3)
        assert [(row["scenario_id"], row["score_before_query"]) for row in first] == [
            (row["scenario_id"], row["score_before_query"]) for row in second]


def test_dual_margin_probes_parent_fail_contexts_then_follows_improvement():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: {**_outcome("old", case, True),
                                     "collision_time_s": 1.0 + sum(case["grid_index"])}
              for case in cases}
    target = {case["scenario_id"]: _outcome(
        "new", case, not (case["context_id"].endswith("c1")
                         and case["grid_index"] == [0, 0])) for case in cases}
    for method in ("static_dual_margin_coverage", "coordinate_dual_margin_frontier",
                   "directed_dual_margin_frontier"):
        queries = select(cases, parent, TargetOracle(target), method, budget=3)
        assert [row["direction"] for row in queries] == ["I", "I", "I"]
        assert [row["context_id"] for row in queries[:2]] == [
            "S01:confirmation:c0", "S01:confirmation:c1"]
        assert queries[1]["discovery"] == 1
        assert queries[1]["selection_phase"] == "parent_failure_context_probe"
        if method != "static_dual_margin_coverage":
            assert queries[2]["context_id"] == "S01:confirmation:c1"
            assert queries[2]["selection_phase"] == "observed_improvement_frontier"


def test_two_probe_local_model_reaches_another_row_before_frontier():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: {**_outcome("old", case, True),
                                     "collision_time_s": 1.0 + sum(case["grid_index"])}
              for case in cases}
    target = {case["scenario_id"]: _outcome(
        "new", case, not (case["context_id"].endswith("c1")
                         and case["grid_index"] == [1, 0])) for case in cases}
    for method in ("static_dual_coverage2", "coordinate_dual_local",
                   "directed_dual_local"):
        queries = select(cases, parent, TargetOracle(target), method, budget=5)
        assert [row["context_id"] for row in queries[:4]] == [
            "S01:confirmation:c0", "S01:confirmation:c1",
            "S01:confirmation:c0", "S01:confirmation:c1"]
        assert [row["selection_phase"] for row in queries[:4]] == [
            "parent_failure_context_probe"] * 4
        assert queries[3]["discovery"] == 1
        if method != "static_dual_coverage2":
            assert queries[4]["context_id"] == "S01:confirmation:c1"
            assert queries[4]["selection_phase"] == "observed_improvement_frontier"


def test_pure_context_ucb_uses_only_posterior_ranking():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: _outcome("old", case, False) for case in cases}
    target = {case["scenario_id"]: _outcome(
        "new", case, not (case["context_id"].endswith("c1")
                         and case["grid_index"] == [1, 0])) for case in cases}
    for method in ("coordinate_context_ucb_pure", "directed_context_ucb_pure",
                   "directed_context_ucb_loose"):
        queries = select(cases, parent, TargetOracle(target), method, budget=4)
        assert len(queries) == 4
        assert all(row["direction"] == "R" for row in queries)
        assert all(row["selection_phase"] == "score_rank" for row in queries)
        assert all(row["score_before_query"] is not None for row in queries)


def test_offset_calibration_is_context_local_and_parent_only():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: _outcome(
        "old", case, case["grid_index"][0] > 0) for case in cases}
    model = ParentModel(cases, parent)
    case = cases[0]
    risk = model.risk_logit(case)

    coordinate = model.features(case, "coordinate_offset_calibrated_ucb")
    assert len(coordinate) == 3 + 5 * len(model.context_ids)
    assert coordinate[3 + 4] == risk
    assert not any(coordinate[3 + 5:])

    directed = model.features(case, "directed_offset_calibrated_ucb")
    assert len(directed) == 3 + 13 * len(model.context_ids)
    assert directed[3 + 12] == risk
    assert not any(directed[3 + 13:])

    # Flipping any unqueried target outcome cannot change the next query.
    target_a = {item["scenario_id"]: _outcome(
        "new", item, item["grid_index"][1] == 0) for item in cases}
    target_b = {sid: dict(row) for sid, row in target_a.items()}
    for row in target_b.values():
        row["ego_collision"] = not row["ego_collision"]
        row["completed"] = not row["ego_collision"]
    for method in ("coordinate_offset_calibrated_ucb",
                   "directed_offset_calibrated_ucb"):
        a = select(cases, parent, TargetOracle(target_a), method, budget=2)
        target_b[a[0]["scenario_id"]] = target_a[a[0]["scenario_id"]]
        b = select(cases, parent, TargetOracle(target_b), method, budget=2)
        assert [(row["scenario_id"], row["score_before_query"]) for row in a] == [
            (row["scenario_id"], row["score_before_query"]) for row in b]


def test_bootstrap_calibrated_ucb_covers_contexts_using_parent_data_only():
    cases = compile_manifest("confirmation", resolution=3)
    parent = {case["scenario_id"]: _outcome(
        "old", case, case["grid_index"][0] == 0) for case in cases}
    target_a = {case["scenario_id"]: _outcome(
        "new", case, case["grid_index"][1] == 0) for case in cases}
    contexts = {case["context_id"] for case in cases}
    for method in ("coordinate_bootstrap_offset_ucb",
                   "directed_bootstrap_offset_ucb"):
        queries = select(cases, parent, TargetOracle(target_a), method, budget=20)
        for direction in ("R", "I"):
            local = [row for row in queries if row["direction"] == direction]
            assert len(local) == 10
            assert all(row["selection_phase"] == "context_bootstrap_probe"
                       for row in local[:9])
            assert {row["context_id"] for row in local[:9]} == contexts
            assert local[9]["selection_phase"] == "score_rank"

        flipped = {sid: dict(row) for sid, row in target_a.items()}
        for row in flipped.values():
            row["ego_collision"] = not row["ego_collision"]
            row["completed"] = not row["ego_collision"]
        for query in queries:
            if query["rank"] < 20:
                flipped[query["scenario_id"]] = target_a[query["scenario_id"]]
        second = select(cases, parent, TargetOracle(flipped), method, budget=20)
        first = queries[:20]
        second = second[:20]
        assert [(row["scenario_id"], row["score_before_query"]) for row in first] == [
            (row["scenario_id"], row["score_before_query"]) for row in second]


def test_regression_only_bootstrap_preserves_improvement_posterior_ranking():
    cases = compile_manifest("confirmation", resolution=3)
    parent = {case["scenario_id"]: _outcome(
        "old", case, case["grid_index"][0] == 0) for case in cases}
    target = {case["scenario_id"]: _outcome(
        "new", case, case["grid_index"][1] == 0) for case in cases}
    contexts = {case["context_id"] for case in cases}
    for method in ("coordinate_regression_bootstrap_ucb",
                   "directed_regression_bootstrap_ucb"):
        queries = select(cases, parent, TargetOracle(target), method, budget=20)
        regressions = [row for row in queries if row["direction"] == "R"]
        improvements = [row for row in queries if row["direction"] == "I"]
        assert [row["selection_phase"] for row in regressions[:9]] == [
            "context_bootstrap_probe"] * 9
        assert {row["context_id"] for row in regressions[:9]} == contexts
        assert regressions[9]["selection_phase"] == "score_rank"
        assert all(row["selection_phase"] == "score_rank" for row in improvements)

        changed = {sid: dict(row) for sid, row in target.items()}
        for row in changed.values():
            row["ego_collision"] = not row["ego_collision"]
            row["completed"] = not row["ego_collision"]
        for row in queries:
            if row["rank"] < 20:
                changed[row["scenario_id"]] = target[row["scenario_id"]]
        replay = select(cases, parent, TargetOracle(changed), method, budget=20)
        assert [(row["scenario_id"], row["score_before_query"])
                for row in queries[:20]] == [
                    (row["scenario_id"], row["score_before_query"])
                for row in replay[:20]]


def test_direction_conditioned_model_excludes_opposite_parent_pool_feedback():
    cases = compile_manifest("confirmation", resolution=3)[:4]
    parent_labels = {case["scenario_id"]: index % 2
                     for index, case in enumerate(cases)}
    observations = [(case, index % 3 == 0)
                    for index, case in enumerate(cases)]

    regression = _direction_conditioned_observations(
        observations, parent_labels, "R")
    improvement = _direction_conditioned_observations(
        observations, parent_labels, "I")

    assert [case["scenario_id"] for case, _ in regression] == [
        cases[0]["scenario_id"], cases[2]["scenario_id"]]
    assert [case["scenario_id"] for case, _ in improvement] == [
        cases[1]["scenario_id"], cases[3]["scenario_id"]]
    with pytest.raises(ValueError):
        _direction_conditioned_observations(observations, parent_labels, "X")


def test_prequential_expert_weights_update_only_for_observed_direction():
    weights = {"R": [0.0, 0.0], "I": [0.0, 0.0]}
    _update_prequential_weights(weights, "R", [0.9, 0.1], 1)
    assert weights["R"][0] > weights["R"][1]
    assert weights["I"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        _update_prequential_weights(weights, "I", [0.5], 1)


def test_bidirectional_change_label_and_boundary_prior_are_symmetric():
    assert _change_label(1, 0) == 1
    assert _change_label(0, 1) == 1
    assert _change_label(1, 1) == 0
    assert _change_label(0, 0) == 0
    assert _change_prior_offset(-0.2) == _change_prior_offset(0.2)
    assert _change_prior_offset(0.2) > _change_prior_offset(1.0)


def test_contextual_confirmation_uses_new_physical_offsets():
    rows = compile_manifest("contextual_confirmation", resolution=3)
    core = compile_manifest("core_confirmation", resolution=3)
    offset = compile_manifest("offset_calibration_confirmation", resolution=3)
    bootstrap = compile_manifest("context_bootstrap_confirmation", resolution=3)
    regression_bootstrap = compile_manifest("regression_bootstrap_confirmation", resolution=3)
    superiority_replication = compile_manifest(
        "superiority_replication_confirmation", resolution=3)
    fields = ("ego_speed_mps", "lead_speed_mps", "event_start_s")

    def contexts(manifest):
        return {tuple(row["fixed_context"][field] for field in fields)
                for row in manifest}

    assert len({row["context_id"] for row in rows}) == 9
    assert not (contexts(rows) & contexts(core))
    assert len({row["context_id"] for row in offset}) == 9
    assert len({row["context_id"] for row in bootstrap}) == 9
    assert len({row["context_id"] for row in regression_bootstrap}) == 9
    assert len({row["context_id"] for row in superiority_replication}) == 9
    assert not (contexts(offset) & (contexts(rows) | contexts(core)))
    assert not (contexts(bootstrap) &
                (contexts(offset) | contexts(rows) | contexts(core)))
    assert not (contexts(regression_bootstrap) &
                (contexts(bootstrap) | contexts(offset) | contexts(rows) | contexts(core)))
    assert not (contexts(superiority_replication) &
                (contexts(regression_bootstrap) | contexts(bootstrap) |
                 contexts(offset) | contexts(rows) | contexts(core)))


def test_role_gated_generalization_confirmation_covers_fresh_supported_families():
    families = ("S01", "S02", "S03", "S04", "S05", "S06", "S08", "S09")
    rows = compile_manifest(
        "role_gated_generalization_confirmation", resolution=3, families=families)
    replication = compile_manifest(
        "superiority_replication_confirmation", resolution=3, families=families)

    def contexts(manifest):
        return {(row["catalogue_id"], json.dumps(
            row["fixed_context"], sort_keys=True, separators=(",", ":")))
                for row in manifest}

    assert len(rows) == len(families) * 3 * 3 * 3
    assert {row["catalogue_id"] for row in rows} == set(families)
    assert len({row["context_id"] for row in rows}) == len(families) * 3
    assert not (contexts(rows) & contexts(replication))


def test_role_gated_improvement_probes_use_parent_collision_partner_only():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: {**_outcome("old", case, True),
                                     "collision_time_s": 1.0 + sum(case["grid_index"]),
                                     "collision_partner_role": (
                                         "lead" if case["context_id"].endswith("c1")
                                         else "static")}
              for case in cases}
    target = {case["scenario_id"]: _outcome(
        "new", case, not (case["context_id"].endswith("c1")
                         and case["grid_index"] == [1, 0])) for case in cases}
    for method in ("static_role_coverage2", "coordinate_role_gated",
                   "directed_role_gated", "adaptive_role_hybrid"):
        queries = select(cases, parent, TargetOracle(target), method, budget=3)
        assert [row["context_id"] for row in queries[:2]] == [
            "S01:confirmation:c1", "S01:confirmation:c1"]
        assert all(row["selection_phase"] == "parent_failure_context_probe"
                   for row in queries[:2])
        assert queries[1]["discovery"] == 1
        if method != "static_role_coverage2":
            assert queries[2]["selection_phase"] == "observed_improvement_frontier"


def test_role_gated_frontier_ignores_unqueried_target_outcomes():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: {**_outcome("old", case, True),
                                     "collision_time_s": 1.0 + sum(case["grid_index"]),
                                     "collision_partner_role": (
                                         "lead" if case["context_id"].endswith("c1")
                                         else "static")}
              for case in cases}
    target_a = {case["scenario_id"]: _outcome(
        "new", case, not (case["context_id"].endswith("c1")
                         and case["grid_index"] == [1, 0])) for case in cases}
    for method in ("directed_role_gated", "adaptive_role_hybrid",
                   "multi_frontier_role", "coordinate_multi_frontier",
                   "directed_role_gated_edges"):
        first = select(cases, parent, TargetOracle(target_a), method, budget=3)
        target_b = {sid: dict(row) for sid, row in target_a.items()}
        for sid, row in target_b.items():
            if sid not in {first[0]["scenario_id"], first[1]["scenario_id"]}:
                row["ego_collision"] = not row["ego_collision"]
                row["completed"] = not row["ego_collision"]
        second = select(cases, parent, TargetOracle(target_b), method, budget=3)
        assert [(row["scenario_id"], row["score_before_query"]) for row in first] == [
            (row["scenario_id"], row["score_before_query"]) for row in second]


def test_adaptive_role_hybrid_uses_parent_margin_for_regressions():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: {**_outcome("old", case, False),
                                     "min_ttc": float(sum(case["grid_index"]))}
              for case in cases}
    target = {case["scenario_id"]: _outcome("new", case, False)
              for case in cases}
    hybrid = select(cases, parent, TargetOracle(target),
                    "adaptive_role_hybrid", budget=4)
    static = select(cases, parent, TargetOracle(target),
                    "static_role_coverage2", budget=4)
    assert [row["scenario_id"] for row in hybrid] == [
        row["scenario_id"] for row in static]
    assert all(row["direction"] == "R" and row["method"] == "adaptive_role_hybrid"
               for row in hybrid)


@pytest.mark.parametrize("family, expected_phase, expected_roles", (
    ("S03", "BRAKE_TO_STOP", {"ego", "lead"}),
    ("S04", "RESTART", {"ego", "lead"}),
    ("S09", "LANE_CHANGE", {"ego", "lead", "rear"}),
))
def test_additional_family_runner_templates(family, expected_phase, expected_roles):
    case = next(case for case in compile_manifest(
        "family_development", 3, families=(family,))
        if case["grid_index"] == [2, 2])
    outcome, _ = run_build_episode("nl_v0", case, case["simulator_seed"])
    assert set(outcome["actor_roles"]) == expected_roles
    assert expected_phase in {event["phase"] for event in outcome["lead_events"]}
    assert outcome["ego_collision"] or outcome["completed"]


def test_missing_parent_ttc_has_json_safe_audit_score():
    cases = compile_manifest("family_development", 3,
                             families=("S05",))
    parent = {case["scenario_id"]: {**_outcome("old", case, False),
                                     "min_ttc": None} for case in cases}
    target = {case["scenario_id"]: _outcome("new", case, False)
              for case in cases}
    queries = select(cases, parent, TargetOracle(target),
                     "static_role_coverage2", budget=2)
    assert queries[0]["score_before_query"] is None


def test_latest_parent_failure_probe_targets_scenes_closer_to_recovery():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: {**_outcome("old", case, True),
                                     "collision_time_s": 1.0 + sum(case["grid_index"]),
                                     "collision_partner_role": (
                                         "lead" if case["context_id"].endswith("c1")
                                         else "static")}
              for case in cases}
    target = {case["scenario_id"]: _outcome("new", case, False)
              for case in cases}
    for method in ("static_role_coverage_latest", "coordinate_role_gated_latest",
                   "directed_role_gated_latest"):
        queries = select(cases, parent, TargetOracle(target), method, budget=2)
        assert queries[0]["direction"] == "I"
        assert queries[0]["context_id"].endswith("c1")
        first = next(case for case in cases
                     if case["scenario_id"] == queries[0]["scenario_id"])
        assert first["grid_index"] == [2, 2]
        assert queries[0]["method"] == method


def test_span_probe_covers_both_ends_of_parent_collision_time():
    cases = compile_manifest("confirmation", resolution=3)[:18]
    parent = {case["scenario_id"]: {**_outcome("old", case, True),
                                     "collision_time_s": 1.0 + sum(case["grid_index"]),
                                     "collision_partner_role": (
                                         "lead" if case["context_id"].endswith("c1")
                                         else "static")}
              for case in cases}
    target = {case["scenario_id"]: _outcome("new", case, False)
              for case in cases}
    queries = select(cases, parent, TargetOracle(target),
                     "directed_role_gated_span", budget=2)
    selected = [next(case for case in cases
                     if case["scenario_id"] == query["scenario_id"])
                for query in queries]
    assert selected[0]["grid_index"] == [2, 2]
    assert selected[1]["grid_index"] == [0, 0]


def test_directed_role_gated_edges_retains_direction_in_majority_failure_context():
    cases = compile_manifest("confirmation", resolution=3)[:9]
    parent = {case["scenario_id"]: {
        **_outcome("old", case, case["grid_index"] != [1, 1]),
        "collision_partner_role": "lead"} for case in cases}
    model = ParentModel(cases, parent)
    case = cases[0]
    coordinate = model.features(case, "coordinate_role_gated")
    directed = model.features(case, "directed_role_gated_edges")
    assert model.failure_fraction[case["context_id"]] > .5
    assert model.lead_failure_fraction[case["context_id"]] == 1.0
    assert directed.size > coordinate.size
    assert directed[7:].any()
