from __future__ import annotations

import inspect
import json
from collections import deque
from types import SimpleNamespace

import numpy as np

from highway_env.road.road import Road, RoadNetwork
from methods.failure_memory_regression.archive import (
    SnapshotStore, build_visibility_view, row_to_record,
)
from methods.failure_memory_regression.bayes_model import fit_logistic
from methods.failure_memory_regression.pattern_memory import (
    build_dictionaries, build_pattern_cards,
)
from methods.failure_memory_regression.schema import BuildSpec
from methods.failure_memory_regression.selector import TargetOracle, run_selector
from highway_sim_env.envs.fbrt_metrics import (
    longitudinal_bumper_clearance, time_to_collision,
)
from highway_sim_env.envs.fbrt_unified_env import FBRTUnifiedEnv
from sut_algorithms.highway_env.regression_builds import make_native_vehicle
from sut_algorithms.highway_env.fbrt_adapters import adapter_for
from sut_algorithms.highway_env.registry import build_spec_factory


def _record(execution_id, build_id, collision, clearance=20.0, *, scenario_id=None,
            context="ctx-a", completed=True):
    scenario_id = scenario_id or execution_id
    scenario = {"scenario_id": scenario_id, "template_id": "fbrt_cutin",
                "initial_clearance_m": clearance, "lane_change_duration_s": 2.0,
                "context_id": context}
    return {"execution_id": execution_id, "scenario_id": scenario_id,
            "build_id": build_id, "family": build_id, "template_id": "fbrt_cutin",
            "context_id": context, "scenario": scenario, "completed": completed,
            "ego_collision": collision, "inconclusive": False, "visibility": "historical",
            "min_ttc": 3.0 if collision else 8.0,
            "min_clearance": 1.0 if collision else 10.0, "episode_cost": 0}


def _candidate(index, clearance=20.0, context="ctx-a"):
    scenario_id = f"candidate-{index}"
    return {"scenario_id": scenario_id, "template_id": "fbrt_cutin",
            "scenario": {"scenario_id": scenario_id, "template_id": "fbrt_cutin",
                         "initial_clearance_m": clearance,
                         "lane_change_duration_s": 2.0, "context_id": context},
            "context_id": context, "parent_pass": True, "completed": True,
            "ego_collision": False}


def test_scenario_id_does_not_change_feature_or_fixed_seed_selection():
    records = [_record("f1", "source", True, 12.0), _record("p1", "source", False, 30.0)]
    cards = build_pattern_cards(records)
    candidates = [_candidate(0, 12.0), _candidate(1, 30.0), _candidate(2, 50.0)]
    dictionaries = build_dictionaries(records, cards, candidates, seed=81)
    x1 = dictionaries["fbrt_cutin"].features(candidates[0]["scenario"])
    changed = {**candidates[0], "scenario": {**candidates[0]["scenario"],
                                               "scenario_id": "renamed-candidate"},
               "scenario_id": "renamed-candidate"}
    x2 = dictionaries["fbrt_cutin"].features(changed["scenario"])
    np.testing.assert_allclose(x1, x2)

    bank1 = {item["scenario_id"]: {"ego_collision": False, "completed": True,
                                    "inconclusive": False} for item in candidates}
    renamed_candidates = [{**item, "scenario_id": f"renamed-{i}",
                           "scenario": {**item["scenario"], "scenario_id": f"renamed-{i}"}}
                          for i, item in enumerate(candidates)]
    bank2 = {item["scenario_id"]: {"ego_collision": False, "completed": True,
                                    "inconclusive": False} for item in renamed_candidates}
    q1, *_ = run_selector("Random", candidates, [], TargetOracle(bank1), 1, 7,
                             "target", mode="cross_agent")
    q2, *_ = run_selector("Random", renamed_candidates, [], TargetOracle(bank2), 1, 7,
                             "target", mode="cross_agent")
    assert candidates[int(q1[0]["scenario_id"].split("-")[-1])]["scenario"][
        "initial_clearance_m"] == renamed_candidates[int(q2[0]["scenario_id"].split("-")[-1])][
        "scenario"]["initial_clearance_m"]


def test_boundary_pass_contrasts_stay_with_the_same_build():
    records = [_record("source-fail", "source-a", True, 20.0),
               _record("other-pass", "source-b", False, 21.0)]
    cards = build_pattern_cards(records)
    assert cards
    assert all("other-pass" not in card.pass_contrast_record_ids for card in cards)
    assert all(all(a.startswith("source-fail") for a, _ in card.boundary_edges)
               for card in cards)


def test_all_safe_history_keeps_a_finite_model_and_can_update():
    records = [_record(f"pass-{i}", "safe-build", False, 15.0 + i) for i in range(5)]
    cards = build_pattern_cards(records)
    dictionary = build_dictionaries(records, cards, records)["fbrt_cutin"]
    x = np.vstack([dictionary.features(row["scenario"]) for row in records])
    fit = fit_logistic(x, np.zeros(len(records)))
    assert np.all(np.isfinite(fit.mean))
    assert np.all(np.isfinite(fit.covariance))
    updated = fit_logistic(x, np.asarray([0, 0, 0, 0, 1]), fit.mean,
                           np.diag(fit.covariance))
    assert np.all(np.isfinite(updated.mean))


def test_new_target_failure_creates_a_local_pattern_without_history():
    candidates = [_candidate(i, 15.0 + i) for i in range(3)]
    bank = {item["scenario_id"]: {"ego_collision": True, "completed": False,
                                  "inconclusive": False, "episode_cost": 0}
            for item in candidates}
    queries, observations, cards, updates = run_selector(
        "FBRT-Memory", candidates, [], TargetOracle(bank), 1, 1,
        "target", session_id="synthetic-session")
    assert len(observations) == 1
    assert any(card.created_in_session == "synthetic-session" for card in cards)
    assert any(row.get("feature_added") for row in updates)
    assert queries[0]["new_pattern_id"]


def test_duplicate_execution_id_is_not_committed_twice_and_snapshot_reloads(tmp_path):
    store = SnapshotStore(tmp_path)
    record = _record("observed-failure", "agent-a", True)
    before, after, inserted = store.commit([record, record])
    assert before != after
    assert inserted == ["observed-failure"]
    loaded = store.load_records()
    assert len(loaded) == 1
    assert loaded[0]["execution_id"] == "observed-failure"
    assert json.loads((tmp_path / "history_snapshots" / "latest.json").read_text())[
        "snapshot_hash"] == after


def test_snapshot_memory_changes_predictions_on_nearby_candidate():
    candidate = _candidate(1, clearance=20.0)
    remote_pass = _record("pass", "agent-a", False, 55.0)
    close_fail = _record("fail", "agent-a", True, 20.0)
    candidates = [candidate]
    cards_before = build_pattern_cards([remote_pass])
    dictionary_before = build_dictionaries([remote_pass], cards_before, candidates)["fbrt_cutin"]
    from methods.failure_memory_regression.bayes_model import (
        build_source_prior, posterior_failure_probabilities, source_fits, target_posterior,
    )
    prior_fit = source_fits([remote_pass], dictionary_before)
    mu0, var0, _ = build_source_prior(prior_fit, {"agent-a": "agent-a"})
    before = target_posterior(dictionary_before, [], mu0, var0)
    before_probability = posterior_failure_probabilities(
        dictionary_before, [candidate], before, seed=1)[0]
    cards_after = build_pattern_cards([remote_pass, close_fail])
    dictionary_after = build_dictionaries([remote_pass, close_fail], cards_after, candidates)[
        "fbrt_cutin"]
    fits_after = source_fits([remote_pass, close_fail], dictionary_after)
    mu1, var1, _ = build_source_prior(fits_after, {"agent-a": "agent-a"})
    after = target_posterior(dictionary_after, [], mu1, var1)
    after_probability = posterior_failure_probabilities(
        dictionary_after, [candidate], after, seed=1)[0]
    assert not np.isclose(before_probability, after_probability)


def test_regression_history_view_hides_evaluator_and_non_parent_rows():
    rows = [_record("parent", "parent", False),
            {**_record("evaluator", "target", True), "visibility": "evaluator_only"},
            _record("other", "other", True)]
    view = build_visibility_view(rows, target_build_id="target", parent_build_id="parent",
                                 regression=True)
    assert [row["execution_id"] for row in view] == ["parent"]


def test_imported_target_labels_stay_evaluator_only_in_contextual_history():
    source = {"build": "target", "scenario_id": "candidate-0", "seed": "1",
              "scenario": json.dumps(_candidate(0)["scenario"]),
              "completed": "True", "ego_collision": "True", "semantic_valid": "True"}
    target = row_to_record(source, "target_response_bank.csv").as_dict()
    reference = row_to_record({**source, "build": "parent"},
                              "reference_archive.csv").as_dict()
    assert target["visibility"] == "evaluator_only"
    assert reference["visibility"] == "historical"
    from methods.failure_memory_regression.experiment import _contextual_history
    history = [_record("parent", "parent", False),
               {**_record("hidden", "target", True), "visibility": "evaluator_only"}]
    assert [row["execution_id"] for row in _contextual_history(history, [_candidate(0)])] == [
        "parent"]


def test_cross_agent_seed_excludes_earlier_offline_replay_commits():
    from methods.failure_memory_regression.experiment import _cross_agent_seed_records
    historical = _record("legacy", "idm_ref", False)
    prior_replay = _record("replayed", "mobil_ref_v2", False)
    hidden = {**_record("hidden", "target", True), "visibility": "evaluator_only"}
    seed = _cross_agent_seed_records(
        [historical, prior_replay, hidden], [{"execution_id": "replayed"}])
    assert [row["execution_id"] for row in seed] == ["legacy"]


def test_cross_agent_summary_counts_collisions_without_a_parent_regression():
    from methods.failure_memory_regression.experiment import _summaries
    queries = [{"rank": 1, "mode": "cross_agent", "ego_collision": True,
                "inconclusive": False, "regression": False},
               {"rank": 2, "mode": "cross_agent", "ego_collision": False,
                "inconclusive": False, "regression": False}]
    rows = _summaries("cross_agent", "ppo_ref_v2", "FBRT-Memory", 0, queries, 1, 2)
    assert all(row["failure_count"] == 1 for row in rows)
    assert all(row["first_failure_rank"] == 1 for row in rows)
    assert all(row["failure_recall"] == 1.0 for row in rows)


def test_only_one_high_level_action_is_applied_per_external_decision_tick():
    class Vehicle:
        def __init__(self):
            self.calls = 0
            self.crashed = False

        def act(self):
            self.calls += 1

    class ActionType:
        actions = {0: "LANE_LEFT", 1: "IDLE"}

        def __init__(self):
            self.calls = 0

        def act(self, action):
            self.calls += 1
            assert action == 0

    class Adapter:
        def act(self, _env, _observation):
            return 0

    class Road:
        def step(self, _dt):
            pass

    env = object.__new__(FBRTUnifiedEnv)
    env.spec = BuildSpec("ppo_ref_v2", "ppo_ece", None, "external_meta_policy",
                         "PPO-ECE", 5.0)
    env.scenario = {"template_id": "fbrt_moving_lead"}
    env.adapter = Adapter()
    env.vehicle = Vehicle()
    env.actors = {"ego": env.vehicle}
    env.action_type = ActionType()
    env.road = Road()
    env.steps = 0
    env.time = 0.0
    env.observation_type = type("Obs", (), {"observe": lambda _self: np.zeros((5, 6))})()
    env.observation_history = deque([(0.0, np.zeros((5, 6)))], maxlen=64)
    env.initial_observation = np.zeros((5, 6))
    env.control_actions = []
    env._record_step = lambda: None
    env._advance()
    assert env.action_type.calls == 1
    assert env.vehicle.calls == 0
    env.steps = 1
    env._advance()
    assert env.action_type.calls == 1
    assert env.vehicle.calls == 1


def test_mobil_mutation_is_limited_to_the_rear_guard_switch():
    road = Road(network=RoadNetwork.straight_road_network(2), np_random=np.random.default_rng(1))
    lane = road.network.get_lane(("0", "1", 0))
    reference = make_native_vehicle("mobil_ref_v2", road, lane.position(30, 0),
                                    heading=lane.heading_at(30), speed=20,
                                    target_lane_index=("0", "1", 0), target_speed=25)
    mutated = make_native_vehicle("mobil_rear_guard_off_v2", road, lane.position(60, 0),
                                  heading=lane.heading_at(60), speed=20,
                                  target_lane_index=("0", "1", 0), target_speed=25)
    assert type(reference) is type(mutated)
    assert reference.rear_guard_enabled is True
    assert mutated.rear_guard_enabled is False
    assert "guard_would_reject and self.rear_guard_enabled" in inspect.getsource(
        type(reference).mobil)


def test_rear_state_age_vehicle_uses_registered_mutation():
    road = Road(network=RoadNetwork.straight_road_network(2), np_random=np.random.default_rng(1))
    lane = road.network.get_lane(("0", "1", 0))
    env = SimpleNamespace(road=road, ego_lane_index=("0", "1", 0))
    for build_id in ("mobil_ref_v2", "mobil_rear_state_age",
                     "mobil_rear_state_age080"):
        spec = build_spec_factory(build_id)
        vehicle = adapter_for(spec).create_vehicle(env, lane, 30.0, 20.0, 25.0)
        assert vehicle.rear_state_age_s == (spec.mutation or {}).get(
            "rear_state_age_s", 0.0)
        assert vehicle.rear_guard_enabled is True


def test_physical_geometry_uses_bumper_clearance_and_closing_speed_units():
    gap = longitudinal_bumper_clearance(30.0, 5.0, 10.0, 5.0)
    assert gap == 15.0
    assert time_to_collision(gap, 5.0) == 3.0
    assert time_to_collision(gap, 0.0) is None


def test_snapshot_accounting_is_unique_by_execution_id(tmp_path):
    store = SnapshotStore(tmp_path)
    rows = [_record("same-id", "agent-a", True), _record("same-id", "agent-a", True)]
    store.commit(rows)
    assert len(store.load_records()) == 1
