"""S01 full-pool contract, model, DFE and oracle isolation checks."""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from highway_sim_env.envs.fbrt_unified_env import FBRTUnifiedEnv, run_build_episode
from methods.failure_memory_regression.fm2_dfe import DiverseFailureExpansion
from methods.failure_memory_regression.fm2_features import ParameterSetEncoder, parameter_batch
from methods.failure_memory_regression.fm2_memory import build_memory
from methods.failure_memory_regression.fm2_model import FM2Model, support_row
from methods.failure_memory_regression.fm2_schema import SOURCES, TARGET, align_scenario, coordinates
from methods.failure_memory_regression.fm2_selector import run_fm2
from methods.failure_memory_regression import fm2_s01_pilot
from methods.failure_memory_regression.fm2_s01_pilot import (
    _exact_history_rank_scores, _failure_distance_replay)
from methods.failure_memory_regression.selector import _history_rank, run_selector
from methods.failure_memory_regression.prepare_scenario_sampling import compile_manifest
from methods.failure_memory_regression.selector import TargetOracle
from sut_algorithms.highway_env.registry import build_spec_factory


@pytest.fixture(scope="module")
def scenarios():
    # Small unit-test subset; the experiment itself uses the full 2,048-case manifest.
    return [row for row in compile_manifest() if row["catalogue_id"] == "S01"][:128]


def test_full_pool_configuration_and_bank_gates(scenarios, tmp_path, monkeypatch):
    config = fm2_s01_pilot._config()
    assert config["candidate_count"] == config["history_candidate_count"] == 2048
    assert tuple(config["sources"]) == SOURCES
    assert config["target_build"] == TARGET
    assert config["budget"] == 50
    assert len(config["training_seeds"]) == config["selector_repeats"] == 1

    pool = scenarios[:2]
    monkeypatch.setattr(fm2_s01_pilot, "ROOT", tmp_path)
    history_path = tmp_path / "history/source_response_bank.jsonl"
    history_path.parent.mkdir()
    history_rows = [{"build_id": build, "scenario_id": scene["scenario_id"]}
                    for build in SOURCES for scene in pool]
    history_path.write_text("\n".join(json.dumps(row) for row in history_rows), encoding="utf-8")
    assert len(fm2_s01_pilot._historical_bank(pool)) == len(SOURCES) * len(pool)
    history_path.write_text("\n".join(json.dumps(row) for row in history_rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="every historical build"):
        fm2_s01_pilot._historical_bank(pool)

    target_path = tmp_path / "target/full_response_bank.jsonl"
    target_path.parent.mkdir()
    target_rows = [{"build_id": TARGET, "scenario_id": scene["scenario_id"]} for scene in pool]
    target_path.write_text("\n".join(json.dumps(row) for row in target_rows), encoding="utf-8")
    assert len(fm2_s01_pilot._target_response_bank(pool)) == len(pool)
    target_path.write_text("\n".join(json.dumps(row) for row in target_rows[:1] * 2), encoding="utf-8")
    with pytest.raises(ValueError, match="target truth bank"):
        fm2_s01_pilot._target_response_bank(pool)


def _empty_memory():
    return {"values": np.empty((0, 4), np.float32), "present": np.empty((0, 4), bool),
            "spread": np.empty((0, 4), np.float32), "contrast": np.empty((0, 4), np.float32),
            "contrast_present": np.empty((0, 4), np.float32),
            "metadata": np.empty((0, 8), np.float32), "source": np.empty((0,), np.int64)}


def test_s01_four_dimensional_schema_and_historical_missing_mask(scenarios):
    aligned = align_scenario(scenarios[0])
    assert aligned.values.shape == (4,)
    assert aligned.present.tolist() == [True] * 4
    old = {"template_id": "fbrt_cutin", "initial_clearance_m": 20,
           "lane_change_duration_s": 2.0}
    mapped = align_scenario(old, historical=True)
    assert mapped.present.tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        align_scenario(old)


def test_parameter_order_and_padding_mask_invariance():
    torch.manual_seed(2)
    model = ParameterSetEncoder(dropout=0).eval()
    values = torch.tensor([[0.2, 0.5, 0.8, 0.1]])
    present = torch.ones_like(values, dtype=torch.bool)
    args = parameter_batch(values, present, torch.device("cpu"))
    with torch.no_grad():
        original = model(*args)
        perm = torch.tensor([3, 1, 0, 2])
        moved = model(*(arg[:, perm] for arg in args))
        assert torch.allclose(original, moved, atol=1e-5)
        padded = [torch.cat((arg, torch.zeros((1, 1), dtype=arg.dtype)), dim=1)
                  for arg in args]
        padded[1][:, -1] = False
        masked = model(*padded)
        assert torch.allclose(original, masked, atol=1e-5)


def test_memory_permutation_and_no_history_representation(scenarios):
    torch.manual_seed(3)
    model = FM2Model(dropout=0).eval()
    coords = coordinates(scenarios[:3])
    memory = _empty_memory()
    with torch.no_grad():
        baseline = model(coords, memory, [])
        assert torch.isfinite(baseline).all()
        memory = {"values": np.asarray([[.2, .3, .4, .5], [.5, .6, .7, .8]], np.float32),
                  "present": np.ones((2, 4), bool),
                  "spread": np.zeros((2, 4), np.float32),
                  "contrast": np.zeros((2, 4), np.float32),
                  "contrast_present": np.zeros((2, 4), np.float32),
                  "metadata": np.asarray([[1, 1, 0, 0, 0, 0, 1, 0],
                                           [2, 1, 0, 0, 0, 0, 0, 1]], np.float32),
                  "source": np.asarray([0, 1])}
        first = model(coords, memory, [])
        swapped = {key: value[::-1].copy() for key, value in memory.items()}
        second = model(coords, swapped, [])
        assert torch.allclose(first, second, atol=1e-5)


def test_support_after_query_and_frozen_history_residual(scenarios):
    outcome = {"ego_collision": True, "completed": False, "inconclusive": False,
               "min_ttc": 1.5, "min_clearance": -.2}
    item = support_row(coordinates(scenarios[:1])[0], outcome, .3)
    assert item["label"] == 1
    assert item["history_target_residual"] == pytest.approx(.7)
    assert support_row(item["coords"], {"inconclusive": True}, .3) is None
    torch.manual_seed(3)
    model = FM2Model(dropout=0).eval()
    with torch.no_grad():
        initial = model(coordinates(scenarios[:3]), _empty_memory(), [])
        updated = model(coordinates(scenarios[:3]), _empty_memory(), [item])
    assert not torch.allclose(initial, updated)


def test_patterncard_preserves_source_and_excludes_target(scenarios):
    def row(source, scene, label, suffix):
        return {"build_id": source, "scenario_id": scene["scenario_id"],
                "scenario": scene, "template_id": "fbrt_cutin", "execution_id": suffix,
                "ego_collision": bool(label), "completed": not bool(label),
                "inconclusive": False, "collision_partner_role": "lead" if label else None}
    records = [row("idm_ref", scenarios[0], 1, "fail"),
               row("idm_ref", scenarios[0], 0, "pass"),
               row(TARGET, scenarios[0], 1, "forbidden")]
    cards = build_memory(records, coordinates(scenarios), exclude=TARGET)
    assert cards and cards[0].source_build_id == "idm_ref"
    assert "fail" in cards[0].failure_record_ids
    assert "pass" in cards[0].pass_contrast_record_ids
    assert all("forbidden" not in card.failure_record_ids for card in cards)


def test_dfe_no_duplicates_direction_novelty_and_pass_barrier(scenarios):
    points = coordinates(scenarios)
    dfe = DiverseFailureExpansion(points)
    dfe.observe(0, 1, None)
    probabilities = np.full(len(points), .8)
    selected = dfe.propose(probabilities, {0})
    assert selected is None or selected[0] != 0
    if selected:
        candidate, origin = selected
        before = dfe.local_score(candidate, origin, .8)
        dfe.observe(candidate, 0, origin)
        assert dfe.local_score(candidate, origin, .8) <= before
        assert dfe.propose(probabilities, {0, candidate}) != (candidate, origin)
    assert all(len(dfe.representatives(component)) <= 4 for component in dfe.components())


def test_multiple_failure_components():
    points = np.asarray([[0, 0, 0, 0], [.05, .05, .05, .05],
                         [.95, .95, .95, .95], [1, 1, 1, 1]], dtype=float)
    dfe = DiverseFailureExpansion(points)
    dfe.observe(0, 1, None)
    dfe.observe(2, 1, None)
    assert len(dfe.components()) == 2


def test_pass_barrier_only_penalizes_local_score():
    points = np.asarray([[0, 0, 0, 0], [.1, .1, .1, .1],
                         [.2, .2, .2, .2], [.9, .9, .9, .9], [1, 1, 1, 1]], dtype=float)
    dfe = DiverseFailureExpansion(points)
    global_probability = np.full(len(points), .8)
    before = dfe.local_score(2, 0, global_probability[2])
    dfe.observe(1, 0, 0)
    after = dfe.local_score(2, 0, global_probability[2])
    assert after < before
    assert global_probability[2] == .8


def test_full_bank_not_selector_visible_and_exact_budget(scenarios):
    # Poisoning an unqueried bank row must not change any pre-query decision.
    coords = coordinates(scenarios[:8])
    pool = scenarios[:8]
    bank = {r["scenario_id"]: {"ego_collision": False, "completed": True,
                               "inconclusive": False} for r in pool}
    changed = deepcopy(bank)
    changed[pool[-1]["scenario_id"]] = {"ego_collision": True, "completed": False,
                                       "inconclusive": False}
    torch.manual_seed(4)
    model = FM2Model(dropout=0).eval()
    first, _, _ = run_fm2(model, pool, coords, _empty_memory(), [], TargetOracle(bank), budget=5)
    second, _, _ = run_fm2(model, pool, coords, _empty_memory(), [], TargetOracle(changed), budget=5)
    assert first[0]["scenario_id"] == second[0]["scenario_id"]
    assert len(first) == len({row["scenario_id"] for row in first}) == 5
    assert all(row["target_observations_before_query"] == row["rank"] - 1 for row in first)


def test_budget_exactly_fifty_and_manifest_shared(scenarios):
    pool = scenarios[:50]
    bank = {row["scenario_id"]: {"ego_collision": False, "completed": True,
                                 "inconclusive": False} for row in pool}
    torch.manual_seed(5)
    model = FM2Model(dropout=0).eval()
    oracle = TargetOracle(bank)
    queries, _, _ = run_fm2(model, pool, coordinates(pool), _empty_memory(), [],
                             oracle, budget=50)
    assert len(queries) == len(oracle.queried) == 50
    assert set(oracle.queried) == set(bank)


def test_full_source_history_rank_cache_matches_legacy_rule(scenarios):
    pool = scenarios[:4]
    builds = ("idm_ref", "merge_brake2", "slow_front_brake2",
              "nl3_v0", "nl3_v1", "nl3_v2")
    history = [{"build_id": build, "scenario_id": scene["scenario_id"],
                "scenario": scene, "template_id": "fbrt_cutin",
                "ego_collision": i == j % 4, "completed": True,
                "inconclusive": False, "min_ttc": 0.5 + i + j / 10,
                "min_clearance": 1.0 + i / 2}
               for j, build in enumerate(builds) for i, scene in enumerate(pool)]
    cached = _exact_history_rank_scores(pool, history)
    legacy = np.asarray([_history_rank(scene, history) for scene in pool])
    assert np.allclose(cached, legacy)


def test_failure_distance_adapter_matches_legacy_sequence(scenarios):
    pool = scenarios[:12]
    history = [{"build_id": build, "scenario_id": scene["scenario_id"],
                "scenario": scene, "template_id": "fbrt_cutin",
                "context_id": scene["context_id"],
                "execution_id": f"{build}:{i}",
                "ego_collision": i % 3 == j % 3, "completed": True,
                "inconclusive": False, "min_ttc": 0.5 + i,
                "min_clearance": 1.0 + i / 2}
               for j, build in enumerate(("idm_ref", "merge_brake2", "slow_front_brake2",
                                           "nl3_v0", "nl3_v1", "nl3_v2"))
               for i, scene in enumerate(pool)]
    bank = {scene["scenario_id"]: {"ego_collision": i % 4 == 0,
                                    "completed": True, "inconclusive": False}
            for i, scene in enumerate(pool)}
    legacy, _, _, _ = run_selector("FailureDistance-v2", pool, history,
                                    TargetOracle(bank), budget=12, random_seed=4179950,
                                    target_build_id=TARGET, mode="cross_agent")
    adapted = _failure_distance_replay(pool, history, TargetOracle(bank),
                                       seed=4179950, budget=12)
    assert [row["scenario_id"] for row in adapted] == [row["scenario_id"] for row in legacy]


@pytest.mark.parametrize("axis", ["initial_clearance_m", "lead_speed_mps",
                                  "lane_change_time_scale_s", "event_start_s"])
def test_s01_axis_extremes_affect_runner_and_event(axis, scenarios):
    case = scenarios[0]
    snapshots = []
    for value in case["research_bounds"][axis]:
        changed = deepcopy(case)
        changed["active_parameters"][axis] = value
        env = FBRTUnifiedEnv(build_spec_factory("idm_ref"), changed)
        try:
            env.reset(seed=changed["simulator_seed"])
            lead = env.actors["lead"]
            assert not env.vehicle.crashed and not lead.crashed
            snapshots.append((float(lead.position[0]), float(lead.speed),
                              lead.lane_change_duration_s, lead.event_start_s))
        finally:
            env.close()
        outcome, _ = run_build_episode("idm_ref", changed, changed["simulator_seed"])
        assert any(event.get("phase") == "LANE_CHANGE" for event in outcome["lead_events"])
        assert outcome["ego_collision"] is True or outcome["completed"] is True
    column = {"initial_clearance_m": 0, "lead_speed_mps": 1,
              "lane_change_time_scale_s": 2, "event_start_s": 3}[axis]
    assert snapshots[0][column] != snapshots[1][column]
