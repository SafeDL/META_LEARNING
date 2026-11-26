"""Tests for fixed-suite priors, paired metrics and evaluator-owned stops."""

import numpy as np
import pytest
import torch

from experiments.regression.fixed_replay import FixedReplayLedger, fixed_selector
from experiments.regression.fixed_report import score
from experiments.formal_scenarios import BRAKING_KIND
from experiments.query import run_isolated
from experiments.regression.protocol import sample_block
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario
from experiments.transfer_settings import native_settings
from simulator.highway.repair_env import RepairEnv
from testing_methods.baselines.birchler_greedy.method import prioritize
from testing_methods.baselines.spectre import method as spectre
from testing_methods.history_transfer.adaptation import adapted_online
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


def test_spectre_public_fitness_and_indicator_direction():
    features = np.array([[.1, 0], [.35, 1], [.8, 0], [1., 1]])
    values = np.array([[.9, .2, 1, .8], [.4, .9, 0, .6], [.8, .5, 1, .3], [.1, .1, 0, .1]])
    orders = np.array([[0, 1, 2], [2, 1, 0], [1, 2, 3]])
    objectives = spectre.fitness(orders, values, features)
    for row, order in zip(objectives, orders):
        position = sum((4 - i) / 3 * values[j, 3] for i, j in enumerate(order)) / 3
        pairs = [abs(features[a] - features[b]).mean() for i, a in enumerate(order) for b in order[i+1:]]
        np.testing.assert_allclose(row, [*values[order, :3].sum(axis=0),
                                         round(np.mean(pairs), 3), round(position, 3)])
    assert objectives[0, -1] > objectives[1, -1]
    keep, _ = spectre.survive(objectives[:2], 1)
    assert keep.tolist() == [0]


@pytest.mark.parametrize("kind,params,changed", [
    (BRAKING_KIND, dict(ego_speed_mps=27, closing_speed_mps=6, gap_m=36, deceleration_mps2=6),
     {"gap_m": 37}),
    ("cut_in_onset", dict(ego_speed_mps=27, closing_speed_mps=6, gap_m=25, lateral_duration_s=2),
     {"lateral_duration_s": 3}),
    ("cut_out_rear_gap", dict(ego_speed_mps=29, obstacle_distance_m=130, rear_gap_m=40),
     {"obstacle_distance_m": 131}),
    ("side_encroachment", dict(ego_speed_mps=27, longitudinal_offset_m=-4, lateral_duration_s=2),
     {"longitudinal_offset_m": -5}),
])
def test_spectre_inclusive_demand_boundaries(kind, params, changed):
    assert spectre.demand_count(Scenario("a", kind, params)) == len(params)
    assert spectre.demand_count(Scenario("b", kind, {**params, **changed})) == len(params) - 1


def test_spectre_estimates_from_prefix_and_searches_a_unique_reproducible_suffix(monkeypatch):
    from types import SimpleNamespace

    coords = set()
    scenes = sample_block("cut_in_onset", 35, 41, coords, "test")
    old = sample_block("cut_in_onset", 12, 73, coords, "history")
    archive = SimpleNamespace(scenarios=old, collision=np.zeros(len(old), dtype=bool),
                              continuous_margin=np.full(len(old), .1))
    initial = [(2, QueryFeedback(scenes[2].scenario_id, True, SafetyMeasurements(0, 0, rss_margin=-.7), 0))]
    features = np.stack([spectre.parameter_features(s) for s in scenes])
    values = spectre.attributes(scenes, [archive], initial, features)
    assert np.allclose(values[2, :2], [1, 1])
    demand = spectre.demand_count(scenes[2])
    diversity = np.abs(features - features[2]).mean(axis=1).sum() / (len(scenes) - 1)
    assert values[2, 2] == (demand >= 3)
    assert values[2, 3] == pytest.approx((1 + .7 + demand + diversity) / 4)
    for key, value in (("block_size", 10), ("population", 10), ("evaluations_per_search", 100)):
        monkeypatch.setitem(spectre.SETTINGS, key, value)
    order = spectre.prioritize(scenes, [archive], initial)
    assert sorted(order + [2]) == list(range(len(scenes)))
    assert order == spectre.prioritize(scenes, [archive], initial)


def test_spectre_estimates_the_frequency_of_potential_collisions_before_averaging_deficits():
    from types import SimpleNamespace

    # Six historical versions share a coordinate. All exact matches count even though k=5.
    candidate, other = sample_block("cut_in_onset", 2, 41, set(), "test")
    values = [0.1, 0.3, 0.4, 0.7, 0.9, 1.0]
    archive = SimpleNamespace(scenarios=[candidate] * len(values) + [other],
                              collision=[False] * 3 + [True] * 4,
                              continuous_margin=-np.array([*values, 0.5]))
    features = np.stack([spectre.parameter_features(s) for s in (candidate, other)])
    attributes = spectre.attributes([candidate, other], [archive], [], features)
    assert attributes[:, :2] == pytest.approx(np.array([[0.5, 0.5], [1.0, 1.0]]))


def test_scoring_uses_the_later_stop_and_direct_predecessor():
    prior = [True, True, False, False, False, False]
    labels = [True, False, True, False, True, False]
    order = [3, 0, 2, 1, 4]
    result = score(prior, labels, order, auc_window=4, new_window=3)
    assert result["q90"] == 5 and result["q90_target"] == 3
    assert result["auc_500"] == pytest.approx(5 / 12)
    assert result["new_collisions"] == 2 and result["new_in_first_100"] == 1
    assert result["first_new"] == 3
    with pytest.raises(ValueError, match="later"):
        score(prior, labels, order + [5], auc_window=4)
    zero = score(labels, [False] * len(labels), order[:4], auc_window=4)
    assert zero["q90"] is zero["auc_500"] is None


def test_static_greedy_uses_the_shared_prefix_and_candidate_only_distances():
    scenes = sample_block("cut_in_onset", 15, 41, set(), "test")
    initial = [0, 3]
    order = prioritize(scenes, initial)
    assert order == prioritize(scenes, initial)
    assert sorted(order + initial) == list(range(len(scenes)))
    assert order != prioritize(scenes, [2, 3])


def test_evaluator_stops_isolated_selector_and_resumes_its_paid_prefix(tmp_path):
    scenes = sample_block("cut_in_onset", 10, 41, set(), "test")
    initial = [0, 3]
    labels = [QueryFeedback(scene.scenario_id, i in initial, SafetyMeasurements(0, 0), 0).to_dict()
              for i, scene in enumerate(scenes)]
    task = {"candidates": scenes, "initial_indices": initial, "method": "birchler"}
    first = FixedReplayLedger(labels, initial, tmp_path, {"test": True}, window=5)
    first.query(0, purpose="initial")
    with pytest.raises(ValueError, match="duplicated"):
        first.query(0, purpose="initial")
    ledger = FixedReplayLedger(labels, initial, tmp_path, {"test": True}, window=5)
    result = run_isolated(fixed_selector, task, ledger, signal_evaluator_stop=True)
    ledger.finish()
    assert len(ledger.records) == len(result["selected_indices"]) == 5
    assert [row["index"] for row in ledger.records[:2]] == initial


def test_sdc_scissor_refits_after_new_label_and_resumes(tmp_path):
    coordinates = set()
    scenes = sample_block("cut_in_onset", 15, 41, coordinates, "test")
    historical = sample_block("cut_in_onset", 1, 73, coordinates, "history")
    initial = [2, 3]
    feedback = [QueryFeedback(s.scenario_id, i == 0, SafetyMeasurements(0, 0), 0).to_dict()
                for i, s in enumerate(scenes)]
    task = {"candidates": scenes, "initial_indices": initial, "method": "sdc_scissor",
            "history": [(historical[0], False)]}
    first = FixedReplayLedger(feedback, initial, tmp_path, {"test": True}, window=6)
    first.query(initial[0], purpose="initial")
    ledger = FixedReplayLedger(feedback, initial, tmp_path, {"test": True}, window=6)
    result = run_isolated(fixed_selector, task, ledger, signal_evaluator_stop=True)
    ledger.finish()
    assert result["selected_indices"][:3] == [2, 3, 0]
    assert result["queries"][2]["prequery_probability"] == 0
    assert 0 < result["queries"][3]["prequery_probability"] < 1
    assert len(set(result["selected_indices"])) == len(ledger.records) == 6


def test_braking_failure_reset_uses_formal_horizon():
    points = sample_block(BRAKING_KIND, 2, 41, set(), "test")
    env = RepairEnv(points, [], set(), preparation=True)
    try:
        _, info = env.reset(seed=1)
        assert info["curriculum"] == "current"
        assert env.world.duration_s == 14
        assert env.world.environment_seed == 81001
    finally:
        env.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("single_class", [False, True])
def test_implicit_evidence_preserves_the_fixed_fit(device, single_class):
    torch.set_num_threads(1)
    rng = np.random.default_rng(420)
    x = rng.random((75, 5))
    labels = np.zeros(len(x), dtype=bool) if single_class else rng.random(len(x)) < .2
    margins = rng.uniform(-.9, 1.5, len(x))
    base = SharedPrivateKernel(x.shape[1], deep=False).double().to(device)
    fit = native_settings().online.raw
    standard = adapted_online(base, x, labels, margins, fit)
    implicit = adapted_online(base, x, labels, margins, fit, implicit_evidence=True)
    assert standard.adaptation_diagnostics["objective"] == pytest.approx(
        implicit.adaptation_diagnostics["objective"], abs=1e-12)
    for name in fit.parameters:
        torch.testing.assert_close(getattr(implicit, name), getattr(standard, name), rtol=0, atol=1e-10)
