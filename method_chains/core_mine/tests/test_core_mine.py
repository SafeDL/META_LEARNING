from __future__ import annotations

from dataclasses import replace

import numpy as np

from method_chains.core_mine.acquisition import marginal_scores
from method_chains.core_mine.config import CoreMineConfig
from method_chains.core_mine.data import CachedTask, response_value
from method_chains.core_mine.experiment import run_campaign
from method_chains.core_mine.metrics import cvs
from method_chains.core_mine.oracle import CacheOracle
from method_chains.core_mine.posterior import PosteriorModel
from method_chains.core_mine.source_safe_development import campaign as source_safe_campaign


def _task() -> CachedTask:
    modes = np.asarray(["fast_intrusion", "fast_intrusion", "slow_lead_following", "slow_lead_following"])
    event = np.asarray([True, True, False, True]); collision = np.asarray([False, True, False, False])
    source_event = np.asarray([[True, False, False, False], [False, True, False, True]])
    source_collision = np.asarray([[False, False, False, False], [False, True, False, False]])
    source_y = response_value(np.asarray([[1., 2., np.inf, 1.], [2., 1., np.inf, 2.]]), source_event, source_collision)
    task = CachedTask(1, "Target-exact-global-1", "exact", "global", np.zeros((4, 2)), modes, np.zeros((4, 2)),
                      np.asarray([[0., 0., 0., 0.], [.1, .1, .1, .1], [.2, .2, 0., 0.], [.3, .3, 0., 0.]]),
                      ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1), (0, 1)), source_y, source_event, source_collision,
                      response_value(np.asarray([1., .1, np.inf, 2.]), event, collision), event, collision,
                      np.asarray([1., .1, np.inf, 2.]), np.asarray([False, False, False, True]))
    return replace(task, anchors=np.tile(task.anchors, (13, 1))[:50], modes=np.tile(task.modes, 13)[:50], controls=np.tile(task.controls, (13, 1))[:50],
                   features=np.tile(task.features, (13, 1))[:50], active_dimensions=task.active_dimensions * 13,
                   source_y=np.tile(task.source_y, (1, 13))[:, :50], source_event=np.tile(task.source_event, (1, 13))[:, :50], source_collision=np.tile(task.source_collision, (1, 13))[:, :50],
                   target_y=np.tile(task.target_y, 13)[:50], target_event=np.tile(task.target_event, 13)[:50], target_collision=np.tile(task.target_collision, 13)[:50], target_ttc=np.tile(task.target_ttc, 13)[:50], source_safe_target_failure=np.tile(task.source_safe_target_failure, 13)[:50])


def test_response_encoding_and_oracle_budget() -> None:
    task = _task(); oracle = CacheOracle(task); value = oracle.reveal(1)
    assert value.event and value.collision and value.severity == 1
    assert response_value(np.asarray([np.inf]), np.asarray([False]), np.asarray([False]))[0] == 0
    try: oracle.reveal(1)
    except ValueError: pass
    else: raise AssertionError("duplicate execution was accepted")


def test_probabilities_are_ordered_and_evidence_is_single_use() -> None:
    task = _task(); model = PosteriorModel(task, CoreMineConfig(), "composition", True)
    before = model.predict(); assert np.all((before["p_collision"] >= 0) & (before["p_collision"] <= before["p_event"]) & (before["p_event"] <= 1))
    outcome = CacheOracle(task).reveal(0); model.observe(0, outcome)
    try: model.observe(0, outcome)
    except ValueError: pass
    else: raise AssertionError("posterior counted one observation twice")


def test_coverage_and_campaign_charge_every_query_once() -> None:
    task = _task(); scores = marginal_scores(task.features, task.modes, [], [], np.ones(task.count), np.zeros(task.count), .1)
    assert np.all(np.isfinite(scores)); assert cvs(task, np.asarray([0, 1])) >= .5
    rows, trace = run_campaign(task, "CoRe-Marginal", CoreMineConfig())
    assert len(trace) == 50
    assert len({row["index"] for row in trace}) == len(trace)
    assert rows[-1]["budget"] == 50


def test_source_safe_campaign_charges_50_eligible_queries_without_target_leakage() -> None:
    modes = np.repeat(np.asarray(["fast_intrusion", "cutin_braking", "lead_braking",
                                  "stop_and_go", "slow_lead_following"]), 12)
    features = np.zeros((60, 4))
    features[:, 0] = np.tile(np.linspace(0.0, 1.0, 12), 5)
    source_event = np.zeros((3, 60), dtype=bool)
    source_event[0, ::12] = True
    target_event = np.zeros(60, dtype=bool)
    target_event[6::12] = True
    target_collision = np.zeros(60, dtype=bool)
    source_y = np.tile(np.linspace(0.0, .2, 60), (3, 1))
    task = CachedTask(41, "Target-synthetic", "leave_one_out", "synthetic",
                      np.zeros((60, 2)), modes, np.zeros((60, 2)), features,
                      tuple((0, 1) for _ in range(60)), source_y, source_event,
                      np.zeros_like(source_event),
                      response_value(np.full(60, np.inf), target_event, target_collision),
                      target_event, target_collision, np.full(60, np.inf),
                      (~source_event.any(axis=0)) & target_event)
    result = source_safe_campaign(task, "mean", 0.0)
    selected = [int(value) for value in result["queried_indices"].split(";")]
    assert len(selected) == len(set(selected)) == 50
    assert not source_event.any(axis=0)[selected].any()
    changed_event = target_event.copy()
    changed_event[59] = not changed_event[59]
    changed = replace(task, target_event=changed_event,
                      target_y=response_value(np.full(60, np.inf), changed_event, target_collision),
                      source_safe_target_failure=(~source_event.any(axis=0)) & changed_event)
    changed_result = source_safe_campaign(changed, "mean", 0.0)
    assert selected[0] == int(changed_result["queried_indices"].split(";")[0])
