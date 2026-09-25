"""Physics and information boundaries used by the regression experiment."""

from __future__ import annotations

import numpy as np
import pytest

from highway_env_benchmark.envs.fbrt_env import run_episode
from highway_env_benchmark.envs.fbrt_scenarios import FBRTScenario
from method_chains.core_mine.idm_revision_pilot import REFERENCE
from method_chains.failure_memory_regression.boundary_memory import Patch
from method_chains.failure_memory_regression.selectors import TargetOracle, run_selector


def test_stop_hold_go_has_real_stop_hold_and_restart() -> None:
    scene = FBRTScenario("phase_check", "fbrt_stop_hold_go", 25.0,
                         lead_deceleration_mps2=3.0)
    outcome, trace = run_episode(REFERENCE, scene, 4179801, with_trace=True)
    phases = [event["phase"] for event in outcome["lead_events"]]
    assert phases == ["BRAKE_TO_STOP", "HOLD_STOP", "RESTART", "CRUISE"]
    stopped = [row for row in trace if row["lead"]["speed_mps"] < 0.02]
    assert stopped[-1]["time_s"] - stopped[0]["time_s"] >= 1.95
    assert len(trace) == 481  # t=0 through t=24, every 0.05 s


def test_emergency_brake_has_measured_reference_pass_and_failure() -> None:
    failed = FBRTScenario("brake_fail", "fbrt_lead_emergency_brake", 20.0,
                          lead_deceleration_mps2=5.0)
    passed = FBRTScenario("brake_pass", "fbrt_lead_emergency_brake", 120.0,
                          lead_deceleration_mps2=5.0)
    failed_outcome, _ = run_episode(REFERENCE, failed, 4179801)
    passed_outcome, _ = run_episode(REFERENCE, passed, 4179801)
    assert failed_outcome["ego_collision"]
    assert failed_outcome["collision_partner"] == "lead"
    assert passed_outcome["completed"]
    assert not passed_outcome["ego_collision"]


def test_oracle_reveals_only_queried_result() -> None:
    oracle = TargetOracle({"one": {"ego_collision": True},
                           "two": {"ego_collision": False}})
    assert oracle.queried == set()
    assert oracle.query("one")["ego_collision"]
    assert oracle.queried == {"one"}
    with pytest.raises(ValueError, match="Repeated query"):
        oracle.query("one")
    assert oracle.queried == {"one"}


def test_region_bandit_uses_failure_boundary_within_function() -> None:
    scenes = [FBRTScenario(name, "fbrt_cutin", 8.0 + 42.0 * x,
                           lane_change_duration_s=1.8)
              for name, x in (("history_margin", 0.8), ("near_boundary", 0.4),
                              ("nearby", 0.41))]
    source = {scene.scenario_id: {"completed": True, "ego_collision": False,
                                  "min_ttc": rank, "min_clearance": rank}
              for rank, scene in enumerate(scenes, 1)}
    patch = Patch("cutin:patch:0", "fbrt_cutin", "old_failure", "old_pass",
                  np.array([0.3, 0.5]), np.array([0.5, 0.5]))
    oracle = TargetOracle({scene.scenario_id: {"ego_collision": False,
                                               "semantic_valid": True}
                           for scene in scenes})
    queries = run_selector("FBRT-RegionBandit", scenes, scenes, source, [patch],
                           oracle, budget=3)
    assert queries[0]["scenario_id"] == "near_boundary"
    assert queries[0]["patch_id"] == patch.patch_id
    assert len(oracle.queried) == 3
