from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from types import SimpleNamespace

import pytest

import benchmark.audit_continuous_results as audit
from benchmark.audit_collision_results import trace_facts
from benchmark.collision_metrics import summarize
from benchmark.scenarios import sample


def test_continuous_audit_reconstructs_actual_calls_and_rejects_cost_and_raw_safety_poisoning(tmp_path, monkeypatch):
    scenes = sample("front_brake", 2, "new")
    monkeypatch.setattr(audit, "DEFAULT", SimpleNamespace(budget=2, initial_count=1))
    monkeypatch.setattr(audit, "initial_scenarios", lambda *args: scenes[:1])
    monkeypatch.setattr(audit, "source_contract", lambda: {"fixture": "code"})
    metadata = {"target_id": "target_idm_mobil", "lineage": "idm_mobil", "family": "rule",
                "scenario": "front_brake", "condition": "continuous", "method": "random",
                "algorithm_seed": 11, "environment_seed": 81001}
    identity = {"metadata": metadata, "method": "random", "algorithm_seed": 11,
                "budget": 2, "initial_count": 1, "ordinary_only": False, "configuration": {},
                "source_contract": {"fixture": "code"}, "initial_scenarios": [asdict(scenes[0])]}
    path = tmp_path / "random_11.json"
    root = path.with_suffix("")
    root.mkdir()
    manifest = {"identity": identity, "environment_seed": 81001,
                "sut_contract": {"object": {"sut_id": "target_idm_mobil"}}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    trajectories, queries, ledger = [], [], []
    for index, scene in enumerate(scenes):
        trajectory = []
        for timestamp in (0, 1):
            actors = [{"actor_id": "ego", "x": timestamp * 10, "y": 4, "speed": 10, "heading": 0, "length": 5, "width": 2},
                      {"actor_id": "lead", "x": 20 + timestamp * 10, "y": 4, "speed": 0, "heading": 0, "length": 5, "width": 2}]
            trajectory.append({"time_s": timestamp, "actors": actors, "collision": False,
                               "collision_partner": None, "control": {"acceleration": 0, "steering": 0}})
        trajectories.append(trajectory)
        feedback = {"scenario_id": scene.scenario_id, "collision": False,
                    "safety": trace_facts(asdict(scene), trajectory)["safety"], "execution_s": 2}
        queries.append({"index": index, "feedback": feedback})
        ledger.append({"call": index + 1, "index": index, "status": "complete", "account": "target",
                       "purpose": "initial" if index == 0 else "target", "scenario_id": scene.scenario_id,
                       "feedback": feedback})
        attempt = {"query": index + 1, "scenario": asdict(scene), "status": "complete", "execution_s": 2, "wall_s": 3}
        (root / f"attempt_{index:04d}.json").write_text(json.dumps(attempt), encoding="utf-8")
        with gzip.open(root / f"query_{index:04d}.json.gz", "wt", encoding="utf-8") as stream:
            json.dump({"scenario": asdict(scene), "feedback": feedback, "trajectory": trajectory}, stream)
    run = {"specification": identity, "candidates": [asdict(s) for s in scenes],
           "result": {"actual_queries": 2, "queries": queries, "selected_indices": [0, 1]},
           "query_ledger": ledger, "metrics": summarize(scenes, queries, trajectories),
           "cost": {"physical_log": str(root), "execution_mode": "actual_on_demand", "actual_physical_calls": 2,
                    "physical_execution_s": 4, "sut_initialization_s": 2}}
    path.write_text(json.dumps(run), encoding="utf-8")
    checked = audit.verify_run(path, metadata)
    assert checked["queries"] == 2 and checked["first_collision"] == {"time": 2, "observed": False}
    poisoned = deepcopy(run)
    poisoned["cost"]["actual_physical_calls"] = 1
    path.write_text(json.dumps(poisoned), encoding="utf-8")
    with pytest.raises(ValueError, match="physical call accounting"):
        audit.verify_run(path, metadata)
    path.write_text(json.dumps(run), encoding="utf-8")
    raw_path = root / "query_0000.json.gz"
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    # Poison all disclosed copies consistently; raw geometry must still reject them.
    payload["feedback"]["safety"]["min_ttc_s"] = 99
    run["result"]["queries"][0]["feedback"]["safety"]["min_ttc_s"] = 99
    run["query_ledger"][0]["feedback"]["safety"]["min_ttc_s"] = 99
    with gzip.open(raw_path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="raw safety"):
        audit.verify_run(path, metadata)
