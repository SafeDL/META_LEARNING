from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import json
from types import SimpleNamespace

import pytest

import benchmark.audit_collision_results as audit
from benchmark.audit_collision_results import RawEvidence, compare, reconstruct, time_to_overlap, trace_facts
from benchmark.collision_metrics import summarize
from benchmark.initialization import initial_indices
from benchmark.scenarios import sample
from benchmark.studies import Task


def actor(name, x, *, y=4, speed=0):
    return {"actor_id": name, "x": x, "y": y, "speed": speed, "heading": 0, "length": 5, "width": 2}


def trace(collision):
    return [{"time_s": time, "actors": [actor("ego", x, speed=10), actor("lead", other)],
             "control": {"acceleration": -8, "steering": 0},
             "collision": collision and time == 1, "collision_partner": "lead" if collision and time == 1 else None}
            for time, x, other in ((0, 0, 20), (1, 10, 14 if collision else 30))]


def test_independent_geometry_and_metric_reconstruction_rejects_poisoned_labels():
    scene = sample("front_brake", 1, "development")[0]
    scene = replace(scene, parameters={**scene.parameters, "event_time_s": 0.5})
    facts = trace_facts(asdict(scene), trace(True))
    assert facts == {"collision": True, "signature": ["lead", "front", ["lane_keep", "brake"], "post_event"],
                     "safety": {"min_clearance_m": 0, "min_ttc_s": 0, "violations": []}}
    safe = trace_facts(asdict(scene), trace(False))
    assert safe["safety"] == {"min_clearance_m": 15, "min_ttc_s": 1.5, "violations": []}
    assert time_to_overlap(actor("ego", 0, speed=10), actor("lead", 20, y=8)) == float("inf")
    scenes = [asdict(scene), asdict(replace(scene, scenario_id="same_bin"))]
    queries = [{"feedback": {"scenario_id": s["scenario_id"], "collision": True,
                            "safety": facts["safety"], "execution_s": 2}, "prequery_probability": 0.8}
               for s in scenes]
    result = reconstruct(queries, scenes, [facts, facts])
    assert result["collision_executions"] == 2 and result["unique_collisions"] == 1
    assert result["discovery_curve"] == [1, 1] and result["discovery_auc"] == 1
    assert result["behavior_coverage"] == 1 and result["behavior_counts"]["ego_action"] == {"lane_keep/brake": 2}
    assert result["brier"] == pytest.approx(0.04) and result["equivalent_execution_s"] == 4
    poisoned = deepcopy(queries)
    poisoned[0]["feedback"]["collision"] = False
    with pytest.raises(ValueError, match="Raw collision"):
        reconstruct(poisoned, scenes, [facts, facts])
    poisoned = deepcopy(queries)
    poisoned[0]["feedback"]["safety"]["min_clearance_m"] = 9
    with pytest.raises(ValueError, match="raw safety"):
        reconstruct(poisoned, scenes, [facts, facts])
    censored_query = {"feedback": {"collision": False, "safety": safe["safety"], "execution_s": 3}}
    censored = reconstruct([censored_query], [scenes[0]], [safe])
    assert censored["first_collision"] == {"time": 1, "observed": False}
    assert censored["first_collision_query"] is None and censored["behavior_signatures"] == []
    with pytest.raises(ValueError):
        compare({"collision": 1}, {"collision": True})


def test_raw_evidence_retains_failed_attempt_cost_and_rejects_forged_safety(tmp_path):
    scene = asdict(sample("front_brake", 1, "development")[0])
    raw_trace = trace(False)
    safety = trace_facts(scene, raw_trace)["safety"]
    feedback = {"scenario_id": scene["scenario_id"], "collision": False, "safety": safety, "execution_s": 2}
    (tmp_path / "executions").mkdir()
    (tmp_path / "attempts").mkdir()
    manifest = {"status": "complete", "scenarios": [scene], "attempted_calls": 2, "successful_calls": 1,
                "failed_or_interrupted_calls": 1, "physical_execution_s": 2,
                "failed_call_wall_s": 3, "interrupted_call_duration_unknown": 0}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [{"call": 1, "index": 0, "status": "scheduled"}, {"call": 1, "index": 0, "status": "failed"},
            {"call": 2, "index": 0, "status": "scheduled"}, {"call": 2, "index": 0, "status": "complete", "execution_s": 2}]
    (tmp_path / "calls.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    for call, attempt in ((1, {"call": 1, "status": "failed", "wall_s": 3}),
                          (2, {"call": 2, "status": "complete", "execution_s": 2, "wall_s": 2.1})):
        (tmp_path / "attempts" / f"{call:06d}.json").write_text(json.dumps(attempt), encoding="utf-8")
    path = tmp_path / "executions/000000.json.gz"
    payload = {"call": 2, "scenario": scene, "feedback": feedback, "trajectory": raw_trace}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    assert RawEvidence().execution(tmp_path, 0)["feedback"] == feedback
    payload["feedback"]["safety"]["min_ttc_s"] = 99
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    with pytest.raises(ValueError, match="raw geometry safety"):
        RawEvidence().execution(tmp_path, 0)


def test_final_run_audit_checks_paid_order_and_matches_independent_metric_implementation(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "DEFAULT", SimpleNamespace(budget=3, initial_count=1, horizon=5, candidate_count=3))
    task = Task("overall", "target_idm_mobil", "rule", "idm_mobil", "front_brake", "new", 81001, 11,
                "gp_collision", "all")
    scenes = sample("front_brake", 3, "new")
    manifest = {"sut_id": task.target_id, "environment_seed": task.environment_seed, "contract": {"fixture": True},
                "scenarios": [asdict(s) for s in scenes], "candidate_count": 3, "attempted_calls": 3, "physical_execution_s": 6}
    evidence = []
    for scene in scenes:
        frames = trace(scene == scenes[1])
        facts = trace_facts(asdict(scene), frames)
        evidence.append({"scenario": asdict(scene), "facts": facts,
                         "feedback": {"scenario_id": scene.scenario_id, "collision": facts["collision"],
                                      "safety": facts["safety"], "execution_s": 2}})
    order = initial_indices(3, 1, 11)
    order.extend(i for i in range(3) if i not in order)
    queries = [{"index": i, "feedback": evidence[i]["feedback"], "prequery_probability": 0.5} for i in order]
    ledger = [{"call": position, "index": i, "status": "complete", "account": "target",
               "purpose": "initial" if position == 1 else "target", "scenario_id": scenes[i].scenario_id,
               "feedback": evidence[i]["feedback"]}
              for position, i in enumerate(order, 1)]
    metrics = summarize(scenes, queries, [trace(i == 1) for i in order])
    run = {"specification": {"metadata": asdict(task), "construction": {
                "budget": 3, "initial_count": 1, "horizon": 5, "method": "gp_collision", "algorithm_seed": 11,
                "history_ids": [], "bank_contract": manifest["contract"],
                "candidate_ids": [s.scenario_id for s in scenes]}},
           "result": {"actual_queries": 3, "queries": queries, "selected_indices": order},
           "query_ledger": ledger, "metrics": metrics,
           "cost": {"physical_bank": str(tmp_path), "execution_mode": "replay", "actual_new_physical_calls": 0,
                    "equivalent_selected_execution_s": 6, "physical_bank_calls": 3, "physical_bank_execution_s": 6}}
    class FixtureEvidence:
        def bank(self, root):
            return (manifest,)

        def execution(self, root, index):
            return evidence[index]

    path = tmp_path / "run.json"
    path.write_text(json.dumps(run), encoding="utf-8")
    assert audit.verify_run(path, task, FixtureEvidence())["collision_executions"] == 1
    for mutation, message in ((lambda r: r["query_ledger"][0].update(purpose="target"), "not charged"),
                              (lambda r: r["specification"]["construction"].update(history_ids=["history_ppo"]),
                               "disallowed historical"),
                              (lambda r: r["metrics"].update(unique_collisions=99), "reconstructed run metrics")):
        poisoned = deepcopy(run)
        mutation(poisoned)
        path.write_text(json.dumps(poisoned), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            audit.verify_run(path, task, FixtureEvidence())
