"""Counterfactual evidence must establish the changed observation and safe outcome."""

import copy

import pytest

from controllers.highway.catalog import FORMAL_TARGET_SUTS
from experiments.storage import write_json
from experiments.scene_design.verify_secondary import compare_pair


def pair():
    scenario = {
        "scenario_id": "interaction-design:cut_out_rear_gap:initial:115",
        "kind": "cut_out_rear_gap", "parameters": {}, "categories": {},
    }
    factual = {"scenario": scenario, "summary": {"native_valid": True, "violations": []}, "trajectory": []}
    control = {"scenario": scenario, "violations": [], "trajectory": []}
    for i in range(15):
        t = round(i * 0.05, 6)
        actors = [
            {"actor_id": "ego", "x": 100, "y": 4, "speed": 20, "heading": 0},
            {"actor_id": "lead", "x": 120, "y": 4-t, "speed": 20, "heading": 0},
            {"actor_id": "guard_rear", "x": 80, "y": 0, "speed": 20, "heading": 0},
        ]
        command = {"acceleration": 0, "steering": 0}
        frame = {
            "time_s": t, "actors": actors, "visible_ids": ["lead", "guard_rear"],
            "control": command.copy(), "requested_control": command.copy(),
            "collision": False, "collision_partner": None, "background_controls": {},
        }
        control["trajectory"].append(frame)
        frame = copy.deepcopy(frame)
        frame["actors"].append({"actor_id": "obstacle", "x": 200, "y": 4, "speed": 0, "heading": 0})
        if t >= 0.25:
            frame["visible_ids"].append("obstacle")
        if t >= 0.45:
            frame["requested_control"]["steering"] = 0.1
        if t == 0.7:
            frame.update(collision=True, collision_partner="guard_rear")
        factual["trajectory"].append(frame)
    control["trajectory"].append({**copy.deepcopy(control["trajectory"][-1]), "time_s": 10})
    return factual, control


def test_secondary_collision_requires_identical_prefix_then_control_change_and_safe_control():
    factual, control = pair()
    result = compare_pair(factual, control)
    assert result["confirmed"]
    assert result["first_responsive_decision_s"] == 0.4
    assert result["first_control_difference_s"] == 0.4
    assert result["prefix_frames_matching"] == 9


@pytest.mark.parametrize("failure", ["prefix", "collision", "departure", "no_control_change", "initially_visible"])
def test_event_order_alone_does_not_confirm_causality(failure):
    factual, control = pair()
    if failure == "prefix":
        control["trajectory"][7]["actors"][0]["x"] += 0.01
    elif failure == "collision":
        control["trajectory"][-1].update(collision=True, collision_partner="lead")
    elif failure == "departure":
        control["violations"] = ["road_departure"]
    elif failure == "no_control_change":
        for frame in factual["trajectory"]:
            frame["requested_control"]["steering"] = 0
    else:
        factual["trajectory"][0]["visible_ids"].append("obstacle")
    assert not compare_pair(factual, control)["confirmed"]


def test_all_started_controls_count_toward_both_caps_and_cannot_be_retried(tmp_path):
    from experiments.scene_design.design_interactions import auxiliary_executions
    from experiments.scene_design.verify_secondary import counterfactual

    sut = next(s for s in FORMAL_TARGET_SUTS if s.sut_id == "target_uct")
    factual, _ = pair()
    write_json(tmp_path / sut.sut_id / "115.json.gz", factual)
    for i in range(8):
        write_json(tmp_path / "counterfactual" / "attempts" / f"other_{i}.json", {"status": "failed"})
    assert auxiliary_executions(tmp_path) == 8
    assert counterfactual(sut, 115, tmp_path) is None
    write_json(tmp_path / "counterfactual" / "attempts" / "target_uct_115.json", {"status": "running"})
    with pytest.raises(ValueError, match="interrupted control"):
        counterfactual(sut, 115, tmp_path)


def test_reference_budget_includes_obstacle_removal_controls(tmp_path):
    from experiments.scene_design.design_interactions import reference, scenes

    for i in range(63):
        write_json(tmp_path / "reference" / f"used_{i}.json.gz", {})
    write_json(tmp_path / "counterfactual" / "attempts" / "used.json", {"status": "complete"})
    with pytest.raises(ValueError, match="64-reference budget"):
        reference(scenes("cut_out_rear_gap")[0], tmp_path, "brake")


def test_intervention_preserves_every_vehicle_and_schedule(monkeypatch):
    from simulator.highway.collision_env import CollisionWorld
    from experiments.scene_design.verify_secondary import NoObstacleWorld

    actors = {name: object() for name in ("ego", "lead", "guard_rear", "obstacle")}
    original = actors.copy()
    monkeypatch.setattr(CollisionWorld, "_create_traffic", lambda world: world.actors.update(actors))
    # Invoke the override on a subclass instance without constructing a native episode.
    layout = object.__new__(NoObstacleWorld)
    layout.actors = {}
    NoObstacleWorld._create_traffic(layout)
    assert layout.actors == {name: actor for name, actor in original.items() if name != "obstacle"}


def test_failed_target_coverage_never_starts_historical_executions(tmp_path, monkeypatch):
    from experiments.scene_design import verify_secondary as verify

    monkeypatch.setattr(verify, "register", lambda directory: None)
    write_json(tmp_path / "extended_protocol.json", {"scenes": []})
    monkeypatch.setattr(verify, "evaluate", lambda sut, group, directory: {"sut_id": sut.sut_id, "coverage_passed": False})
    monkeypatch.setattr(verify, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("Failed targets cannot authorize histories"))
    result = verify.run(tmp_path)
    assert result["status"] == "coverage_insufficient"
    assert len(result["rows"]) == 8
