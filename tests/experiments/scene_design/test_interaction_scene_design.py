"""Fixed development domains, cohort routing and causal reference inputs."""

from math import prod
from types import SimpleNamespace

import pytest

from controllers.highway.catalog import FORMAL_HISTORY_SUTS, FORMAL_TARGET_SUTS
from experiments.scene_design.design_interactions import DEVELOPMENT_SCENARIOS, scenes
from experiments.scene_design.references import EscapeReference
from experiments.scenarios import DEVELOPMENT_BOUNDS, validate
from simulator.highway.observation import VehicleState, VisibleWorld, observe


@pytest.mark.parametrize("kind", DEVELOPMENT_SCENARIOS)
def test_development_coordinates_are_integer_unique_and_within_retained_ranges(kind):
    ranges = [("initial", DEVELOPMENT_BOUNDS[kind])]
    for variant, bounds in ranges:
        batch = scenes(kind, variant)
        keys = {s.coordinate() for s in batch}
        assert len(keys) == len(batch) == (32 if kind == "cut_out_rear_gap" else 128)
        assert prod(hi-lo+1 for lo, hi in bounds.values()) >= 2500
        for scene in batch:
            validate(scene)
            assert all(type(v) is int for v in scene.parameters.values())
        if not kind.startswith("cut_out"):
            assert sum(s.categories["side"] == "left" for s in batch) == 64


def test_both_registered_cohorts_have_eight_distinct_systems_despite_role_names():
    assert len(FORMAL_HISTORY_SUTS) == len(FORMAL_TARGET_SUTS) == 8
    assert {s.role for s in FORMAL_TARGET_SUTS} == {"independent"}
    assert {s.sut_id for s in FORMAL_TARGET_SUTS}.isdisjoint(s.sut_id for s in FORMAL_HISTORY_SUTS)


@pytest.mark.parametrize(
    "kind,variant",
    (("cut_out_restricted", "initial"), ("cut_out_reveal", "initial"), ("cut_in_onset", "critical")),
)
def test_retired_domains_cannot_generate_new_development_candidates(kind, variant):
    with pytest.raises(ValueError, match="retained initial development domains"):
        scenes(kind, variant)


def test_rear_gap_extension_retains_original_coordinates_and_frozen_protocol(tmp_path,monkeypatch):
    from experiments.scene_design import design_interactions as design
    from experiments.storage import read_json

    monkeypatch.setattr(design,"OUTPUT",tmp_path)
    directory = design.register("cut_out_rear_gap","initial",32)
    original = (directory/"protocol.json").read_bytes()
    design.register("cut_out_rear_gap","initial",128)
    extended = read_json(directory/"extended_protocol.json")
    assert (directory/"protocol.json").read_bytes() == original
    assert extended["scenes"][:32] == read_json(directory/"protocol.json")["scenes"]
    assert len(extended["scenes"]) == 128
    assert extended["maximum_additional_ads_executions"] == 1792
    assert extended["maximum_additional_reference_executions"] == 38


def test_cached_paid_coordinates_never_start_a_new_physical_execution(tmp_path,monkeypatch):
    from experiments.scene_design import design_interactions as design
    from experiments.storage import write_json

    sut = FORMAL_TARGET_SUTS[0]
    group = scenes("cut_out_rear_gap")[:2]
    for scene in group:
        path = tmp_path/sut.sut_id/f"{scene.scenario_id.rsplit(':',1)[-1]}.json.gz"
        write_json(path,{"summary":{"scenario_id":scene.scenario_id}})
    monkeypatch.setattr(design,"execute",lambda *a,**kw:pytest.fail("Cached observations must not be executed"))
    name,records = design.run_chunk((sut,group,tmp_path))
    assert name == sut.sut_id and [r["scenario_id"] for r in records] == [s.scenario_id for s in group]


@pytest.mark.parametrize("partner,reveal,moved,expected",[
    ("obstacle",0.5,True,True),
    ("guard_rear",0.5,True,False),
    ("obstacle",0,True,False),
    ("obstacle",1,True,False),
    ("obstacle",0.5,False,False),
])
def test_independent_audit_requires_obstacle_reveal_and_actual_cut_out(partner,reveal,moved,expected):
    from experiments.scene_design.audit_interactions import mechanism_event

    trace = [
        {"time_s":t,"actors":[{"actor_id":"lead","y":4-(t if moved else 0)}],
         "visible_ids":["obstacle"] if t>=reveal else [],
         "collision":t==1,"collision_partner":partner if t==1 else None}
        for t in (0,0.5,1)
    ]
    assert mechanism_event({"scenario":{"kind":"cut_out_rear_gap"},"trajectory":trace})[0] == expected


def test_rear_gap_layout_and_real_initial_occlusion_without_native_execution(monkeypatch):
    from simulator.highway import collision_env

    class StaticObstacle:
        LENGTH, WIDTH = 2.0, 2.0

        def __init__(self, road, position):
            self.position, self.speed = position, 0

    class Layout:
        def __init__(self, scenario):
            self.scenario, self.road, self.actors = scenario, object(), {}

        def add(self, role, x, y, speed, **schedule):
            self.actors[role] = SimpleNamespace(
                position=(x,y), speed=speed, LENGTH=5.0, WIDTH=2.0,
            )

    monkeypatch.setattr(collision_env, "Obstacle", StaticObstacle)
    for scene in scenes("cut_out_rear_gap"):
        layout = Layout(scene)
        collision_env.CollisionWorld._create_traffic(layout)
        assert layout.actors["lead"].position[0] == 116.5
        assert layout.actors["guard_rear"].position == (100-scene.parameters["rear_gap_m"],0)
        ego = VehicleState("ego",100,4,scene.parameters["ego_speed_mps"],0)
        actors = tuple(
            VehicleState(role,*a.position,a.speed,0,a.LENGTH,a.WIDTH)
            for role,a in layout.actors.items()
        )
        visible = observe(0,(ego,*actors),lane_centers=(0,4))
        assert "lead" in {a.actor_id for a in visible.vehicles}
        assert "obstacle" not in {a.actor_id for a in visible.vehicles}
        bodies = [a.polygon() for a in (ego,*actors)]
        assert all(not a.intersects(b) for i,a in enumerate(bodies) for b in bodies[i+1:])


@pytest.mark.parametrize("side_y,target_y", ((0,8),(8,0)))
def test_escape_reference_uses_visible_relative_velocity_and_is_mirror_symmetric(side_y,target_y):
    ego = VehicleState("ego",100,4,20,0)
    slow = VehicleState("anonymous",110,side_y,8,0)
    guards = (VehicleState("front",115,target_y,20,0),VehicleState("rear",85,target_y,20,0))
    reference = EscapeReference()
    command = reference.act(VisibleWorld(0,ego,(slow,*guards)))
    assert reference.target_y == target_y
    assert command.acceleration == 0
    assert (command.steering > 0) == (target_y > 4)
