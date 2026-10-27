import itertools
import json
from dataclasses import asdict

import numpy as np
import pytest

from controllers.highway.catalog import BY_ID, INDEPENDENT_SUTS, SUTS
from controllers.highway.control import Control
from experiments.execution_events import analyze_events
from experiments.executions import execution_contract
from experiments.protocol import TEST_RECORDS_ROOT
from experiments.scenarios import features, sample
from experiments.schema import Scenario
from experiments.source_registration import specification_matches
from simulator.highway.collision_env import CollisionWorld
from simulator.highway.interaction_env import InteractionWorld


def test_interaction_pairs_preserve_physical_parameters_and_split_isolation():
    coordinates = []
    for split in ("history", "development", "new"):
        scenes = sample("cut_in_interaction", 512, split)
        assert len({s.scenario_id for s in scenes}) == 512
        assert [s.categories["adjacent_space"] for s in scenes].count("open") == 256
        for opened, restricted in zip(scenes[::2], scenes[1::2]):
            assert opened.parameters == restricted.parameters
            assert opened.categories["side"] == restricted.categories["side"]
            assert opened.categories["adjacent_space"] == "open"
            assert restricted.categories["adjacent_space"] == "restricted"
        matrix = features(scenes)
        assert matrix.shape == (512, 9)
        np.testing.assert_array_equal(matrix[::2, :7], matrix[1::2, :7])
        coordinates.append({s.coordinate() for s in scenes})
    assert all(not a & b for a, b in itertools.combinations(coordinates, 2))


@pytest.mark.parametrize("pair", (0, 7, 21))
def test_open_world_matches_legacy_and_restricted_background_has_independent_motion(pair):
    scenes = sample("cut_in_interaction", 64, "history")
    opened, restricted = scenes[2 * pair : 2 * pair + 2]
    base = Scenario(
        opened.scenario_id, "cut_in", opened.parameters, {"side": opened.categories["side"]}
    )
    legacy, open_world, restricted_world = (
        CollisionWorld(base),
        InteractionWorld(opened),
        InteractionWorld(restricted),
    )
    for world in (open_world, restricted_world):
        states = world.states()
        assert all(
            a.polygon().distance(b.polygon()) > 0 for a, b in itertools.combinations(states, 2)
        )
        assert world.lane_centers == (0.0, 4.0, 8.0)
    while not open_world.terminal and not restricted_world.terminal:
        for world in (legacy, open_world, restricted_world):
            world.advance(Control(-1, 0))
        assert legacy.trace[-1] == open_world.trace[-1]
        for name in ("cut_in", "guard_front", "guard_rear"):
            actor = restricted_world.actors[name]
            if actor.crashed:
                continue
            expected_speed = (
                restricted.parameters["ego_speed_mps"] - restricted.parameters["closing_speed_mps"]
                if name == "cut_in"
                else restricted.parameters["ego_speed_mps"]
            )
            assert actor.longitudinal_speed == expected_speed
            if name.startswith("guard"):
                initial = next(
                    a for a in restricted_world.trace[0]["actors"] if a["actor_id"] == name
                )
                assert actor.position[0] == pytest.approx(
                    initial["x"] + expected_speed * actor.elapsed
                )
                assert actor.position[1] == initial["y"]
    events = analyze_events(restricted, restricted_world.trace)
    assert events["scheduled_event_s"] == restricted.parameters["event_time_s"]
    assert events["measured_background_onset_s"] is not None


def test_independent_registry_keeps_old_contracts_and_declares_eight_fresh_objects():
    assert len(SUTS) == 20 and len(INDEPENDENT_SUTS) == 8 and len(BY_ID) == 28
    assert not {s.sut_id for s in SUTS} & {s.sut_id for s in INDEPENDENT_SUTS}
    for sut in INDEPENDENT_SUTS:
        assert sut.role == "independent"
        assert len(sut.supported_scenarios) == 5
    learned = {s.lineage: s.training_seed for s in INDEPENDENT_SUTS if s.family == "learning"}
    assert learned == {"ppo": 5107, "dqn": 5113}
    assert BY_ID["independent_fvdm_lane"].parameters["velocity_gain"] == 0.5
    # Existing banks remain valid under the unchanged old physical worlds and object values.
    for sut in SUTS:
        if sut.role not in ("history", "development"):
            continue
        path = TEST_RECORDS_ROOT / "physical" / sut.role / "cut_in" / sut.sut_id / "manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert specification_matches(
            manifest["contract"], json.loads(json.dumps(execution_contract(sut, kind="cut_in")))
        )
        assert asdict(sut) == manifest["contract"]["object"]
        assert (
            "simulator/highway/interaction_env.py"
            in execution_contract(sut, kind="cut_in_interaction")["source_files"]
        )
