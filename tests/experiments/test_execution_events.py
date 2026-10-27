import pytest

from controllers.highway.control import Control
from experiments.execution_events import analyze_events
from experiments.qualification import brake_scene
from simulator.highway.collision_env import CollisionWorld


def test_event_and_response_times_reconstruct_from_actual_controls():
    scene = brake_scene("event_reconstruction", event_time=1)
    world = CollisionWorld(scene, duration_s=2, lane_centers=(4.0,))
    while not world.terminal:
        world.advance(Control(0 if world.tick < 24 else -8, 0))
    events = analyze_events(scene, world.trace)
    assert events["scheduled_event_s"] == 1
    assert 1 < events["measured_background_onset_s"] < 1.3
    assert events["first_ego_brake_s"] == 1.2
    assert events["brake_latency_from_scheduled_event_s"] == pytest.approx(0.2)
    assert events["obstacle_visible_s"] is None
    assert events["brake_latency_from_reveal_s"] is None
