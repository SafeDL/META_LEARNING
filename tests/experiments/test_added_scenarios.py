import pytest

from controllers.highway.catalog import CURRENT_SUTS
from controllers.highway.control import Control
from controllers.highway.factory import create
from experiments.scenarios import DOMAINS, RECORDED_SCENARIOS, features, sample
from simulator.highway.collision_env import CollisionWorld, execute


def test_following_is_deterministic_nonoverlapping_and_hides_future_schedules():
    scene = sample("car_following", 1, "qualification")[0]
    worlds = [CollisionWorld(scene, duration_s=1) for _ in range(2)]
    states = worlds[0].states()
    for i, first in enumerate(states):
        assert all(not first.polygon().intersects(other.polygon()) for other in states[i + 1 :])
    for world in worlds:
        while not world.terminal:
            visible = world.visible_world()
            assert not hasattr(visible, "event_time_s")
            world.advance(Control(-8, 0))
    assert worlds[0].trace == worlds[1].trace
    assert len(worlds[0].actors) == 2
    assert features([scene]).shape == (1, DOMAINS["car_following"].dimension)
    assert len(RECORDED_SCENARIOS) == 5 and len(DOMAINS) == 6


@pytest.mark.parametrize("sut", CURRENT_SUTS, ids=lambda s: s.sut_id)
def test_registered_controller_routes_execute_following(sut):
    scene = sample("car_following", 1, "qualification")[0]
    execution = execute(create(sut.sut_id), scene, duration_s=0.4)
    assert execution.feedback.scenario_id == scene.scenario_id
    assert isinstance(execution.feedback.collision, bool)
    assert execution.trajectory[-1]["time_s"] > 0
