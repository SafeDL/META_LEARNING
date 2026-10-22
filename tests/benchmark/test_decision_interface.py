from dataclasses import replace

import numpy as np

from controllers.highway.decision_interface import DecisionInterface, public_world
from controllers.highway.control import Control
from simulation.highway.observation import VehicleState, VisibleWorld, policy_observation


def test_script_intent_names_do_not_enter_the_controller_boundary():
    ego = VehicleState("ego", 0, 4, 25, 0)
    actor = VehicleState("cut_in", 25, 0, 20, 0)
    a = VisibleWorld(0, ego, (actor,))
    b = VisibleWorld(0, ego, (replace(actor, actor_id="cruise"),))
    assert public_world(a) == public_world(b)
    assert public_world(a).vehicles[0].actor_id == "vehicle_00"
    class Policy:
        def act(self, world):
            assert [v.actor_id for v in world.vehicles] == ["vehicle_00"]
            assert not hasattr(world, "scenario")
            return Control(0, 0)
    DecisionInterface(Policy()).act(a)


def test_anonymous_boundary_preserves_the_frozen_learning_vectors_including_ties():
    ego = VehicleState("ego", 0, 4, 25, 0)
    # All actors have the same center distance, making the tie order observable.
    actors = (VehicleState("rear", -10, 4, 28, 0),
              VehicleState("obstacle", 10, 4, 0, 0),
              VehicleState("lead", 0, 14, 20, 0))
    world = VisibleWorld(2, ego, actors)
    for target in (None, 0.0, 4.0, 8.0):
        np.testing.assert_array_equal(policy_observation(world, target_lane_y=target),
                                      policy_observation(public_world(world), target_lane_y=target))
