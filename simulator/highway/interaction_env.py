"""Paired cut-in scenarios use the same native highway-env implementation."""

from .collision_env import CollisionWorld

GUARD_FRONT_GAP_M = 10.0
GUARD_REAR_GAP_M = 10.0


class InteractionWorld(CollisionWorld):
    def __init__(self, scenario, **kwargs):
        if scenario.kind != "cut_in_interaction":
            raise ValueError("InteractionWorld requires the paired cut-in task")
        super().__init__(scenario, **kwargs)
