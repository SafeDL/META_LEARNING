"""Common policy boundary: visible geometry without scripted actor-role names."""

from dataclasses import replace


def public_world(world):
    # Ranks preserve the established numerical tie order while exposing no script tags.
    names = {
        name: f"vehicle_{index:02d}"
        for index, name in enumerate(sorted(v.actor_id for v in world.vehicles))
    }
    return replace(
        world, vehicles=tuple(replace(v, actor_id=names[v.actor_id]) for v in world.vehicles)
    )


class DecisionInterface:
    def __init__(self, policy):
        self.policy = policy

    def act(self, world):
        return self.policy.act(public_world(world))
