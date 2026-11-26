"""OnlineMC's Maude search with only present, visible actors."""

import math
from functools import lru_cache

from .public_agents import CACHE, sha256

MODELS = {
    "vehicle.maude": "a01710839223ff3c3a4fccdcd41134311419596fe0978cab5eeac49ec1adbb6b",
    "fmodel.maude": "cb1bf6bf42898f30f2920102f1e098e153c1f09556e420657c18b4c4d7c3699f",
}


def shield_assets():
    for name, expected in MODELS.items():
        if sha256((CACHE / "online_mc" / name).read_bytes()) != expected:
            raise ValueError(f"OnlineMC model differs from registered SHA-256: {name}")
    return dict(MODELS)


@lru_cache(maxsize=1)
def propositions():
    import maude

    shield_assets()
    maude.init()
    for name in MODELS:
        if not maude.load(str(CACHE / "online_mc" / name)):
            raise ValueError(f"Could not load OnlineMC Maude model: {name}")
    return maude.getModule("PROPOSITIONS")


def rounded(value):
    return round(float(value), 3) or 0.0


def shield_states(world, observation):
    ego = world.ego
    ex, ey, vx, vy = (rounded(v) for v in (ego.x, -ego.y, ego.vx, -ego.vy))
    ego_state = ex, ey, vx, vy, rounded(math.degrees(ego.heading))
    npcs = [
        (
            rounded(row[1] * 200 + ex),
            rounded(-row[2] * 12 + ey),
            rounded(row[3] * 80 + vx),
            rounded(-row[4] * 80 + vy),
            rounded(math.degrees(row[5])),
        )
        for row in observation[1:] if row[0]
    ]
    return ego_state, npcs


def vehicle_term(index, state):
    x, y, vx, vy, heading = state
    return f"veh({index}, vector2({x}, {y}), vector2({vx}, {vy}), {heading})"


class OnlineShield:
    def __init__(self):
        import maude

        self.module = propositions()
        self.steps = maude.ANY_STEPS
        parse = self.module.parseTerm
        self.pattern = parse("S:Sys")
        self.condition = [
            maude.AssignmentCondition(parse("< T:Float ; ID:Int >"), parse("ttc(S:Sys)")),
            maude.EqualityCondition(parse("T:Float < 2.0"), parse("true")),
        ]

    def allows(self, action, ego, npcs):
        actor_terms = " & ".join(vehicle_term(i, s) for i, s in enumerate(npcs)) or "empty"
        state = self.module.parseTerm(
            f"-1 | {vehicle_term(1000, ego)} # {action} # 1.0 | ({actor_terms})"
        )
        if state is None:
            raise ValueError("OnlineMC could not construct its state term")
        hits = state.search(self.steps, self.pattern, condition=self.condition, depth=5)
        return next(hits, None) is None

    def choose(self, ranking, observation, world, available):
        ego, npcs = shield_states(world, observation)
        rejected = []
        for item in ranking:
            action = int(item)
            if action == 4 or (action in available and self.allows(action, ego, npcs)):
                return action, rejected
            rejected.append(action)
        return 4, rejected
