"""Twenty preregistered SUT objects; qualification is recorded separately."""

from dataclasses import dataclass, field

from benchmark.scenarios import DOMAINS


@dataclass(frozen=True)
class SUT:
    sut_id: str
    family: str
    lineage: str
    version: str
    role: str
    parameters: dict[str, float]
    training_run: str | None = None
    training_seed: int | None = None
    reference_id: str | None = None
    supported_scenarios: tuple[str, ...] = field(default_factory=lambda: tuple(DOMAINS))


ROUTES = {
    "idm_mobil": ("rule", {"headway_s": 1.6, "desired_speed_mps": 30,
                            "politeness": 0.3, "rear_brake_limit": 3}),
    "fvdm_lane": ("rule", {"headway_s": 1.5, "desired_speed_mps": 30,
                            "velocity_gain": 0.6, "rear_brake_limit": 3}),
    "linear_mpc": ("optimization", {"horizon_s": 3, "desired_speed_mps": 30,
                                     "headway_s": 1.5, "effort_weight": 0.1}),
    "nonlinear_trajectory": ("optimization", {"horizon_s": 3, "desired_speed_mps": 30,
                                               "headway_s": 1.5, "effort_weight": 0.15}),
    "uct": ("search", {"horizon_steps": 5, "iterations": 96,
                        "exploration": 1.4, "desired_speed_mps": 30}),
    "beam": ("search", {"horizon_steps": 5, "beam_width": 12,
                         "desired_speed_mps": 30}),
    "ppo": ("learning", {}),
    "dqn": ("learning", {}),
}


def _objects():
    output = []
    for route, (family, base) in ROUTES.items():
        for role in ("history", "target"):
            parameters = dict(base)
            if role == "target" and family != "learning":
                parameters["desired_speed_mps"] = 32
                if "headway_s" in parameters:
                    parameters["headway_s"] -= 0.2
                if "exploration" in parameters:
                    parameters["exploration"] = 1.1
                if "beam_width" in parameters:
                    parameters["beam_width"] = 16
            name = f"{role}_{route}"
            seed = ((1101 if route == "ppo" else 1103) if role == "history"
                    else (3109 if route == "ppo" else 3121)) if family == "learning" else None
            reconstruction = role == "target" and route == "dqn"
            output.append(SUT(name, family, route, "target-reconstruction-1" if reconstruction else role,
                              role, parameters,
                              "target_dqn_seed3121" if reconstruction else name if seed is not None else None, seed,
                              f"history_{route}" if role == "target" else None))
    for family, route in (("rule", "idm_mobil"), ("optimization", "linear_mpc"),
                          ("search", "uct"), ("learning", "ppo")):
        parameters = dict(ROUTES[route][1])
        if family != "learning":
            parameters["desired_speed_mps"] = 31
            if "headway_s" in parameters:
                parameters["headway_s"] += 0.2
            if "exploration" in parameters:
                parameters["exploration"] = 1.7
        name = f"development_{route}"
        output.append(SUT(name, family, route, "development", "development", parameters,
                          name if family == "learning" else None,
                          2111 if family == "learning" else None))
    return tuple(output)


SUTS = _objects()
BY_ID = {sut.sut_id: sut for sut in SUTS}
