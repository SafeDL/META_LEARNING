"""Legacy experiment objects and eight separately registered independent systems."""

from dataclasses import dataclass, field

from experiments.scenarios import DOMAINS


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
    supported_scenarios: list[str] = field(
        default_factory=lambda: ["front_brake", "cut_in", "cut_out", "lane_change_rear"]
    )


ROUTES = {
    "idm_mobil": (
        "rule",
        {"headway_s": 1.6, "desired_speed_mps": 30, "politeness": 0.3, "rear_brake_limit": 3},
    ),
    "fvdm_lane": (
        "rule",
        {"headway_s": 1.5, "desired_speed_mps": 30, "velocity_gain": 0.6, "rear_brake_limit": 3},
    ),
    "linear_mpc": (
        "optimization",
        {"horizon_s": 3, "desired_speed_mps": 30, "headway_s": 1.5, "effort_weight": 0.1},
    ),
    "nonlinear_trajectory": (
        "optimization",
        {"horizon_s": 3, "desired_speed_mps": 30, "headway_s": 1.5, "effort_weight": 0.15},
    ),
    "uct": (
        "search",
        {"horizon_steps": 5, "iterations": 96, "exploration": 1.4, "desired_speed_mps": 30},
    ),
    "beam": ("search", {"horizon_steps": 5, "beam_width": 12, "desired_speed_mps": 30}),
    "ppo": ("learning", {}),
    "dqn": ("learning", {}),
}


LEARNING_RUNS = {
    "history_ppo": ("history_ppo_persistent_speed", 4117),
    "history_dqn": ("history_dqn_persistent_speed", 1103),
    "development_ppo": ("development_ppo_persistent_speed", 2111),
    "target_ppo": ("target_ppo_persistent_speed", 3109),
    "target_dqn": ("target_dqn_persistent_speed", 3137),
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
            run, seed = LEARNING_RUNS.get(name, (None, None))
            output.append(
                SUT(
                    name,
                    family,
                    route,
                    f"{role}-persistent-speed-training" if seed is not None else role,
                    role,
                    parameters,
                    run,
                    seed,
                    f"history_{route}" if role == "target" else None,
                )
            )
    for family, route in (
        ("rule", "idm_mobil"),
        ("optimization", "linear_mpc"),
        ("search", "uct"),
        ("learning", "ppo"),
    ):
        parameters = dict(ROUTES[route][1])
        if family != "learning":
            parameters["desired_speed_mps"] = 31
            if "headway_s" in parameters:
                parameters["headway_s"] += 0.2
            if "exploration" in parameters:
                parameters["exploration"] = 1.7
        name = f"development_{route}"
        run, seed = LEARNING_RUNS.get(name, (None, None))
        output.append(
            SUT(
                name,
                family,
                route,
                "development-persistent-speed-training" if family == "learning" else "development",
                "development",
                parameters,
                run,
                seed,
            )
        )
    return tuple(output)


SUTS = _objects()


def _independent_objects():
    changes = {
        "idm_mobil": {
            "desired_speed_mps": 31,
            "headway_s": 1.3,
            "politeness": 0.5,
            "rear_brake_limit": 2.5,
        },
        "fvdm_lane": {
            "desired_speed_mps": 29,
            "headway_s": 1.8,
            "velocity_gain": 0.5,
            "rear_brake_limit": 4,
        },
        "linear_mpc": {
            "desired_speed_mps": 31,
            "horizon_s": 4,
            "headway_s": 1.7,
            "effort_weight": 0.15,
        },
        "nonlinear_trajectory": {
            "desired_speed_mps": 29,
            "horizon_s": 2,
            "headway_s": 1.3,
            "effort_weight": 0.10,
        },
        "uct": {"desired_speed_mps": 31, "horizon_steps": 6, "iterations": 128, "exploration": 1.2},
        "beam": {"desired_speed_mps": 29, "horizon_steps": 4, "beam_width": 16},
        "ppo": {},
        "dqn": {},
    }
    output = []
    for route, (family, base) in ROUTES.items():
        seed = {"ppo": 5107, "dqn": 5113}.get(route)
        name = f"independent_{route}"
        output.append(
            SUT(
                name,
                family,
                route,
                "independent",
                "independent",
                {**base, **changes[route]},
                f"{name}_persistent_speed" if seed is not None else None,
                seed,
                f"history_{route}",
                list(DOMAINS),
            )
        )
    return tuple(output)


INDEPENDENT_SUTS = _independent_objects()
BY_ID = {sut.sut_id: sut for sut in (*SUTS, *INDEPENDENT_SUTS)}
