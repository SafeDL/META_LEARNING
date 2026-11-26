"""Eight historical ADSs and eight independently configured target systems."""

from dataclasses import dataclass, field, replace

from experiments.scenarios import FORMAL_SCENARIOS, TEST_SCENARIOS
from experiments.scenarios import RECORDED_DOMAINS as DOMAINS


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
}


def _historical_objects():
    output = []
    for route, (family, base) in ROUTES.items():
        name = f"history_{route}"
        run, seed = LEARNING_RUNS.get(name, (None, None))
        output.append(
            SUT(
                name,
                family,
                route,
                "history-persistent-speed-training" if seed is not None else "history",
                "history",
                dict(base),
                run,
                seed,
            )
        )
    return tuple(output)


HISTORY_SUTS = _historical_objects()


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
BY_ID = {sut.sut_id: sut for sut in (*HISTORY_SUTS, *INDEPENDENT_SUTS)}

# The original objects are part of frozen execution contracts. Current experiments
# register their four supported scenarios without changing a controller or checkpoint.
CURRENT_SUTS = tuple(
    replace(sut, supported_scenarios=list(TEST_SCENARIOS))
    for sut in (*HISTORY_SUTS, *INDEPENDENT_SUTS)
)
CURRENT_BY_ID = {sut.sut_id: sut for sut in CURRENT_SUTS}

# Registered before testing transfer performance. Old objects and their weights
# remain addressable for development and frozen execution records.
FORMAL_CHANGES = {
    "idm_mobil": {"desired_speed_mps": 29, "headway_s": 1.45, "politeness": 0.4},
    "fvdm_lane": {"desired_speed_mps": 31, "headway_s": 1.65, "velocity_gain": 0.7},
    "linear_mpc": {"desired_speed_mps": 29, "horizon_s": 3.4, "headway_s": 1.6},
    "nonlinear_trajectory": {
        "desired_speed_mps": 30,
        "horizon_s": 3,
        "headway_s": 1.6,
        "effort_weight": 0.12,
    },
    "uct": {"desired_speed_mps": 29, "iterations": 112, "exploration": 1.3},
    "beam": {"desired_speed_mps": 31, "horizon_steps": 6, "beam_width": 14},
    "ppo": {},
    "dqn": {},
}
FORMAL_TARGET_SUTS = tuple(
    SUT(
        f"target_{route}",
        family,
        route,
        "registered-native",
        "independent",
        {**base, **FORMAL_CHANGES[route]},
        f"target_{route}_native" if family == "learning" else None,
        {"ppo": 6113, "dqn": 6119}.get(route),
        "history_dqn_native" if route == "dqn" else f"history_{route}",
        list(FORMAL_SCENARIOS),
    )
    for route, (family, base) in ROUTES.items()
)
FORMAL_HISTORY_SUTS = tuple(
    replace(
        sut,
        **(
            {
                "sut_id": "history_dqn_native",
                "training_run": "history_dqn_native_adapted",
                "training_seed": 4119,
                "version": "native-training",
            }
            if sut.lineage == "dqn"
            else {}
        ),
        supported_scenarios=list(FORMAL_SCENARIOS),
    )
    for sut in HISTORY_SUTS
)
FORMAL_SUTS = (*FORMAL_HISTORY_SUTS, *FORMAL_TARGET_SUTS)
FORMAL_BY_ID = {sut.sut_id: sut for sut in FORMAL_SUTS}
BY_ID.update({s.sut_id: s for s in FORMAL_SUTS if s.sut_id not in BY_ID})
CURRENT_BY_ID.update(FORMAL_BY_ID)
