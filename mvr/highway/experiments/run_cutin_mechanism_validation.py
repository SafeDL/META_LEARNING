"""Record and validate the physical action semantics of E6 interaction modes."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mvr.highway.envs.cutin_env import (
    CutInEnv,
    CutInScenario,
    LeadVehicleTrace,
    run_cutin_episode_with_trace,
)
from mvr.highway.sut.idm_profiles import get_profile


VALIDATION_PATH = Path(
    "results/diva_highway/cutin_mvp_e6_action_fix/mechanism_validation.json"
)
SAFE_SCENARIO = (40.0, 0.0)
SEED = 20260912
SIMULATION_DT = 1 / CutInEnv.default_config()["simulation_frequency"]
BRAKE_START = CutInEnv.CUTIN_START + CutInEnv.DEFAULT_CUTIN_DURATION
BRAKE_END = BRAKE_START + CutInEnv.BRAKE_DURATION
BRAKE_ACCELERATION = -CutInEnv.BRAKING_DECELERATION


def _first_time(trace: LeadVehicleTrace, condition: np.ndarray) -> float | None:
    indices = np.flatnonzero(condition)
    return float(trace.time[indices[0]]) if len(indices) else None


def _trace_payload(trace: LeadVehicleTrace) -> dict:
    return {
        "time_seconds": trace.time.tolist(),
        "lead_acceleration_mps2": trace.acceleration.tolist(),
        "lead_speed_mps": trace.speed.tolist(),
        "lead_lateral_position_m": trace.lateral_position.tolist(),
    }


def _cutin_timing(trace: LeadVehicleTrace) -> dict:
    initial_lateral_position = float(trace.lateral_position[0])
    return {
        "first_lateral_motion_seconds": _first_time(
            trace, np.abs(trace.lateral_position - initial_lateral_position) >= 0.05
        ),
        "lane_centerline_crossing_seconds": _first_time(
            trace, trace.lateral_position <= initial_lateral_position / 2
        ),
        "target_lane_settling_seconds": _first_time(
            trace, np.abs(trace.lateral_position) <= 0.1
        ),
    }


def _braking_measurement(trace: LeadVehicleTrace) -> dict:
    brake_indices = np.flatnonzero(
        np.isclose(trace.acceleration, BRAKE_ACCELERATION, atol=1e-9)
    )
    if len(brake_indices) == 0:
        raise AssertionError("The cut-in + braking trace contains no -4.5 m/s^2 action")
    first = int(brake_indices[0])
    last = int(brake_indices[-1])
    return {
        "sample_count": int(len(brake_indices)),
        "start_seconds": float(trace.time[first]),
        "end_seconds": float(trace.time[last]),
        "duration_seconds": float(trace.time[last] - trace.time[first] + SIMULATION_DT),
        "acceleration_mps2": float(np.mean(trace.acceleration[brake_indices])),
        "speed_drop_mps": float(trace.speed[first - 1] - trace.speed[last]),
    }


def _window_speed_drop(trace: LeadVehicleTrace) -> float:
    start = np.flatnonzero(trace.time <= BRAKE_START)[-1]
    end = np.flatnonzero(trace.time >= BRAKE_END)[0]
    return float(trace.speed[start] - trace.speed[end])


def main() -> None:
    profile = get_profile("SUT-A")
    braking_result, braking_trace = run_cutin_episode_with_trace(
        profile, CutInScenario(*SAFE_SCENARIO, CutInEnv.CUTIN_BRAKING), seed=SEED
    )
    baseline_result, baseline_trace = run_cutin_episode_with_trace(
        profile, CutInScenario(*SAFE_SCENARIO, "single"), seed=SEED
    )
    _, fast_trace = run_cutin_episode_with_trace(
        profile, CutInScenario(*SAFE_SCENARIO, CutInEnv.FAST_INTRUSION), seed=SEED
    )
    braking = _braking_measurement(braking_trace)
    checks = {
        "safe_braking_sample_did_not_collide": not braking_result.collision,
        "brake_acceleration_is_minus_4_5_mps2": bool(
            np.isclose(braking["acceleration_mps2"], BRAKE_ACCELERATION, atol=1e-9)
        ),
        "brake_duration_is_one_second": bool(
            np.isclose(braking["duration_seconds"], 1.0, atol=1e-9)
        ),
        "braking_speed_drop_is_4_5_mps": bool(
            np.isclose(braking["speed_drop_mps"], 4.5, atol=1e-9)
        ),
        "unbraked_control_has_no_same_window_speed_drop": bool(
            np.isclose(_window_speed_drop(baseline_trace), 0.0, atol=1e-9)
        ),
    }
    artifact = {
        "schema": "highway_diva_mine_e6_action_validation_v1",
        "simulator": {"highway_env_version": version("highway-env")},
        "scenario": {
            "sut": "SUT-A",
            "initial_gap_m": SAFE_SCENARIO[0],
            "relative_speed_mps": SAFE_SCENARIO[1],
            "seed": SEED,
            "simulation_dt_seconds": SIMULATION_DT,
        },
        "braking_measurement": braking,
        "unbraked_speed_drop_mps": _window_speed_drop(baseline_trace),
        "cutin_timing": {
            "fast_intrusion": _cutin_timing(fast_trace),
            "cutin_braking": _cutin_timing(braking_trace),
        },
        "checks": checks,
        "traces": {
            "cutin_braking": _trace_payload(braking_trace),
            "unbraked_control": _trace_payload(baseline_trace),
            "fast_intrusion": _trace_payload(fast_trace),
        },
        "episode_outcomes": {
            "cutin_braking_collision": braking_result.collision,
            "unbraked_control_collision": baseline_result.collision,
        },
    }
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit(f"Action-semantic validation failed; see {VALIDATION_PATH}")
    summary = {"checks": checks, "cutin_timing": artifact["cutin_timing"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
