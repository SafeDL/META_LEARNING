"""Real highway-env execution with side-effect-free full-state recording."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from highway_sim_env.envs.cutin_env import CutInEnv, CutInScenario
from sut_algorithms.highway_env.idm_profiles import SUTProfile

from .corpus import ScenarioSpec


@dataclass(frozen=True)
class Observation:
    execution_id: str
    scenario_id: str
    collision: bool
    near_miss: bool
    vulnerability: float
    valid: bool
    invalid_reason: str | None
    simulation_seconds: float
    wall_seconds: float
    min_ttc: float
    min_distance: float
    completed: bool
    trajectory_path: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class TraceCutInEnv(CutInEnv):
    """Recorder-only subclass; it does not change actions or dynamics."""

    def __init__(self, *args, **kwargs):
        self.full_trace: list[tuple[float, ...]] = []
        super().__init__(*args, **kwargs)

    def _record_lead_trace(self) -> None:
        super()._record_lead_trace()
        lead = self._cutin_vehicle
        ego = self.vehicle
        if lead is None or ego is None:
            return
        center = float(np.linalg.norm(lead.position - ego.position))
        clearance = max(0.0, center - (lead.LENGTH + ego.LENGTH) / 2)
        longitudinal = float(lead.position[0] - ego.position[0])
        lateral = float(abs(lead.position[1] - ego.position[1]))
        closing = float(ego.speed - lead.speed)
        ttc = longitudinal / closing if longitudinal > 0 and lateral < ego.WIDTH and closing > 1e-6 else np.inf
        self.full_trace.append((
            float(lead.elapsed), float(ego.position[0]), float(ego.position[1]), float(ego.speed),
            float(ego.heading), float(ego.action.get("acceleration", 0.0)), float(ego.action.get("steering", 0.0)),
            float(lead.position[0]), float(lead.position[1]), float(lead.speed), float(lead.heading),
            float(lead.action.get("acceleration", 0.0)), float(lead.action.get("steering", 0.0)), clearance, ttc,
        ))


TRACE_FIELDS = (
    "time", "ego_x", "ego_y", "ego_speed", "ego_heading", "ego_acceleration", "ego_steering",
    "npc_x", "npc_y", "npc_speed", "npc_heading", "npc_acceleration", "npc_steering", "clearance", "ttc",
)


def execute_scenario(profile: SUTProfile, spec: ScenarioSpec, simulation_seed: int, trajectory_dir: Path | None = None) -> Observation:
    identity = f"{spec.scenario_id}|{profile.name}|{simulation_seed}|highway-env-1.9.1|oracle-v1"
    execution_id = "execution-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    env = TraceCutInEnv(profile, CutInScenario(spec.initial_gap, spec.relative_speed, spec.mode))
    started = time.perf_counter()
    trajectory_path: str | None = None
    try:
        env.reset(seed=simulation_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(1)
        result = env.episode_result()
        if trajectory_dir is not None:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            path = trajectory_dir / f"{execution_id}.npz"
            values = np.asarray(env.full_trace, dtype=float)
            np.savez_compressed(path, values=values, fields=np.asarray(TRACE_FIELDS), scenario_id=spec.scenario_id, execution_id=execution_id)
            trajectory_path = path.as_posix()
        return Observation(
            execution_id, spec.scenario_id, bool(result.collision), bool(result.near_miss),
            float(result.vulnerability), True, None, float(env.time), time.perf_counter() - started,
            float(result.min_ttc), float(result.min_distance), bool(result.completed), trajectory_path,
        )
    except Exception as exc:  # execution failures consume budget and remain auditable
        return Observation(execution_id, spec.scenario_id, False, False, 0.0, False, type(exc).__name__, float(getattr(env, "time", 0.0)), time.perf_counter() - started, float("inf"), float("inf"), False, None)
    finally:
        env.close()

