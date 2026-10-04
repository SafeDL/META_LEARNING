"""Passive physical execution for an explicitly supplied SUT build."""
from __future__ import annotations

import time

from highway_sim_env.envs.unified_env import PHYSICS_HZ, UnifiedHighwayEnv
from .risk import PassiveRiskObserver


def execute(scene, measurement, *, spec, observe=True):
    started = time.perf_counter()
    env = UnifiedHighwayEnv(spec, scene)
    observer = PassiveRiskObserver(measurement)
    try:
        env.reset(seed=scene["simulator_seed"])
        if observe:
            observer.observe(env)
        limit = int(round(float(env.config["duration"]) * PHYSICS_HZ))
        while env.steps < limit:
            if any(actor.crashed for actor in env.actors.values()):
                break
            env._advance()
            if observe:
                observer.observe(env)
        result = env.result()
        result["ego_distance_m"] = env.trace[-1]["ego"]["x_m"] - env.trace[0]["ego"]["x_m"]
        if hasattr(env.vehicle, "first_detected_s"):
            result["perception"] = {
                "first_detected_s": env.vehicle.first_detected_s,
                "first_used_s": env.vehicle.first_used_s,
                "predictive_detection_count": env.vehicle.predictive_detection_count,
                "effective_target_speed_mps": env.vehicle.profile.target_speed,
                "command_steps": len(env.vehicle.commanded_accelerations),
                "brake_clipped_steps": sum(a <= -env.vehicle.profile.max_brake + 1e-8
                                           for a in env.vehicle.commanded_accelerations),
            }
        return result, env.trace, observer.summary() if observe else {}, observer.frames, time.perf_counter() - started
    finally:
        env.close()
