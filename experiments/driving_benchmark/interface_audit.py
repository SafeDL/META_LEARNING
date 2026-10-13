"""Check controller ownership, PPO spaces and deterministic physical replays."""
from dataclasses import asdict
import time

import numpy as np

from highway_sim_env.envs.unified_env import PHYSICS_HZ, UnifiedHighwayEnv
from methods.history_guided_testing.io import write_json
from sut_algorithms.highway_env.ppo_ece import PPO_CHECKPOINT, PPOPolicy

from .config import AUDIT_INDICES, OUTPUT, SUT_IDS, sut_spec
from .pools import scene_pool
from .prepare import measure, protocol


def check_interface(sut_id, scene):
    spec = sut_spec(sut_id)
    env = UnifiedHighwayEnv(spec, scene)
    try:
        observation, _ = env.reset(seed=scene["simulator_seed"])
        if env.vehicle.speed != 25 or len(env.actors) != 2:
            raise AssertionError("Initial speed/actor layout differs from the shared scene")
        if PHYSICS_HZ != 20 or env.config["policy_frequency"] != spec.control_hz:
            raise AssertionError("Control/physics rate differs from the declared build")
        if sut_id == "ppo_ref_v2":
            policy = PPOPolicy(PPO_CHECKPOINT)
            policy.load()
            if policy.model.observation_space != env.observation_space:
                raise AssertionError("PPO checkpoint and runner observation spaces differ")
            if policy.model.action_space != env.action_space:
                raise AssertionError("PPO checkpoint and runner action spaces differ")
            if not policy.model.observation_space.contains(observation):
                raise AssertionError("PPO observation is outside its checkpoint space")
        env._advance()
        if spec.adapter_kind == "external_meta_policy":
            if len(env.control_actions) != 1:
                raise AssertionError("External controller did not own the first policy tick")
        runtime = env.result()["ego_runtime"]
        return {
            "sut_id": sut_id, "build": asdict(spec),
            "observation_shape": list(np.shape(observation)),
            "observation_features": env.config["observation"]["features"],
            "action_names": env.action_type.actions,
            "action_target_speeds_mps": np.asarray(env.action_type.target_speeds).tolist(),
            "vehicle_target_speed_mps": float(env.vehicle.target_speed),
            "ego_runtime": runtime,
        }
    finally:
        env.close()


def main():
    started = time.perf_counter()
    scenes = scene_pool(0)
    interfaces, rows, replays = [], [], []
    for sut_id in SUT_IDS:
        interfaces.append(check_interface(sut_id, scenes[0]))
        for index in AUDIT_INDICES:
            row = measure((sut_id, scenes[index]))
            rows.append({"sut_id": sut_id, "index": index, **row})
        first = next(row for row in rows if row["sut_id"] == sut_id)
        replay = measure((sut_id, scenes[first["index"]]))
        fields = ("risk", "collision", "risk_components", "collision_time_s", "event_times")
        if any(first[key] != replay[key] for key in fields):
            raise AssertionError(f"Deterministic replay mismatch: {sut_id}")
        replays.append({"sut_id": sut_id, "index": first["index"], "matched": True})
        print("INTERFACE PASS", sut_id, "collision_count",
              sum(row["collision"] for row in rows if row["sut_id"] == sut_id), flush=True)
    write_json(OUTPUT / "interface_audit.json", {
        "passed": True, "interfaces": interfaces, "measurements": rows,
        "replays": replays, "physical_full_episode_calls": len(rows) + len(replays),
        "one_step_interface_checks": len(interfaces),
        "elapsed_s": time.perf_counter() - started,
        "scope": "Interface and replay audit, not full-pool failure prevalence or performance",
    })
    write_json(OUTPUT / "protocol.json", protocol())


if __name__ == "__main__":
    main()
