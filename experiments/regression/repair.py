"""Continue a policy on the paid reset curriculum and retain the fixed final step."""

import functools
import random
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from controllers.highway.train import ModelController, TRAINING
from simulator.highway.repair_env import RepairEnv

from experiments.parallel import prepare_processes
from experiments.qualification import qualify
from experiments.regression.execution import checkpoint_identity
from experiments.regression.preflight import POLICIES
from experiments.regression.protocol import DQN_DECAY, DQN_SEED, DQN_WARMUP, TRAINING_STEPS
from experiments.storage import read_json, write_json


class RolloutProgress(BaseCallback):
    def __init__(self, start):
        super().__init__()
        self.start = start

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        print("PPO fresh rollout", (self.num_timesteps - self.start) // 8192,
              "/ 16", flush=True)


def repair(parent, current, historical, excluded, destination, *, preparation=False,
           parent_random_state=None, iteration=None, policy="ppo", qualify_endpoint=True):
    destination = Path(destination)
    label = "repair" if iteration is None else f"repair_{iteration}"
    report_path = destination / f"{label}.json"
    policy_name = "policy.zip" if iteration is None else f"{label}.zip"
    random_name = "random_state.pt" if iteration is None else f"{label}_random.pt"
    specification = {
        "parent_sha256": checkpoint_identity(parent),
        "current": [asdict(s) for s in current], "historical": [asdict(s) for s in historical],
        "excluded_count": len(excluded), "preparation": preparation,
        "steps": TRAINING_STEPS, policy: TRAINING[policy], "workers": 8,
    }
    if policy == "dqn":
        specification[policy] = {**TRAINING[policy], "gamma": 0.99,
                                 "warmup": DQN_WARMUP if preparation else 0,
                                 "decay": DQN_DECAY if preparation else 0}
    if not qualify_endpoint:
        specification["qualification_pending"] = True
    if report_path.exists():
        saved = read_json(report_path)
        if saved["specification"] != specification:
            raise ValueError("Saved repair uses different feedback or training settings")
        if saved["status"] == "complete":
            return saved
        raise RuntimeError(f"An interrupted {policy.upper()} repair requires audit before retry")
    torch.set_num_threads(1)
    prepare_processes()
    destination.mkdir(parents=True, exist_ok=True)
    write_json(report_path, {"status": "running", "specification": specification})
    env = SubprocVecEnv([
        functools.partial(RepairEnv, current, historical, excluded, preparation=preparation)
        for _ in range(8)
    ], start_method="spawn")
    started = time.perf_counter()
    try:
        seed = DQN_SEED if policy == "dqn" else 6113
        env.seed(seed)
        model = POLICIES[policy].load(parent, env=env, device="cpu", **TRAINING[policy])
        if policy == "dqn":
            env.seed(seed)
        start = model.num_timesteps
        epochs_before = model._n_updates
        if parent_random_state is None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:
            state = torch.load(parent_random_state, map_location="cpu", weights_only=False)
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            # Discard queued initial seeds so restoring RNG is effective at reset.
            env._reset_seeds()
            for worker, generator in enumerate(state["environments"]):
                env.env_method("restore_random_state", generator, indices=worker)
        continuation = {}
        if policy == "dqn":
            warmup = DQN_WARMUP if preparation else 0
            initial_epsilon = model.exploration_rate
            if parent_random_state is not None:
                model.load_replay_buffer(parent_random_state.with_suffix(".pkl"))
                model.action_space.np_random.bit_generator.state = state["actions"]
            buffer_start = (model.replay_buffer.pos, model.replay_buffer.full)
            target_start = model._n_calls
            model.learning_starts = start + warmup
            decay = DQN_DECAY if preparation else 0

            def exploration(_):
                elapsed = model.num_timesteps - start - warmup
                fraction = min(1.0, max(0, elapsed) / decay) if decay else 1.0
                return initial_epsilon + fraction * (0.05 - initial_epsilon)

            model.exploration_schedule = exploration
            model.learn(total_timesteps=TRAINING_STEPS, reset_num_timesteps=False)
            # SB3 will reconstruct the schedule on loading. Save its endpoint and
            # the replay buffer separately from the weights and Adam state.
            del model.exploration_schedule
            replay_path = destination / Path(random_name).with_suffix(".pkl")
            model.save_replay_buffer(replay_path)
            continuation = {
                "warmup": warmup, "initial_epsilon": initial_epsilon,
                "final_epsilon": model.exploration_rate,
                "buffer_start": list(buffer_start),
                "buffer_end": [model.replay_buffer.pos, model.replay_buffer.full],
                "buffer_sha256": checkpoint_identity(replay_path),
                "target_counter_start": target_start, "target_counter_end": model._n_calls,
            }
        else:
            model.learn(total_timesteps=TRAINING_STEPS, reset_num_timesteps=False,
                        callback=RolloutProgress(start))
        model.save(destination / policy_name)
        states = env.env_method("curriculum_state")
        random_state = {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(), "environments": [s["rng"] for s in states]}
        if policy == "dqn":
            random_state["actions"] = model.action_space.np_random.bit_generator.state
        torch.save(random_state, destination / random_name)
        capability = None
        if qualify_endpoint:
            capability = capture_capability(model, destination, label)
        saved = {
            "status": "complete", "specification": specification,
            "checkpoint_sha256": checkpoint_identity(destination / policy_name),
            "checkpoint_name": policy_name, "random_state_name": random_name,
            "actual_environment_steps": model.num_timesteps - start,
            "cumulative_environment_steps": model.num_timesteps,
            "optimization_epochs": model._n_updates - epochs_before,
            "gradient_updates": (model._n_updates - epochs_before) * (1 if policy == "dqn" else 32),
            "curriculum": states, "capability": capability,
            "qualification_executions": len(capability["courses"]) if capability else 0,
            "wall_s": time.perf_counter() - started,
        }
        if policy == "dqn":
            saved["continuation"] = continuation
        if saved["actual_environment_steps"] != TRAINING_STEPS:
            raise ValueError("Policy repair missed the registered step budget")
        write_json(report_path, saved)
        return saved
    except Exception as error:
        write_json(report_path, {"status": "failed", "specification": specification,
                                 "error": repr(error), "wall_s": time.perf_counter() - started})
        raise
    finally:
        env.close()


def capture_capability(model, directory, label):
    with tempfile.TemporaryDirectory(prefix="qualification_", dir=directory) as temporary:
        capability = qualify(lambda: ModelController(model), save_root=temporary)
        evidence = {path.name: read_json(path) for path in Path(temporary).iterdir()}
    write_json(directory / f"{label}_qualification.json.gz", evidence)
    return capability
