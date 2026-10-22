"""Construction-only action-interface diagnostic; never a registered SUT checkpoint."""

import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from benchmark.qualification import qualify
from controllers.highway.control import Control
from controllers.highway.learning import DiscreteDriver
from controllers.highway.train import TRAINING, training_contract
from simulation.highway.training_env import DrivingEnv


ROOT = Path(__file__).resolve().parent
SETTINGS = {"seed": 4117, "total_timesteps": 131072, "evaluation_interval": 32768,
            "action_order": ["left", "keep", "right", "faster", "slower"],
            "speed_increment_mps": 5.0, "speed_gain_per_s": 2.0,
            "maximum_target_speed_mps": 35.0}


class PersistentSpeedDriver(DiscreteDriver):
    def __init__(self):
        super().__init__()
        self.target_speed = None

    def observation(self, world):
        if self.target_speed is None:
            self.target_speed = world.ego.speed
        return np.append(super().observation(world), self.target_speed / 40).astype(np.float32)

    def command(self, action, world):
        if not 0 <= int(action) < 5 or int(action) != action:
            raise ValueError("Construction probe requires one of its five declared actions")
        self.observation(world)
        if action in (3, 4):
            self.target_speed = float(np.clip(self.target_speed + (5 if action == 3 else -5), 0, 35))
        lateral = int(action) if action < 3 else 1
        steering = super().command(3 + lateral, world).steering
        return Control(float(np.clip(2 * (self.target_speed - world.ego.speed), -8, 3)), steering)


class ProbeEnv(DrivingEnv):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(-1, 1, (83,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.driver = PersistentSpeedDriver()
        return self.driver.observation(self.world.visible_world()), {}


class ProbeController:
    def __init__(self, model):
        self.model, self.driver = model, PersistentSpeedDriver()

    def act(self, world):
        action, _ = self.model.predict(self.driver.observation(world), deterministic=True)
        return self.driver.command(int(action), world)


class ProbeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.next_evaluation = SETTINGS["evaluation_interval"]
        self.best = (-1, -1, -1, -np.inf)
        self.curve = []

    def _on_step(self):
        if self.num_timesteps >= self.next_evaluation:
            self.evaluate()
            self.next_evaluation += SETTINGS["evaluation_interval"]
        return True

    def evaluate(self):
        report = qualify(lambda: ProbeController(self.model))
        rows = report["courses"]
        score = (sum(not r["collision"] for r in rows), sum(not r["violations"] for r in rows),
                 sum(r["passed"] for r in rows), sum(r["progress_m"] for r in rows))
        selected = score > self.best
        if selected:
            self.best = score
            self.model.save(ROOT / "best.zip")
        self.curve.append({"steps": self.num_timesteps, "selected": selected, **report})
        (ROOT / "curve.json").write_text(json.dumps(self.curve, indent=2), encoding="utf-8")
        print(self.num_timesteps, "qualified", report["qualified"], "selected", selected, flush=True)


def main():
    started = time.perf_counter()
    registration = {"use": "construction_probe_only", "settings": SETTINGS,
                    "common_optimizer_and_distribution": TRAINING,
                    "source_contract": training_contract(),
                    "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "eligibility": "Not eligible as a registered SUT; truncated construction budget."}
    with (ROOT / "registration.json").open("x", encoding="utf-8") as stream:
        json.dump(registration, stream, indent=2)
    torch.set_num_threads(1)
    environment = None
    manifest = {"status": "running", **registration}
    (ROOT / "training.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        environment = SubprocVecEnv([ProbeEnv for _ in range(TRAINING["workers"])], start_method="spawn")
        environment.seed(SETTINGS["seed"])
        model = PPO("MlpPolicy", environment, seed=SETTINGS["seed"], device="cpu", verbose=0,
                    gamma=TRAINING["gamma"], policy_kwargs={"net_arch": TRAINING["policy_network"]},
                    **TRAINING["ppo"])
        callback = ProbeCallback()
        model.learn(SETTINGS["total_timesteps"], callback=callback)
        best = PPO.load(ROOT / "best.zip", device="cpu")
        replay = qualify(lambda: ProbeController(best), save_root=ROOT / "qualification")
        manifest.update(status="complete", qualification=replay, actual_timesteps=model.num_timesteps,
                        qualification_executions=5 * (len(callback.curve) + 1))
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if environment is not None:
            environment.close()
        manifest["elapsed_s"] = time.perf_counter() - started
        (ROOT / "training.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
