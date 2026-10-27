"""From-scratch SB3 runs; qualification, never final discovery, chooses checkpoints."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.protocol import ROOT
from experiments.qualification import QUALIFICATION_ROOT, qualify
from experiments.source_registration import source_contract_matches
from simulator.highway.training_env import EPISODE_DURATIONS, DrivingEnv

from .catalog import BY_ID
from .learning import ACTION_INTERFACE, CHECKPOINT_ROOT, DiscreteDriver

TRAINING = {
    "total_timesteps": 4194304,
    "workers": 8,
    "evaluation_interval": 32768,
    "action_interface": ACTION_INTERFACE,
    "policy_network": [128, 128],
    "gamma": 0.99,
    "ppo": {
        "learning_rate": 0.0003,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    "dqn": {
        "learning_rate": 0.00005,
        "buffer_size": 200000,
        "learning_starts": 20000,
        "batch_size": 256,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 10000,
        "exploration_fraction": 0.4,
        "exploration_final_eps": 0.1,
    },
}


def training_contract():
    paths = (
        "controllers/highway/train.py",
        "controllers/highway/learning.py",
        "simulator/highway/training_env.py",
        "simulator/highway/observation.py",
        "simulator/highway/collision_env.py",
        "controllers/highway/control.py",
        "experiments/scenarios.py",
        "experiments/qualification.py",
    )
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


class ModelController:
    def __init__(self, model):
        self.model = model
        self.driver = DiscreteDriver()

    def act(self, world):
        action, _ = self.model.predict(self.driver.observation(world), deterministic=True)
        return self.driver.command(int(action), world)


class QualificationCallback(BaseCallback):
    def __init__(self, sut, root, configuration):
        super().__init__()
        self.sut = sut
        self.root = root
        self.configuration = configuration
        self.next_evaluation = configuration["evaluation_interval"]
        self.best = (-1, -1, -1, -np.inf)
        self.curve = []

    def _on_step(self):
        if self.num_timesteps >= self.next_evaluation:
            self.evaluate()
            self.next_evaluation += self.configuration["evaluation_interval"]
        return True

    def evaluate(self):
        report = qualify(lambda: ModelController(self.model))
        rows = report["courses"]
        score = (
            sum(not r["collision"] for r in rows),
            sum(not r["violations"] for r in rows),
            sum(r["passed"] for r in rows),
            sum(r["progress_m"] for r in rows),
        )
        selected = score > self.best
        if selected:
            self.best = score
            self.model.save(self.root / "best.zip")
        self.curve.append({"steps": self.num_timesteps, "selected": selected, **report})
        (self.root / "curve.json").write_text(json.dumps(self.curve, indent=2), encoding="utf-8")
        self.model.save(self.root / "latest.zip")
        print(
            self.sut.sut_id,
            self.num_timesteps,
            "qualified",
            report["qualified"],
            "selected",
            selected,
            flush=True,
        )


def train(sut_id, *, configuration=TRAINING):
    sut = BY_ID[sut_id]
    if sut.family != "learning":
        raise ValueError("Only PPO and DQN are trained")
    capability = json.loads((QUALIFICATION_ROOT / "nonlearning.json").read_text(encoding="utf-8"))
    interface = json.loads(
        (ROOT / "experiment_outputs/benchmark/collision/interface.json").read_text(encoding="utf-8")
    )
    if (
        not capability["qualified"]
        or len(capability["objects"]) != 15
        or interface["status"] != "passed"
    ):
        raise ValueError("Common interface and nonlearning capabilities must pass before training")
    if sut.role == "target":
        frozen_path = CHECKPOINT_ROOT / "training_protocol.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen["configuration"] != configuration or not source_contract_matches(
            frozen["source_contract"], training_contract()
        ):
            raise ValueError("Final target training must follow the frozen training protocol")
    root = CHECKPOINT_ROOT / sut.training_run
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    manifest = {
        "status": "running",
        "sut_id": sut.sut_id,
        "training_seed": sut.training_seed,
        "configuration": configuration,
        "object": asdict(sut),
        "source_contract": training_contract(),
    }
    (root / "training.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env = None
    try:
        env = SubprocVecEnv(
            [DrivingEnv for _ in range(configuration["workers"])], start_method="spawn"
        )
        env.seed(sut.training_seed)
        algorithm = PPO if sut.lineage == "ppo" else DQN
        model = algorithm(
            "MlpPolicy",
            env,
            seed=sut.training_seed,
            device="cpu",
            verbose=0,
            gamma=configuration["gamma"],
            policy_kwargs={"net_arch": configuration["policy_network"]},
            **configuration[sut.lineage],
        )
        callback = QualificationCallback(sut, root, configuration)
        model.learn(total_timesteps=configuration["total_timesteps"], callback=callback)
        callback.evaluate()
        best = algorithm.load(root / "best.zip", device="cpu")
        final = qualify(lambda: ModelController(best), save_root=QUALIFICATION_ROOT / sut.sut_id)
        qualification = {
            "sut_id": sut.sut_id,
            "parameters": sut.parameters,
            "training_seed": sut.training_seed,
            **final,
        }
        (QUALIFICATION_ROOT / sut.sut_id / "qualification.json").write_text(
            json.dumps(qualification, indent=2), encoding="utf-8"
        )
        manifest.update(
            status="qualified" if final["qualified"] else "unqualified",
            qualification=final,
            elapsed_s=time.perf_counter() - started,
            actual_timesteps=model.num_timesteps,
            qualification_executions=5 * (len(callback.curve) + 1),
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error), elapsed_s=time.perf_counter() - started)
        raise
    finally:
        if env is not None:
            env.close()
        (root / "training.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sut",
        nargs="?",
        default="all",
        choices=(
            "all",
            "history_ppo",
            "history_dqn",
            "development_ppo",
            "target_ppo",
            "target_dqn",
        ),
    )
    selected = parser.parse_args().sut
    if selected != "all":
        report = train(selected)
        if report["status"] != "qualified":
            raise RuntimeError(f"{selected} did not qualify; return to policy construction")
        return
    for name in ("history_ppo", "history_dqn", "development_ppo"):
        manifest = train(name)
        if manifest["status"] != "qualified":
            raise RuntimeError(f"{name} did not qualify; return to policy construction")
    freeze_training()
    for name in ("target_ppo", "target_dqn"):
        manifest = train(name)
        if manifest["status"] != "qualified":
            raise RuntimeError(f"{name} did not qualify; return to policy construction")


def freeze_training():
    evidence = []
    for name in ("history_ppo", "history_dqn", "development_ppo"):
        sut = BY_ID[name]
        report = json.loads(
            (CHECKPOINT_ROOT / sut.training_run / "training.json").read_text(encoding="utf-8")
        )
        if (
            report["status"] != "qualified"
            or report["configuration"] != TRAINING
            or not source_contract_matches(report["source_contract"], training_contract())
            or report["object"] != asdict(sut)
        ):
            raise ValueError("Historical/development runs must qualify under one training protocol")
        evidence.append(name)
    manifest = {
        "configuration": TRAINING,
        "qualification_courses": [
            "normal_following",
            "avoidable_braking",
            "full_stop",
            "legal_lane_change",
            "rear_constraint",
        ],
        "checkpoint_order": ["collision_free", "road_compliance", "qualified_courses", "progress"],
        "qualified_before_targets": evidence,
        "qualified_runs_before_targets": [BY_ID[name].training_run for name in evidence],
        "training_episode_durations_s": EPISODE_DURATIONS,
        "source_contract": training_contract(),
    }
    with (CHECKPOINT_ROOT / "training_protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()
