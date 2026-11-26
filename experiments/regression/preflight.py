"""Verify policy ancestry and execute the registered capability gate."""

import hashlib
import time

import torch
from stable_baselines3 import DQN, PPO

from controllers.highway.train import ModelController

from experiments.protocol import ROOT
from experiments.qualification import course_contract, qualify
from experiments.storage import read_json, write_json

RUN_ROOT = ROOT / "results/policy_regression"
CHECKPOINTS = {
    "ancestor": ROOT / "controllers/highway/checkpoints/collision/independent_ppo_persistent_speed/best.zip",
    "parent": ROOT / "controllers/highway/checkpoints/collision/target_ppo_native/best.zip",
}
DQN_CONTINUAL = "dqn_continual"
POLICIES = {"ppo": PPO, "dqn": DQN, DQN_CONTINUAL: DQN}
DQN_CHECKPOINTS = {
    "ancestor": ROOT / "controllers/highway/checkpoints/collision/independent_dqn_persistent_speed/best.zip",
    "parent": ROOT / "controllers/highway/checkpoints/collision/target_dqn_native/best.zip",
}


def preflight(policy="ppo"):
    if policy == DQN_CONTINUAL:
        from experiments.regression.dqn_evaluation import admission
        from experiments.regression.dqn_learning import STUDY

        admission()
        return {"status": "passed", "results": {"parent": read_json(STUDY / "base/confirmation.json")["capability"]}}
    torch.set_num_threads(1)
    checkpoints = CHECKPOINTS if policy == "ppo" else DQN_CHECKPOINTS
    root = RUN_ROOT
    identities = {
        name: {"path": path.relative_to(ROOT).as_posix(),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in checkpoints.items()
    }
    training = read_json(checkpoints["parent"].parent / "training.json")
    if training["initial_checkpoint"]["sha256"] != identities["ancestor"]["sha256"]:
        raise ValueError("Registered parent is not descended from the registered ancestor")
    specification = {"checkpoints": identities, "capability": course_contract()}
    destination = root / "preflight.json"
    if destination.exists():
        saved = read_json(destination)
        if saved["specification"] != specification:
            raise ValueError("Preflight inputs changed after execution")
        return saved
    rows = {}
    for name, path in checkpoints.items():
        started = time.perf_counter()
        model = POLICIES[policy].load(path, device="cpu")
        report = qualify(
            lambda: ModelController(model),
            save_root=root / "qualification" / name,
        )
        rows[name] = {**report, "wall_s": time.perf_counter() - started,
                      "physical_executions": len(report["courses"])}
        print(name, "qualified", report["qualified"],
              "failed", [r["course"] for r in report["courses"] if not r["passed"]],
              flush=True)
    saved = {"specification": specification, "results": rows,
             "status": "passed" if all(r["qualified"] for r in rows.values()) else "failed",
             "physical_executions": sum(r["physical_executions"] for r in rows.values())}
    write_json(destination, saved)
    return saved


if __name__ == "__main__":
    result = preflight()
    print("regression preflight", result["status"], flush=True)
