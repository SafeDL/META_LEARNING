"""Frozen scenario-resampling PPO weight-inheritance pilot for FBRT.

Training never reads the final confirmation bank. V0 adapts an existing PPO
checkpoint; V1 and V2 continue its weights on new plus replayed old scenario
definitions, collecting fresh on-policy rollouts at every stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stable_baselines3 import PPO

from highway_sim_env.envs.fbrt_training_env import FBRTTrainingEnv, TRAINING_CONTRACT
from methods.failure_memory_regression.bidirectional import compile_manifest
from sut_algorithms.highway_env.ppo_ece import PPO_CHECKPOINT


ROOT = Path("results/method_chains/failure_memory_regression/ppo_release")
CHECKPOINTS = ROOT / "checkpoints"
SEED = 20260927
STAGES = ("ppo_release_v0", "ppo_release_v1", "ppo_release_v2")
REQUESTED_STEPS = (2048, 4096, 4096)
TRAIN_FILES = ("train_v0.jsonl", "train_v1_new.jsonl", "train_v2_new.jsonl")
VALIDATION_FILE = "validation.jsonl"
CONFIRMATION_FILE = "confirmation/scenario_manifest.jsonl"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _frozen_write(path: Path, value: object, json_lines: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if json_lines:
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False) + "\n" for row in value)
    else:
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                             allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"frozen file differs: {path}")
    path.write_text(payload, encoding="utf-8")


def freeze(root: Path = ROOT) -> dict:
    if not PPO_CHECKPOINT.is_file():
        raise FileNotFoundError(PPO_CHECKPOINT)
    manifests = {
        TRAIN_FILES[0]: compile_manifest("development", resolution=5),
        TRAIN_FILES[1]: compile_manifest("development2", resolution=5),
        TRAIN_FILES[2]: compile_manifest("development3", resolution=5),
        VALIDATION_FILE: compile_manifest("calibration", resolution=5),
        CONFIRMATION_FILE: compile_manifest("ppo_confirmation", resolution=5),
    }
    for name, rows in manifests.items():
        _frozen_write(root / name, rows, json_lines=True)
    plan = {
        "version": "ppo-weight-inheritance-pilot-v1",
        "base_checkpoint": str(PPO_CHECKPOINT),
        "base_checkpoint_sha256": _digest(PPO_CHECKPOINT),
        "stages": STAGES,
        "requested_decision_steps": REQUESTED_STEPS,
        "rollout_decision_steps": 1024,
        "batch_size": 256,
        "train_epochs_per_rollout": 5,
        "learning_rate": 0.0003,
        "clip_range": 0.2,
        "seed": SEED,
        "training_contract": TRAINING_CONTRACT,
        "stage_pools": {
            STAGES[0]: {TRAIN_FILES[0]: 1},
            STAGES[1]: {TRAIN_FILES[0]: 1, TRAIN_FILES[1]: 1},
            STAGES[2]: {TRAIN_FILES[0]: 1, TRAIN_FILES[1]: 1, TRAIN_FILES[2]: 2},
        },
        "checkpoint_selection": "the last checkpoint after each fixed step budget; validation is diagnostic only",
        "manifest_sha256": {name: _digest(root / name) for name in manifests},
        "confirmation_outcomes_excluded_from_training": True,
    }
    _frozen_write(root / "study_plan.json", plan)
    return {"train_scenes_per_new_stage": [len(manifests[name]) for name in TRAIN_FILES],
            "validation_scenes": len(manifests[VALIDATION_FILE]),
            "confirmation_scenes": len(manifests[CONFIRMATION_FILE])}


def _plan(root: Path) -> dict:
    plan = json.loads((root / "study_plan.json").read_text(encoding="utf-8"))
    if _digest(Path(plan["base_checkpoint"])) != plan["base_checkpoint_sha256"]:
        raise ValueError("base PPO checkpoint changed after study freeze")
    for name, expected in plan["manifest_sha256"].items():
        if _digest(root / name) != expected:
            raise ValueError(f"PPO study manifest changed: {name}")
    if plan["training_contract"] != TRAINING_CONTRACT:
        raise ValueError("PPO training physics/reward contract changed")
    return plan


def _weight_hash(model: PPO) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_stage(stage: str, root: Path = ROOT) -> dict:
    plan = _plan(root)
    if stage not in STAGES:
        raise ValueError(stage)
    index = STAGES.index(stage)
    source = (Path(plan["base_checkpoint"]) if index == 0 else
              root / "checkpoints" / f"{STAGES[index - 1]}.zip")
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = root / "checkpoints" / f"{stage}.zip"
    metadata_path = root / "training" / f"{stage}.json"
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("stage checkpoint or ledger already exists; do not overwrite")
    cases = [case for name, count in plan["stage_pools"][stage].items()
             for _ in range(count) for case in _jsonl(root / name)]
    parent_build = "ppo_ref_v2" if index == 0 else STAGES[index - 1]
    env = FBRTTrainingEnv(cases, build_id=parent_build)
    try:
        overrides = {"n_steps": int(plan["rollout_decision_steps"]),
                     "batch_size": int(plan["batch_size"]),
                     "n_epochs": int(plan["train_epochs_per_rollout"]),
                     "lr_schedule": lambda _: float(plan["learning_rate"]),
                     "clip_range": lambda _: float(plan["clip_range"])}
        model = PPO.load(source, env=env, device="cpu", custom_objects=overrides)
        # The imported checkpoint records a TensorBoard path, but this
        # environment does not include tensorboard. Keep the numerical model
        # and disable only that optional logger before continuing training.
        model.tensorboard_log = None
        model.set_random_seed(int(plan["seed"]) + index)
        before_steps = int(model.num_timesteps)
        before_weights = _weight_hash(model)
        model.learn(total_timesteps=int(plan["requested_decision_steps"][index]),
                    reset_num_timesteps=False, progress_bar=False)
        after_weights = _weight_hash(model)
        if before_weights == after_weights:
            raise AssertionError("PPO stage did not change policy weights")
        actual_steps = int(model.num_timesteps) - before_steps
        if actual_steps != env.total_decision_steps:
            raise AssertionError("PPO rollout and physical decision counts disagree")
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.save(destination)
        metadata = {
            "stage": stage, "parent": parent_build,
            "source_checkpoint": str(source), "source_sha256": _digest(source),
            "checkpoint": str(destination), "checkpoint_sha256": _digest(destination),
            "policy_weights_before_sha256": before_weights,
            "policy_weights_after_sha256": after_weights,
            "requested_decision_steps": int(plan["requested_decision_steps"][index]),
            "actual_decision_steps": actual_steps,
            "actual_physics_steps": env.total_physics_steps,
            "training_pool_size": len(cases),
            "training_pool_manifest_sha256": {
                name: plan["manifest_sha256"][name]
                for name in plan["stage_pools"][stage]},
            "training_contract": TRAINING_CONTRACT,
        }
        _frozen_write(metadata_path, metadata)
        return metadata
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "train"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    result = freeze(args.root) if args.action == "freeze" else train_stage(
        args.stage or STAGES[0], args.root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
