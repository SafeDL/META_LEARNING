"""Run SUT qualification and build the retained response bank."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr

from highway_sim_env.data.generate_anchor_bank import generate_anchor_bank
from highway_sim_env.envs.cutin_env import CutInScenario
from highway_sim_env.envs.external_cutin import ExternalCutInEnv
from sut_algorithms.highway_env.base import Policy
from sut_algorithms.highway_env.ppo_ece import PPO_CHECKPOINT, PPOPolicy
from sut_algorithms.highway_env.registry import RETAINED_SUTS, policy_factory

ROOT = Path("results/highway_replications/sut_selection")
ASSETS = PPO_CHECKPOINT.parents[1]
DEFAULT_SUTS = RETAINED_SUTS
COMMON_MODES = (
    "fast_intrusion",
    "cutin_braking",
    "lead_braking",
    "stop_and_go",
    "slow_lead_following",
    "passing_cutin",
)
ECE_NATIVE_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["x", "y", "vx", "vy", "sin_h", "cos_h"],
        "features_range": {
            "x": [-50, 50],
            "y": [-50, 50],
            "vx": [-40, 40],
            "vy": [-40, 40],
        },
        "absolute": False,
        "order": "sorted",
        "normalize": True,
        "see_behind": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 30,
    "vehicles_density": 1.5,
    "controlled_vehicles": 1,
    "duration": 40,
    "simulation_frequency": 5,
    "policy_frequency": 3,
    "collision_reward": -20.0,
    "lane_change_reward": -1.0,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.7,
    "reward_speed_range": [10.0, 30.0],
    "normalize_reward": False,
    "offroad_terminal": True,
}


def snapshot(
    output: Path = ROOT / "provenance/environment_snapshot.json",
) -> None:
    import gymnasium
    import highway_env
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "gymnasium": gymnasium.__version__,
        "highway_env": highway_env.__version__,
        "torch": torch.__version__,
        "git_head": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _command(parts: list[str]) -> str:
    return subprocess.check_output(
        parts,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def run_episode(policy: Policy, scenario: CutInScenario, seed: int) -> dict:
    env = ExternalCutInEnv(scenario, ego_kind=policy.ego_kind)
    started = time.perf_counter()
    action_history: list[int] = []
    try:
        env.reset(seed=seed)
        policy.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy.act(env)
            action_history.append(int(action))
            _, _, terminated, truncated, _ = env.step(action)
        result = asdict(env.external_result())
        result.update(
            {
                "actions": json.dumps(action_history),
                "inference_seconds": time.perf_counter() - started,
            }
        )
        return result
    finally:
        env.close()


def qualification_scenarios(count: int, seed: int) -> list[CutInScenario]:
    rng = np.random.default_rng(seed)
    return [
        CutInScenario(
            float(rng.uniform(28, 40)),
            float(rng.uniform(-1, 2)),
            "slow_lead_following",
        )
        for _ in range(count)
    ]


def common_scenarios(count: int, seed: int) -> list[CutInScenario]:
    if count < len(COMMON_MODES) or count % len(COMMON_MODES):
        raise ValueError("common scenarios require a positive multiple of six")

    per_mode = count // len(COMMON_MODES)
    anchors = np.vstack(
        [
            generate_anchor_bank(per_mode, seed + offset)
            for offset in range(len(COMMON_MODES))
        ]
    )
    modes = np.repeat(np.asarray(COMMON_MODES, dtype="U32"), per_mode)
    controls = np.random.default_rng(seed + 1).random((count, 2))
    return [
        CutInScenario(
            float(anchor[0]),
            float(anchor[1]),
            str(mode),
            float(control[0]),
            float(control[1]),
        )
        for anchor, mode, control in zip(anchors, modes, controls)
    ]


def evaluate(
    suts: Iterable[str],
    scenarios: list[CutInScenario],
    output: Path,
    seed: int,
) -> list[dict]:
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for sut in suts:
        policy = policy_factory(sut, ASSETS)
        for index, scenario in enumerate(scenarios):
            record = {
                "sut": sut,
                "scenario_index": index,
                "seed": seed + index,
                "mode": scenario.mode,
                "initial_gap": scenario.initial_gap,
                "relative_speed": scenario.relative_speed,
                "timing": scenario.timing,
                "intensity": scenario.intensity,
            }
            try:
                record.update(run_episode(policy, scenario, seed + index))
                record["error"] = ""
            except Exception as error:
                record.update(_failed_episode(error))
            records.append(record)
    _write_csv(output, records)
    return records


def _failed_episode(error: Exception) -> dict:
    return {
        "ego_collision": False,
        "background_collision": False,
        "near_miss": False,
        "completed": False,
        "min_ttc": np.nan,
        "min_distance": np.nan,
        "distance": 0.0,
        "mean_speed": 0.0,
        "steps": 0,
        "actions": "[]",
        "inference_seconds": 0.0,
        "error": f"{type(error).__name__}: {error}",
    }


def native_checkpoint_validation(
    episodes: int,
    output: Path,
    seed: int,
) -> list[dict]:
    """Validate the retained PPO checkpoint in its native environment."""
    import gymnasium as gym
    import highway_env  # noqa: F401 - registers highway-fast-v0

    policy = policy_factory("ppo_ece", ASSETS)
    if not isinstance(policy, PPOPolicy):
        raise TypeError("ppo_ece must resolve to PPOPolicy")
    policy.load()
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for episode in range(episodes):
        env = gym.make("highway-fast-v0", config=ECE_NATIVE_CONFIG)
        started = time.perf_counter()
        try:
            observation, _ = env.reset(seed=seed + episode)
            terminated = False
            truncated = False
            total_reward = 0.0
            speeds: list[float] = []
            steps = 0
            while not (terminated or truncated):
                action, _ = policy.model.predict(
                    observation,
                    deterministic=True,
                )
                observation, reward, terminated, truncated, _ = env.step(
                    int(action)
                )
                total_reward += float(reward)
                speeds.append(float(env.unwrapped.vehicle.speed))
                steps += 1
            records.append(
                {
                    "sut": "ppo_ece",
                    "episode": episode,
                    "seed": seed + episode,
                    "reward": total_reward,
                    "ego_collision": bool(env.unwrapped.vehicle.crashed),
                    "distance": float(env.unwrapped.vehicle.position[0]),
                    "mean_speed": float(np.mean(speeds)),
                    "steps": steps,
                    "inference_seconds": time.perf_counter() - started,
                    "error": "",
                }
            )
        finally:
            env.close()
    _write_csv(output, records)
    return records


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"No records to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_response_bank(records: list[dict], output: Path) -> None:
    suts = tuple(dict.fromkeys(str(record["sut"]) for record in records))

    def matrix(field: str, dtype) -> np.ndarray:
        return np.asarray(
            [
                [record[field] for record in records if record["sut"] == sut]
                for sut in suts
            ],
            dtype=dtype,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sut_names=np.asarray(suts),
        ego_collision=matrix("ego_collision", bool),
        background_collision=matrix("background_collision", bool),
        near_miss=matrix("near_miss", bool),
        completed=matrix("completed", bool),
        min_ttc=matrix("min_ttc", float),
        min_distance=matrix("min_distance", float),
        distance=matrix("distance", float),
        mean_speed=matrix("mean_speed", float),
    )


def audit(records: list[dict], output: Path) -> None:
    suts = tuple(dict.fromkeys(str(record["sut"]) for record in records))
    rows: list[dict] = []
    for index, left in enumerate(suts):
        left_records = [record for record in records if record["sut"] == left]
        left_risk = _risk_vector(left_records)
        left_failures = _failure_set(left_records)
        for right in suts[index + 1 :]:
            right_records = [
                record for record in records if record["sut"] == right
            ]
            right_risk = _risk_vector(right_records)
            right_failures = _failure_set(right_records)
            union = left_failures | right_failures
            rows.append(
                {
                    "sut_a": left,
                    "sut_b": right,
                    "spearman_risk": _spearman(left_risk, right_risk),
                    "jaccard_failures": (
                        len(left_failures & right_failures) / len(union)
                        if union
                        else "NA"
                    ),
                    "a_only_failures": len(left_failures - right_failures),
                    "b_only_failures": len(right_failures - left_failures),
                }
            )
    _write_csv(output, rows)


def _risk_vector(records: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            float(record["ego_collision"])
            + 0.5 * float(record["near_miss"])
            for record in records
        ]
    )


def _failure_set(records: list[dict]) -> set[int]:
    return {
        int(record["scenario_index"])
        for record in records
        if record["ego_collision"]
    }


def _spearman(left: np.ndarray, right: np.ndarray):
    if np.std(left) == 0 or np.std(right) == 0:
        return "NA"
    return float(spearmanr(left, right).statistic)


def audit_by_function(records: list[dict], output: Path) -> None:
    rows: list[dict] = []
    groups = _group_records(records, ("sut", "mode"))
    for (sut, mode), grouped in groups.items():
        rows.append(
            {
                "sut": sut,
                "mode": mode,
                "episodes": len(grouped),
                "ego_collisions": sum(
                    bool(record["ego_collision"]) for record in grouped
                ),
                "near_misses": sum(
                    bool(record["near_miss"]) for record in grouped
                ),
                "completed": sum(bool(record["completed"]) for record in grouped),
                "mean_speed": float(
                    np.mean([float(record["mean_speed"]) for record in grouped])
                ),
            }
        )
    _write_csv(output, rows)


def _group_records(
    records: list[dict],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict]]:
    groups: dict[tuple[str, ...], list[dict]] = {}
    for record in records:
        key = tuple(str(record[field]) for field in keys)
        groups.setdefault(key, []).append(record)
    return groups


def final_manifest(output: Path) -> tuple[str, ...]:
    entries = [
        {
            "id": "idm_mobil",
            "status": "retained",
            "family": "native IDM+MOBIL",
            "reason": "20/20 qualifying episodes complete",
        },
        {
            "id": "vi_ttc",
            "status": "retained",
            "family": "finite-MDP value iteration",
            "reason": "20/20 qualifying episodes complete",
        },
        {
            "id": "mcts_cv",
            "status": "retained",
            "family": "constant-velocity MCTS",
            "reason": "20/20 qualifying episodes complete",
        },
        {
            "id": "ppo_ece",
            "status": "retained",
            "family": "PPO",
            "reason": "17/20 native no-collision; 20/20 qualification complete",
        },
        {
            "id": "dqn_ece",
            "status": "rejected",
            "family": "DQN",
            "reason": "20/20 native attempts failed under SB3 2.3.0",
        },
        {
            "id": "ddqn_ece",
            "status": "rejected",
            "family": "Double DQN",
            "reason": "33 ego collisions; 63/96 common completion",
        },
    ]
    payload = {
        "oracle_version": "external_ego_v1",
        "retained_suts": RETAINED_SUTS,
        "screening": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return RETAINED_SUTS


def report(qualifications: list[dict], common: list[dict], output: Path) -> None:
    grouped = _group_records(qualifications, ("sut",))
    lines = [
        "# Highway-env SUT qualification report",
        "",
        "| SUT | Qualified driving | Common completed | Ego collisions | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for sut in RETAINED_SUTS:
        qualification_rows = grouped.get((sut,), [])
        common_rows = [record for record in common if record["sut"] == sut]
        qualified = sum(
            bool(record["completed"]) and not record["error"]
            for record in qualification_rows
        )
        completed = sum(
            bool(record["completed"]) and not record["error"]
            for record in common_rows
        )
        collisions = sum(
            bool(record["ego_collision"]) for record in common_rows
        )
        lines.append(
            f"| {sut} | {qualified}/{len(qualification_rows)} | "
            f"{completed}/{len(common_rows)} | "
            f"{collisions}/{len(common_rows)} | retain |"
        )
    lines.extend(
        [
            "",
            "DQN-ECE was rejected after 20 native loading failures. "
            "Double DQN-ECE was rejected after 33 ego collisions in 96 common "
            "scenarios. Their runtime integrations and checkpoints were removed; "
            "the aggregate decisions and checkpoint metadata are retained.",
            "",
            "MCTS-CV uses a constant-velocity prediction model and has no access "
            "to the scheduled future traffic state.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
