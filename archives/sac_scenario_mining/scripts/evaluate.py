"""Evaluate Random or SAC on the same explicit held-out logical cases."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..src.casebook import build_case_table
from ..src.env import Stage1AdversarialMergeEnv
from ..src.metrics import summarize
from ..src.scenario_manifest import save_manifest
from ..src.utils import dump_json, load_config, write_csv


def evaluate(cfg: dict, policy: str, policy_path: str | None, split: str,
             episodes: int | None, seed: int, deterministic: bool, output: Path,
             env: Stage1AdversarialMergeEnv | None = None,
             save_case_table: bool = True) -> dict:
    cases = build_case_table(cfg, split)
    if episodes is not None and episodes != len(cases):
        raise ValueError(f"{split} has {len(cases)} fixed cases; --episodes must be omitted or equal to that count")
    output.mkdir(parents=True, exist_ok=True)
    if save_case_table:
        dump_json(output / "case_table.json", cases)
    run_env = env or Stage1AdversarialMergeEnv(cfg, split=split, seed=seed)
    run_env.action_space.seed(seed)
    model = None
    if policy_path:
        from stable_baselines3 import SAC
        # Rollout is inherently one action at a time; CPU avoids thousands of
        # tiny GPU launches while leaving all gradient updates on the GPU.
        model = SAC.load(policy_path, device="cpu")
    records, candidates = [], []
    try:
        for case in cases:
            observation, reset_info = run_env.reset(options={"case": case})
            actions, done = [], False
            while not done:
                if model is not None:
                    action, _ = model.predict(observation, deterministic=deterministic)
                elif policy == "random":
                    action = run_env.action_space.sample()
                elif policy == "zero":
                    action = np.zeros(2, dtype=np.float32)
                else:
                    raise ValueError("policy must be random, zero, or provide --policy-path")
                observation, _, terminated, truncated, _ = run_env.step(action)
                actions.append(action)
                done = terminated or truncated
            record = run_env.episode_record()
            metrics = {key: value for key, value in record.items() if key != "theta"}
            records.append(metrics)
            if record["valid_critical"]:
                candidates.append((record["min_ttc"], metrics, np.asarray(actions, dtype=np.float32), case,
                                   reset_info))
        write_csv(output / "episodes.csv", records)
        summary = summarize(records)
        summary.update(split=split, policy_name="sac_best" if policy_path else policy, policy_seed=seed)
        if save_case_table:
            summary["case_table_file"] = "case_table.json"
        dump_json(output / "summary.json", summary)
        for rank, (_, metrics, actions, case, reset_info) in enumerate(
                sorted(candidates, key=lambda item: item[0])[:int(cfg["evaluation"]["top_k_scenarios"])], 1):
            manifest = {
                "topology": "on_ramp_merge", "case": case, "policy_name": summary["policy_name"],
                "policy_seed": seed, "adversary_id": reset_info["adversary_id"],
                "sut_id": reset_info["sut_id"], "termination_reason": metrics["termination_reason"],
                "environment": cfg["environment"],
            }
            save_manifest(output / "critical_scenarios" / f"rank_{rank:03d}", manifest, actions, metrics)
        return summary
    finally:
        if env is None:
            run_env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="archives/sac_scenario_mining/configs/merge_sac.yaml")
    parser.add_argument("--policy", default="random", choices=["random", "zero"])
    parser.add_argument("--policy-path")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="results/sac_scenario_mining/eval")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(evaluate(cfg, args.policy, args.policy_path, args.split, args.episodes, args.seed,
                   args.deterministic, Path(args.output)))


if __name__ == "__main__":
    main()
