"""Evaluate random, zero, or Stable-Baselines3 SAC on a held-out split."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ..src.env import Stage1AdversarialMergeEnv
from ..src.metrics import summarize
from ..src.scenario_manifest import save_manifest
from ..src.utils import dump_json, load_config, write_csv


def evaluate(
    cfg: dict,
    policy: str,
    policy_path: str | None,
    split: str,
    episodes: int,
    seed: int,
    deterministic: bool,
    output: Path,
) -> dict:
    env = Stage1AdversarialMergeEnv(cfg, split=split, seed=seed)
    env.action_space.seed(seed)
    model = None
    if policy_path:
        from stable_baselines3 import SAC

        model = SAC.load(policy_path, device="auto")
    records, candidates = [], []
    try:
        for index in range(episodes):
            observation, reset_info = env.reset(seed=seed + index)
            actions, done = [], False
            while not done:
                if model is not None:
                    action, _ = model.predict(observation,
                                              deterministic=deterministic)
                elif policy == "random":
                    action = env.action_space.sample()
                elif policy == "zero":
                    action = np.zeros(2, dtype=np.float32)
                else:
                    raise ValueError(
                        "policy must be random, zero, or --policy-path must be supplied"
                    )
                observation, _, terminated, truncated, _ = env.step(action)
                actions.append(action)
                done = terminated or truncated

            record = env.episode_record()
            record.update(
                episode_index=index,
                adversary_id=reset_info["adversary_id"],
                sut_id=reset_info["sut_id"],
            )
            records.append(record)
            if record["valid_critical"]:
                candidates.append((record["min_ttc"], record,
                                   np.asarray(actions, dtype=np.float32)))

        write_csv(output / "episodes.csv", records)
        summary = summarize(records)
        dump_json(output / "summary.json", summary)
        policy_name = "sac_best" if policy_path else policy
        top_k = int(cfg["evaluation"]["top_k_scenarios"])
        for rank, (_, record, actions) in enumerate(
                sorted(candidates, key=lambda item: item[0])[:top_k], 1):
            manifest = {
                "observation_schema": cfg["environment"]["observation_schema"],
                "topology": "merge",
                "scenario_source": "procedural",
                "scenario_seed": record["scenario_seed"],
                "policy_name": policy_name,
                "policy_seed": seed,
                "adversary_id": record["adversary_id"],
                "sut_id": record["sut_id"],
                "env_config": cfg["environment"],
                "termination_reason": record["termination_reason"],
            }
            save_manifest(output / "critical_scenarios" / f"rank_{rank:03d}",
                          manifest, actions, record)
        return summary
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="sac_scenario_mining/configs/merge_sac.yaml")
    parser.add_argument("--policy",
                        default="random",
                        choices=["random", "zero"])
    parser.add_argument("--policy-path")
    parser.add_argument("--split",
                        default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="results/sac_scenario_mining/eval")
    args = parser.parse_args()
    cfg = load_config(args.config)
    episodes = args.episodes or int(cfg["evaluation"]["episodes_per_policy"])
    print(
        evaluate(
            cfg,
            args.policy,
            args.policy_path,
            args.split,
            episodes,
            args.seed,
            args.deterministic,
            Path(args.output),
        ))


if __name__ == "__main__":
    main()
