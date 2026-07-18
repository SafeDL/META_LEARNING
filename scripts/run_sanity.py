#!/usr/bin/env python
"""
M0 Sanity check — runs R001 and R002 from EXPERIMENT_TRACKER.md.

R001: env smoke-test  (10 steps in each topology, verify obs/reward shapes)
R002: PPO random-init 1k steps on roundabout (verify training loop, shapes, save)

Usage:
  python scripts/run_sanity.py --config configs/default.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_r001_env_check(cfg: dict) -> bool:
    """Verify environment builds and returns correct shapes for all topologies."""
    from src.env_wrapper import make_env, TOPOLOGY_MAP

    print("\n[R001] Environment smoke-test")
    all_pass = True
    expected_obs = cfg["env"]["obs_dim"]
    expected_act = cfg["env"]["act_dim"]

    for topo in TOPOLOGY_MAP:
        try:
            env = make_env(topo, seed=0, num_scenarios=cfg["env"]["num_scenarios"])
            obs, info = env.reset()
            assert obs.shape == (expected_obs,), \
                f"Obs shape mismatch: got {obs.shape}, expected ({expected_obs},)"

            action = env.action_space.sample()
            assert action.shape == (expected_act,), \
                f"Action shape mismatch: got {action.shape}, expected ({expected_act},)"

            next_obs, reward, done, _, _ = env.step(action)
            assert next_obs.shape == (expected_obs,), "Next obs shape mismatch"
            assert isinstance(reward, (float, int, np.floating)), \
                f"Reward is not scalar: {type(reward)}"
            env.close()
            print(f"  [{topo}]  obs={obs.shape}  reward={reward:.3f}  PASS")
        except Exception as e:
            print(f"  [{topo}]  FAIL: {e}")
            all_pass = False

    return all_pass


def run_r002_ppo_smoke(cfg: dict, out_dir: str) -> bool:
    """Run PPO for 1k steps on roundabout with random init; verify results.json written."""
    from src.adapt_ppo import run_ppo_adapt

    print("\n[R002] PPO smoke-test (random-init, roundabout, 1k steps)")
    smoke_cfg = dict(cfg)
    # Override total timesteps for speed
    smoke_cfg = yaml.safe_load(yaml.dump(cfg))  # deep copy
    smoke_cfg["ppo"]["total_timesteps"] = 1024

    try:
        result = run_ppo_adapt(
            topology    = "roundabout",
            init_mode   = "random",
            encoder_ckpt= None,
            total_steps = 1024,
            seed        = 0,
            cfg         = smoke_cfg,
            out_dir     = out_dir,
            device      = "cuda" if torch.cuda.is_available() else "cpu",
        )
        assert os.path.exists(os.path.join(out_dir, "results.json")), \
            "results.json not written"
        assert "return_auc" in result, "return_auc missing from result"
        print(f"  return_auc={result['return_auc']:.3f}  "
              f"episodes={result['total_episodes']}  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out_dir", default="results/sanity")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    r001 = run_r001_env_check(cfg)
    r002 = run_r002_ppo_smoke(cfg, os.path.join(args.out_dir, "r002_ppo_smoke"))

    summary = {"R001_env_check": r001, "R002_ppo_smoke": r002}
    with open(os.path.join(args.out_dir, "sanity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    passed = r001 and r002
    print(f"\n{'='*50}")
    print(f"M0 SANITY: {'PASS' if passed else 'FAIL'}")
    print(f"  R001 env_check : {'PASS' if r001 else 'FAIL'}")
    print(f"  R002 ppo_smoke : {'PASS' if r002 else 'FAIL'}")
    print(f"{'='*50}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
