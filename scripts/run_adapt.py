#!/usr/bin/env python
"""
B1 main adaptation grid — runs a single (algorithm, topology, init_mode, seed) cell.

To sweep the full grid, call this script in parallel via screen/tmux.
The EXPERIMENT_TRACKER.md contains all R010-R054 run IDs.

Usage:
  python scripts/run_adapt.py \\
      --algo ppo \\
      --topology roundabout \\
      --init_mode maml \\
      --encoder_ckpt results/meta_train/encoder_final.pt \\
      --seed 0 \\
      --config configs/default.yaml

  # scratch baseline (no encoder ckpt needed)
  python scripts/run_adapt.py --algo sac --topology roundabout --init_mode random --seed 0

Algorithms: ppo | sac | td3
Init modes: maml | joint | random
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",         required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--topology",     required=True)
    parser.add_argument("--init_mode",    required=True, choices=["maml", "joint", "random", "highway"])
    parser.add_argument("--encoder_ckpt", default=None,
                        help="Path to encoder .pt (required for maml/joint)")
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--total_steps",  type=int, default=None,
                        help="Override total_timesteps from config")
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--out_root",     default="results")
    parser.add_argument("--out_suffix", default="",
                        help="Optional suffix to append to the output directory name")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.total_steps is not None:
        cfg["ppo"]["total_timesteps"] = args.total_steps
        cfg["sac"]["total_timesteps"] = args.total_steps
        cfg["td3"]["total_timesteps"] = args.total_steps

    if args.init_mode in ("maml", "joint", "highway") and args.encoder_ckpt is None:
        parser.error(f"--encoder_ckpt is required for init_mode={args.init_mode}")

    out_dir = os.path.join(
        args.out_root,
        args.algo,
        f"{args.topology}_{args.init_mode}{args.out_suffix}_seed{args.seed}",
    )

    print(f"\n[run_adapt] algo={args.algo}  topo={args.topology}  "
          f"init={args.init_mode}  seed={args.seed}  device={args.device}")
    print(f"  out_dir: {out_dir}")

    if args.algo == "ppo":
        from src.adapt_ppo import run_ppo_adapt
        run_ppo_adapt(
            topology     = args.topology,
            init_mode    = args.init_mode,
            encoder_ckpt = args.encoder_ckpt,
            total_steps  = cfg["ppo"]["total_timesteps"],
            seed         = args.seed,
            cfg          = cfg,
            out_dir      = out_dir,
            device       = args.device,
        )
    elif args.algo == "sac":
        from src.adapt_sac import run_sac_adapt
        run_sac_adapt(
            topology     = args.topology,
            init_mode    = args.init_mode,
            encoder_ckpt = args.encoder_ckpt,
            total_steps  = cfg["sac"]["total_timesteps"],
            seed         = args.seed,
            cfg          = cfg,
            out_dir      = out_dir,
            device       = args.device,
        )
    elif args.algo == "td3":
        from src.adapt_td3 import run_td3_adapt
        run_td3_adapt(
            topology     = args.topology,
            init_mode    = args.init_mode,
            encoder_ckpt = args.encoder_ckpt,
            total_steps  = cfg["td3"]["total_timesteps"],
            seed         = args.seed,
            cfg          = cfg,
            out_dir      = out_dir,
            device       = args.device,
        )

    print(f"\n[run_adapt] Done → {out_dir}/results.json")


if __name__ == "__main__":
    main()
