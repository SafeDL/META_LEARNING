#!/usr/bin/env python
"""
R020 — FOMAML meta-training.

Produces:  results/meta_train/encoder_final.pt

Usage:
  python scripts/run_meta_train.py --config configs/default.yaml
  python scripts/run_meta_train.py --config configs/default.yaml --device cuda
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fomaml import FOMAMLTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="results/meta_train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import random
    import numpy as np
    import torch as _torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    _torch.manual_seed(args.seed)

    print(f"[R020] FOMAML meta-training")
    print(f"  device    : {args.device}")
    print(f"  out_dir   : {args.out_dir}")
    print(f"  total_env_steps: {cfg['fomaml']['total_env_steps']}")

    trainer = FOMAMLTrainer(cfg, device=args.device, out_dir=args.out_dir)
    trainer.train()
    trainer.close()

    ckpt = os.path.join(args.out_dir, "encoder_final.pt")
    print(f"\n[R020] Done. Encoder checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
