"""Synchronize checkpoint provenance after an evaluation-only config change.

The model weights are copied unchanged.  This is safe only when the changed
configuration fields are not used by training; the caller must verify that
condition before running the utility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvr.training.pipeline import checkpoint_config_hash, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    config, _, _ = load_config(args.config)
    digest = checkpoint_config_hash(config)
    root = Path(args.root)
    filenames = ["context_meta.pt"]
    if (root / "interaction_prior.pt").exists():
        filenames.append("interaction_prior.pt")
    elif (root / "interaction_prior_migrated.pt").exists():
        filenames.append("interaction_prior_migrated.pt")
    for filename in filenames:
        checkpoint_path = root / filename
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["config_hash"] = digest
        checkpoint["state"]["provenance_config"] = config
        torch.save(checkpoint, checkpoint_path)

        metadata_path = checkpoint_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        metadata.update(
            {
                "stage": checkpoint["stage"],
                "compatibility_hash": digest,
                "config_hash": digest,
                "config": config,
                "checkpoint_hash": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(digest)


if __name__ == "__main__":
    main()
