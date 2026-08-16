"""Resumable, provenance-complete PEARL checkpoints."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping
import subprocess
import torch

from .io import content_hash, write_json


# One maintained checkpoint format.  The method contract, rather than a
# numeric suffix, distinguishes it from retired context/replay artifacts.
CHECKPOINT_SCHEMA = "pearl_checkpoint"
METHOD_CONTRACT = "transition_product_recent_context"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path: str | Path, agent: Any, config: Mapping[str, Any], taskbook_hash: str, step: int, *,
                    casebook_hashes: Mapping[str, str], training_seed: int, rng_state: Mapping[str, Any],
                    trainer_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": CHECKPOINT_SCHEMA, "method_contract": METHOD_CONTRACT,
        "git_commit": git_commit(), "config_hash": content_hash(config), "taskbook_hash": taskbook_hash,
        "casebook_hashes": dict(casebook_hashes), "training_seed": int(training_seed), "step": int(step),
        "observation_schema": config["environment"]["observation_schema"], "observation_dim": int(config["environment"]["observation_dim"]), "action_dim": int(config["environment"]["action_dim"]),
        "no_topology_ablation": bool(config.get("ablation", {}).get("no_topology", False)),
        "no_context_training": bool(config.get("ablation", {}).get("no_context_training", False)),
        "run_kind": str(config.get("experiment", {}).get("run_kind", "formal")),
        "architecture": agent.architecture_metadata(),
    }
    payload = {"agent": agent.state_dict(), "rng_state": dict(rng_state), **metadata}
    if trainer_state is not None:
        payload["trainer_state"] = dict(trainer_state)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata["checkpoint_hash"] = _file_hash(target)
    write_json(target.with_suffix(".manifest.json"), {"checkpoint": target.name, "resumable_trainer_state": trainer_state is not None, **metadata})
    return metadata


def load_checkpoint(path: str | Path, agent: Any, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("method_contract") != METHOD_CONTRACT:
        raise ValueError(
            f"unsupported checkpoint contract "
            f"{(payload.get('schema'), payload.get('method_contract'))!r}; only "
            f"{(CHECKPOINT_SCHEMA, METHOD_CONTRACT)!r} is accepted and incompatible checkpoints must be retrained"
        )
    saved_dim = payload.get("observation_dim")
    if saved_dim != agent.observation_dim:
        raise ValueError(f"checkpoint observation_dim={saved_dim!r} is incompatible with current observation_dim={agent.observation_dim}; retrain it")
    if payload.get("observation_schema") != agent.observation_schema:
        raise ValueError(f"checkpoint observation_schema={payload.get('observation_schema')!r} is incompatible with current observation_schema={agent.observation_schema!r}; retrain it")
    if payload.get("action_dim") != agent.action_dim:
        raise ValueError(f"checkpoint action_dim={payload.get('action_dim')!r} is incompatible with current action_dim={agent.action_dim!r}; retrain it")
    if "rng_state" not in payload:
        raise ValueError("checkpoint lacks RNG state and is not resumable")
    if payload.get("architecture") != agent.architecture_metadata():
        raise ValueError("checkpoint architecture metadata is incompatible with the configured agent")
    agent.load_state_dict(payload["agent"])
    return payload
