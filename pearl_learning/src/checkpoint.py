from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import torch
from .io import content_hash, write_json


def save_checkpoint(path: str | Path, agent: Any, config: Mapping[str, Any], taskbook_hash: str, step: int) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config_hash": content_hash(config), "taskbook_hash": taskbook_hash, "step": step,
        "observation_schema": config["environment"]["observation_schema"],
        "observation_dim": int(config["environment"]["observation_dim"]),
        "action_dim": int(config["environment"]["action_dim"]),
    }
    torch.save({"agent": agent.state_dict(), **metadata}, target)
    write_json(target.with_suffix(".manifest.json"), {"checkpoint": target.name, **metadata})


def load_checkpoint(path: str | Path, agent: Any, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_dim = payload.get("observation_dim")
    expected_dim = agent.actor.backbone[0].in_features - agent.latent_dim
    if saved_dim != expected_dim:
        raise ValueError(
            f"checkpoint observation_dim={saved_dim!r} is incompatible with current observation_dim={expected_dim}; retrain it"
        )
    if payload.get("observation_schema") != agent.observation_schema:
        raise ValueError(
            f"checkpoint observation_schema={payload.get('observation_schema')!r} is incompatible with current observation_schema={agent.observation_schema!r}; retrain it"
        )
    if payload.get("action_dim") != agent.action_dim:
        raise ValueError(
            f"checkpoint action_dim={payload.get('action_dim')!r} is incompatible with current action_dim={agent.action_dim}; retrain it"
        )
    agent.load_state_dict(payload["agent"])
    return payload
