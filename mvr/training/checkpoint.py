from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import torch


@dataclass(frozen=True)
class HierarchicalCheckpoint:
    schema: str
    stage: str
    config_hash: str
    state: Mapping[str, Any]

    SCHEMA = "hierarchical_checkpoint"

    def save(self, path: str | Path) -> None:
        if self.schema != self.SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        torch.save({"schema": self.schema, "stage": self.stage, "config_hash": self.config_hash, "state": dict(self.state)}, path)

    @classmethod
    def load(cls, path: str | Path, *, expected_config_hash: str | None = None) -> "HierarchicalCheckpoint":
        value = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint = cls(**value)
        if checkpoint.schema != cls.SCHEMA:
            raise ValueError("incompatible checkpoint schema")
        if expected_config_hash is not None and checkpoint.config_hash != expected_config_hash:
            raise ValueError("checkpoint config hash mismatch")
        return checkpoint
