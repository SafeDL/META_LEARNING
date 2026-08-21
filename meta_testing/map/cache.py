"""Versioned raw-token and frozen-embedding caches."""
from __future__ import annotations

from pathlib import Path
import torch

from .schema import MapTokens


class MapCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.embeddings = self.root / "embeddings"

    def raw_path(self, map_hash: str) -> Path:
        return self.raw / f"{map_hash}.pt"

    def embedding_path(self, map_hash: str) -> Path:
        return self.embeddings / f"{map_hash}.pt"

    def save_raw(self, tokens: MapTokens) -> Path:
        self.raw.mkdir(parents=True, exist_ok=True)
        path = self.raw_path(tokens.map_hash)
        torch.save(tokens, path)
        return path

    def load_raw(self, map_hash: str) -> MapTokens | None:
        path = self.raw_path(map_hash)
        return torch.load(path, weights_only=False) if path.exists() else None

    def save_embeddings(self, map_hash: str, local: torch.Tensor, global_embedding: torch.Tensor, *, encoder_frozen: bool) -> Path:
        if not encoder_frozen:
            raise ValueError("trainable map encoders must not persist final embeddings")
        self.embeddings.mkdir(parents=True, exist_ok=True)
        path = self.embedding_path(map_hash)
        torch.save({"local": local.detach().cpu(), "global": global_embedding.detach().cpu()}, path)
        return path

    def load_embeddings(self, map_hash: str, *, encoder_frozen: bool) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not encoder_frozen:
            return None
        path = self.embedding_path(map_hash)
        if not path.exists():
            return None
        value = torch.load(path, weights_only=True)
        return value["local"], value["global"]
