"""Configuration and deterministic JSON helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import yaml


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_config(path: str | Path) -> dict[str, Any]:
    """Read the single project YAML configuration."""
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("configuration root must be a mapping")
    local = dict(value)
    parent = local.pop("extends", None)
    if parent is None:
        return local
    base = read_config(source.parent / str(parent))

    def merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(left)
        for key, item in right.items():
            if isinstance(result.get(key), Mapping) and isinstance(item, Mapping):
                result[key] = merge(result[key], item)
            else:
                result[key] = item
        return result

    return merge(base, local)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
