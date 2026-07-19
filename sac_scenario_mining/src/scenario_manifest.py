"""Versioned, lightweight scenario artefacts for deterministic action replay."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MANIFEST_SCHEMA = "stage1_manifest_v1"


def save_manifest(
    directory: str | Path,
    manifest: Mapping[str, Any],
    actions: np.ndarray,
    metrics: Mapping[str, Any],
) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    payload = {**manifest, "schema_version": MANIFEST_SCHEMA}
    (path / "manifest.json").write_text(json.dumps(payload,
                                                   indent=2,
                                                   sort_keys=True),
                                        encoding="utf-8")
    np.save(path / "actions.npy", np.asarray(actions, dtype=np.float32))
    (path / "metrics.json").write_text(json.dumps(dict(metrics),
                                                  indent=2,
                                                  sort_keys=True),
                                       encoding="utf-8")
    return path / "manifest.json"


def load_manifest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported manifest schema: {data.get('schema_version')}")
    return data
