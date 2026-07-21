"""Self-contained, replayable PEARL critical-scenario artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def save_manifest(directory: str | Path, manifest: Mapping[str, Any], actions: np.ndarray, metrics: Mapping[str, Any]) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.save(target / "actions.npy", np.asarray(actions, dtype=np.float32))
    (target / "metrics.json").write_text(
        json.dumps(dict(metrics), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(directory: str | Path) -> dict[str, Any]:
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
