"""Load geometry definitions used by the taskbook and adapters."""
from __future__ import annotations

import json
from pathlib import Path

from .geometry import GeometrySpec


def load_geometry_catalog(path: str | Path = "mvr/configs/geometry_catalog.json") -> dict[str, GeometrySpec]:
    geometries = [GeometrySpec.from_dict(row) for row in json.loads(Path(path).read_text(encoding="utf-8"))]
    result = {geometry.geometry_id: geometry for geometry in geometries}
    if len(result) != len(geometries):
        raise ValueError("geometry ids must be unique")
    return result


def load_adapters() -> dict[str, object]:
    """Return simulator translators without importing them at module load."""
    from .adapters import CutInScenarioAdapter, MergeScenarioAdapter, RoundaboutScenarioAdapter

    return {
        "merge": MergeScenarioAdapter(),
        "cutin": CutInScenarioAdapter(),
        "roundabout": RoundaboutScenarioAdapter(),
    }
