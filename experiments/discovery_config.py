"""Shared result paths, JSON records and development task enumeration."""

from pathlib import Path

from .protocol import DEFAULT, ROOT

OUTPUT = ROOT / "experiment_outputs/benchmark/collision/discovery"
SCENARIOS = ("cut_in", "cut_out")


def read_json(path):
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def development_tasks():
    from controllers.highway.catalog import SUTS

    return [
        {
            "target_id": s.sut_id,
            "lineage": s.lineage,
            "family": s.family,
            "scenario": kind,
            "condition": "development",
            "environment_seed": 81001,
            "algorithm_seed": seed,
        }
        for s in SUTS
        if s.role == "development"
        for kind in SCENARIOS
        for seed in DEFAULT.algorithm_seeds
    ]
