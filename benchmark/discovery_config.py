"""Fixed development experiment; independent of the old frozen final matrix."""

from pathlib import Path

from .protocol import DEFAULT, ROOT


OUTPUT = ROOT / "results/benchmark/collision/discovery"
DATA = ROOT / "data/collision/discovery"
SCENARIOS = ("cut_in", "cut_out")
BANDWIDTHS = (0.15, 0.30, 0.60)
WORKERS = 4
SUPPLEMENT_LIMIT = 2048
BASELINES = ("random", "history_ranking", "gp_collision", "nearest_residual",
             "pooled_ft", "fsbo", "history_response", "greedy", "source_global")
METHODS = (*BASELINES, "shared_local", "source_local")
STRONG_BASELINES = ("gp_collision", "nearest_residual", "pooled_ft", "fsbo")
CONTRACT_PATHS = ("benchmark/discovery.py", "benchmark/discovery_config.py",
                  "benchmark/discovery_diagnosis.py", "benchmark/collision_runs.py",
                  "benchmark/collision_metrics.py", "benchmark/query.py",
                  "methods/gcrt/discovery.py", "methods/gcrt/posterior.py",
                  "methods/gcrt/selection.py")


def run_path(target_id, kind, method, seed, bandwidth=None):
    if method in ("history_response", "greedy", "source_global", "shared_local", "source_local",
                  "source_global_h5", "source_global_full", "source_local_h5", "source_local_full"):
        root = ROOT / "results/gcrt/collision/discovery"
    else:
        root = ROOT / "results/baselines" / method / "collision/discovery"
    setting = "global" if bandwidth is None else f"bandwidth_{bandwidth:g}"
    return root / "runs" / target_id / kind / method / setting / f"{seed}.json"


def read_json(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def development_tasks():
    from controllers.highway.catalog import SUTS
    return [{"target_id": s.sut_id, "lineage": s.lineage, "family": s.family,
             "scenario": kind, "condition": "development", "environment_seed": 81001,
             "algorithm_seed": seed}
            for s in SUTS if s.role == "development"
            for kind in SCENARIOS for seed in DEFAULT.algorithm_seeds]
