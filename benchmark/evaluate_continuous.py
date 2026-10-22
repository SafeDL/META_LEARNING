"""Actual continuous searches; no candidate-pool bank or target labels enter selectors."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json

from controllers.highway.catalog import BY_ID

from .build_collision_models import model_path
from .construction import load_construction
from .continuous_runs import CONFIGURATIONS, run_task, source_contract
from .continuous_selection import METHODS
from .protocol import DATA_ROOT, DEFAULT, RESULT_ROOT, ROOT, validate_sut_freeze
from .studies import CONTINUOUS_TASKS


OUTPUT = RESULT_ROOT / "continuous"
WORKERS = 12


def evaluate(sut_id, kind, method, seed, environment_seed):
    history, _ = load_construction(kind)
    checkpoint = (model_path("history_response" if method == "history_response" else "pooled_ft", "all", kind)
                  if method in ("history_response", "indago") else None)
    sut = BY_ID[sut_id]
    metadata = {"target_id": sut_id, "lineage": sut.lineage, "family": sut.family,
                "scenario": kind, "condition": "continuous", "method": method,
                "algorithm_seed": seed, "environment_seed": environment_seed}
    path = OUTPUT / sut_id / kind / f"env_{environment_seed}" / f"{method}_{seed}.json"
    output = run_task(sut_id, kind, history if checkpoint else (), method=method, algorithm_seed=seed,
                      checkpoint=checkpoint, destination=path, metadata=metadata,
                      budget=DEFAULT.budget, initial_count=DEFAULT.initial_count,
                      environment_seed=environment_seed)
    return {"path": str(path), **metadata, **output["metrics"]}


def validate_frozen():
    validate_sut_freeze(json.loads((DATA_ROOT / "frozen_protocol.json").read_text(encoding="utf-8")))
    manifest = json.loads((DATA_ROOT / "frozen_continuous.json").read_text(encoding="utf-8"))
    if (manifest["protocol"] != json.loads(json.dumps(asdict(DEFAULT)))
            or manifest["tasks"] != [list(t) for t in CONTINUOUS_TASKS]
            or manifest["methods"] != list(METHODS) or manifest["configurations"] != CONFIGURATIONS
            or manifest["execution_workers"]["final"] != WORKERS
            or manifest["source_contract"] != source_contract()):
        raise ValueError("Continuous experiment changed after its own preregistration")
    artifacts = [*manifest["models"],
                 {"path": str((DATA_ROOT / "frozen_protocol.json").relative_to(ROOT)),
                  "sha256": manifest["main_protocol_sha256"]},
                 {"path": manifest["development_report"], "sha256": manifest["development_report_sha256"]}]
    for artifact in artifacts:
        if hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("Continuous frozen evidence or historical weights changed")


def main():
    validate_frozen()
    plan = [(sut, kind, method, seed, env) for sut, kind in CONTINUOUS_TASKS
            for seed in DEFAULT.algorithm_seeds for env in DEFAULT.environment_seeds for method in METHODS]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        pending = [pool.submit(evaluate, *task) for task in plan]
        for done in as_completed(pending):
            rows.append(done.result())
            print("continuous comparisons", len(rows), "/", len(plan), flush=True)
    (OUTPUT / "run_index.json").write_text(
        json.dumps({"status": "complete", "runs": rows}, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
