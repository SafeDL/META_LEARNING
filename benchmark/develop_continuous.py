"""Validate continuous mechanisms on development objects before the extension freeze."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from .build_collision_models import model_path
from .collect import require_learning_freeze
from .construction import load_construction
from .continuous_runs import CONFIGURATIONS, run_task, source_contract
from .continuous_selection import METHODS
from .protocol import ROOT, DEFAULT


OUTPUT = ROOT / "results/benchmark/collision/continuous_development"
SETTINGS = {"budget": 32, "initial_count": 8, "algorithm_seeds": [11, 23], "ordinary_only": True}
WORKERS = 4
TASKS = (("development_idm_mobil", "front_brake"), ("development_linear_mpc", "cut_out"),
         ("development_uct", "cut_in"), ("development_ppo", "lane_change_rear"))


def evaluate(sut_id, kind, method, seed):
    history, _ = load_construction(kind)
    checkpoint = (model_path("history_response" if method == "history_response" else "pooled_ft", "all", kind)
                  if method in ("history_response", "indago") else None)
    metadata = {"target_id": sut_id, "scenario": kind, "method": method, "algorithm_seed": seed,
                "environment_seed": DEFAULT.environment_seeds[0], "condition": "continuous_development"}
    destination = OUTPUT / sut_id / kind / f"{method}_{seed}.json"
    output = run_task(sut_id, kind, history if checkpoint else (), method=method, algorithm_seed=seed,
                      checkpoint=checkpoint, destination=destination, metadata=metadata,
                      budget=SETTINGS["budget"], initial_count=SETTINGS["initial_count"],
                      ordinary_only=SETTINGS["ordinary_only"])
    return {"path": str(destination), **metadata, **output["metrics"]}


def main():
    require_learning_freeze()
    rows = []
    plan = [(sut_id, kind, method, seed) for sut_id, kind in TASKS
            for seed in SETTINGS["algorithm_seeds"] for method in METHODS]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        pending = [pool.submit(evaluate, *task) for task in plan]
        for future in as_completed(pending):
            rows.append(future.result())
            print("continuous development", len(rows), "/ 40", flush=True)
    report = {"status": "complete", "data_role": "development_only", "settings": SETTINGS,
              "configurations": CONFIGURATIONS, "source_contract": source_contract(),
              "execution_workers": WORKERS, "runs": rows}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
