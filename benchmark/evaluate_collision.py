"""Frozen final comparisons; identities and all unqueried outcomes stay in evaluators."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import hashlib

from .build_collision_models import model_path
from .collision_runs import MAIN_METHODS, run_task
from .construction import load_construction
from .executions import ExecutionBank
from .protocol import DATA_ROOT, DEFAULT, RESULT_ROOT, ROOT, method_contract, validate_sut_freeze
from .studies import SCOPES, tasks


def task_path(task):
    return (RESULT_ROOT / "runs" / task.study / task.scope / task.target_id / task.condition
            / task.scenario / f"env_{task.environment_seed}" / f"{task.method}_{task.algorithm_seed}.json")


def evaluate(task):
    scope = SCOPES[task.scope]
    all_history, _ = load_construction(task.scenario)
    history = [a for a in all_history if a.source_id in scope.history_ids]
    bank = ExecutionBank(DATA_ROOT / "targets" / task.target_id / task.condition / task.scenario
                         / f"env_{task.environment_seed}")
    if task.method in MAIN_METHODS:
        checkpoint = model_path("history_response", task.scope, task.scenario)
    elif task.method in ("fsbo", "pooled_ft"):
        checkpoint = model_path(task.method, task.scope, task.scenario)
    else:
        checkpoint = None
    used_history = history if task.method not in ("random", "space_filling", "gp_collision") else ()
    result = run_task(bank, used_history, method=task.method, algorithm_seed=task.algorithm_seed,
                      budget=DEFAULT.budget, initial_count=DEFAULT.initial_count,
                      destination=task_path(task), metadata=asdict(task), checkpoint=checkpoint,
                      horizon=DEFAULT.horizon)
    return {"path": str(task_path(task)), **asdict(task), **result["metrics"]}


def main():
    frozen = json.loads((DATA_ROOT / "frozen_protocol.json").read_text(encoding="utf-8"))
    validate_sut_freeze(frozen)
    if frozen["protocol"] != json.loads(json.dumps(asdict(DEFAULT))):
        raise ValueError("Final run configuration differs from its frozen protocol")
    if frozen["source_contract"] != method_contract():
        raise ValueError("Method, feedback, deduplication or metrics changed after protocol freezing")
    for artifact in frozen["method"]["models"]:
        if hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("A required model changed after final protocol freezing")
    plan = tasks()
    if frozen["study_tasks"] != [asdict(task) for task in plan]:
        raise ValueError("Final study matrix changed after protocol freezing")
    rows = []
    # Four evaluators each own an independent spawn-isolated selector and its paid ledger.
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(evaluate, task): task for task in plan}
        for done in as_completed(pending):
            rows.append(done.result())
            if len(rows) % 16 == 0 or len(rows) == len(plan):
                print("final comparisons", len(rows), "/", len(plan), flush=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "run_index.json").write_text(
        json.dumps({"status": "complete", "runs": rows}, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
