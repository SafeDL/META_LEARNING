"""Run the registered E02 matrix, report it, and stop before E03."""

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack

from controllers.highway.catalog import FORMAL_TARGET_SUTS

from experiments.budget import protocol
from experiments.budget.execution import execute
from experiments.executions import ExecutionBank
from experiments.parallel import METHOD_WORKERS, initialize_worker, prepare_processes
from experiments.storage import read_json, write_json
from experiments.transfer_batches import prepare_context

_CONTEXT = None
_BANKS = None


def initialize(context):
    global _CONTEXT, _BANKS
    initialize_worker()
    _CONTEXT, _BANKS = context, {}


def run_case(kind, sut_id, method):
    key = kind, sut_id
    if key not in _BANKS:
        _BANKS[key] = ExecutionBank(protocol.INPUT / "physical/target" / kind / sut_id)
    print({"scene": kind, "target": sut_id, "method": method,
           "status": "running", "pid": os.getpid()}, flush=True)
    try:
        execute(_BANKS[key], method, _CONTEXT)
        result = {"scene": kind, "target": sut_id, "method": method, "status": "complete"}
    except Exception as error:
        result = {"scene": kind, "target": sut_id, "method": method,
                  "status": "failed", "error": repr(error)}
    print(result, flush=True)
    return result


def run_methods(methods):
    if not set(methods).issubset(set(protocol.METHODS) - {"random"}):
        raise ValueError("Only registered E02 query methods can run")
    initialize_worker()
    if read_json(protocol.OUTPUT / "protocol.json") != protocol.declaration():
        raise ValueError("The registered E02 protocol changed")
    prepare_processes()
    rows = []
    for kind in protocol.RUN_SCENARIOS:
        context, _ = prepare_context(protocol.INPUT, kind, "prediction")
        with ExitStack() as stack:
            pending = {}
            for group, workers in (([m for m in methods if m != "pooled_history"], METHOD_WORKERS),
                                   ([m for m in methods if m == "pooled_history"], 1)):
                if not group:
                    continue
                pool = stack.enter_context(ProcessPoolExecutor(
                    max_workers=workers, mp_context=mp.get_context("spawn"),
                    initializer=initialize, initargs=(context,)))
                for method in group:
                    for sut in FORMAL_TARGET_SUTS:
                        pending[pool.submit(run_case, kind, sut.sut_id, method)] = (kind, sut.sut_id, method)
            for future in as_completed(pending):
                row = future.result()
                rows.append(row)
                status_dir = protocol.OUTPUT / "status" / row["scene"] / row["target"]
                status_path = status_dir / f"{row['method']}.json"
                if row["status"] == "failed":
                    write_json(status_path, row)
                else:
                    status_path.unlink(missing_ok=True)
    return rows


def run_gp():
    return run_methods(("full", "target_only", "pooled_history", "static_prior"))


def run_baselines():
    return run_methods(("scenario_fuzz", "paired_system_active_evaluation"))


def finish(started):
    from experiments.budget.report import report

    completed = report()
    if completed["status"] != "complete" or completed["cases"] != 224:
        raise ValueError("E02 has incomplete or unaudited registered results")
    from experiments.plotting.budget_discovery import render

    render()
    print({"status": "complete", "experiment_state": "paused_after_budget_discovery",
           "started": started, "completed": time.time()}, flush=True)


def run():
    from experiments.budget.prepare import run as prepare

    started = time.time()
    prepare()
    run_gp()
    run_baselines()
    finish(started)


if __name__ == "__main__":
    run()
