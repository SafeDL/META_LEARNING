"""The current SRD-TNP-BQD main chain and its S01 baseline comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import multiprocessing as mp
import time
import traceback

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from .acquisition import Selector
from .baseline import METHODS, select
from .benchmark import BUDGET, SEEDS
from .common import append_jsonl, read_json, write_json
from .data import HistoryOutput
from .matched_history import SOURCE, prepare_source, source_context
from .mean_readout import fit_readout, local_mean
from .oracle import ContinuousOracle
from .s01 import (CELL_COUNT, ROOT, SOURCES, TARGET, cells_for,
                  configuration, measure_target, numeric_inputs, response_guard,
                  response_rows, source_contexts)
from .session import RemoteOracle, tensor_digest
from .train import load_modules

BASELINES = {**METHODS, "ras_frt_uq": "RAS-FRT-UQ"}
IMPLEMENTATION = "matched_history_mean_readout"


def history_ranking(x, contexts):
    values = []
    for context in contexts:
        nearest = np.argsort(cdist(x, context.x), axis=1, kind="stable")[:, :configuration()["baselines"]["knn_neighbors"]]
        risk = 1 / (1 + np.exp(-context.z[:, 0]))
        values.append(risk[nearest].mean(axis=1))
    return np.mean(values, axis=0)


def worker(connection, task, x, cells):
    try:
        response_guard()
        threadpool_limits(limits=1)
        oracle = RemoteOracle(connection)
        method, seed = task["method"], task["seed"]
        if method == "ras_frt_uq":
            from methods.ras_frt_uq.s01 import select as select_ras
            result = select_ras(x, cells, oracle, seed, BUDGET)
        elif method in METHODS:
            scores = history_ranking(x, source_contexts()) if method == "knn_history" else None
            result = select(method, x, cells, oracle, seed, BUDGET,
                            configuration()["baselines"], scores, CELL_COUNT)
        elif method == "SRD_TNP_BQD":
            import torch
            contexts = source_contexts()
            model, kernel = load_modules(ROOT / "model" / f"seed_{seed}.pt")
            model.freeze()
            kernel.freeze()
            before = tensor_digest(model), tensor_digest(kernel)
            matched_context = source_context()
            coefficients, readout_training = fit_readout(model, matched_context, seed)
            matched_mean = model.predict([matched_context], x).m
            mean = (coefficients[0] * matched_mean
                    + coefficients[1] * local_mean(matched_context, x) + coefficients[2])
            # Restore the original complete contexts for the independent frozen h.
            prior = model.predict(contexts, x)
            frozen_cache = [layer.detach().clone() for source in model.deployment_context_cache for layer in source]
            frozen_mean, frozen_h = prior.m.copy(), prior.h.copy()
            history = HistoryOutput(mean, prior.h)
            result = Selector(history, kernel, cell_count=CELL_COUNT,
                              mean_calibration=configuration()["mean_calibration"]).run(
                x, cells, oracle, BUDGET,
                lambda row: connection.send({"type": "record", "row": row}))
            assert before == (tensor_digest(model), tensor_digest(kernel))
            assert np.array_equal(prior.m, frozen_mean) and np.array_equal(prior.h, frozen_h)
            current_cache = [layer for source in model.deployment_context_cache for layer in source]
            assert all(torch.equal(a, b) for a, b in zip(frozen_cache, current_cache))
            result.update(frozen_modules=True, frozen_history_cache=True, sources=list(SOURCES),
                          feedback_used="continuous risk",
                          implementation=IMPLEMENTATION, mean_source=SOURCE, h_sources=list(SOURCES),
                          source_selection="match known target desired speed and max_brake",
                          mean_source_physical_responses=2048, mean_readout=readout_training)
        else:
            raise ValueError(f"unknown method: {method}")
        connection.send({"type": "complete", "result": result})
    except BaseException:
        connection.send({"type": "error", "message": traceback.format_exc()})
    finally:
        connection.close()


def run(method, seed, online=False):
    section = "online_s01" if online else "comparison/runs"
    path = ROOT / section / f"{method}_seed_{seed}.json"
    if path.exists():
        return read_json(path)
    scenes, rows = response_rows("D")
    x, cells = numeric_inputs(scenes), cells_for(scenes)

    def response(index):
        expected = rows[index]
        if online:
            actual = measure_target(scenes[index])
            assert actual["ego_collision"] == expected["ego_collision"]
            assert abs(actual["risk"] - expected["risk"]) < 1e-12
            return int(actual["ego_collision"]), actual["risk"], actual["valid_risk"]
        return int(expected["ego_collision"]), expected["risk"], expected["valid_risk"]

    oracle = ContinuousOracle([s["scenario_id"] for s in scenes], response, BUDGET)
    task = {"method": method, "seed": seed}
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=worker, args=(child, task, x, cells))
    log = path.with_suffix(".jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise FileExistsError(f"unfinished query log: {log}")
    process.start()
    child.close()
    started = time.perf_counter()
    try:
        while True:
            message = parent.recv()
            if message["type"] == "query":
                parent.send({"observation": asdict(oracle.query(message["index"]))})
            elif message["type"] == "record":
                append_jsonl(log, message["row"])
            elif message["type"] == "complete":
                result = message["result"]
                break
            else:
                raise RuntimeError(message["message"])
    finally:
        parent.close()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
    if method in BASELINES:
        for row in result["queries"]:
            append_jsonl(log, row)
    assert len(set(result["selected_indices"])) == len(oracle.queried) == BUDGET
    result.update(task, target=TARGET, elapsed_s=time.perf_counter() - started,
                  physical_executions=BUDGET if online else 0)
    write_json(path, result)
    print(method, seed, "complete", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="verify one complete physical trajectory")
    args = parser.parse_args()
    prepare_source()
    if args.online:
        online = run("SRD_TNP_BQD", 11, True)
        replay = run("SRD_TNP_BQD", 11)
        assert online["selected_indices"] == replay["selected_indices"]
        write_json(ROOT / "comparison/online_verification.json",
                   {"status": "PASS", "physical_queries": BUDGET,
                    "selected_indices_and_observations_equal": True, "target": TARGET})
        return
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
        tasks = [executor.submit(run, method, seed)
                 for method in (*BASELINES, "SRD_TNP_BQD")
                 for seed in ((11,) if method == "knn_history" else SEEDS)]
        for task in as_completed(tasks):
            task.result()


if __name__ == "__main__":
    main()
