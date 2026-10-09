"""Fresh confirmation of budget-dependent risk and calibrated-score readouts."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing as mp
import threading

import numpy as np
import torch

from methods.history_guided_testing.config import TARGET
from methods.history_guided_testing.history import historical_risk, load_history, split_indices, subset
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.prepare import measure_scene
from methods.history_guided_testing.scenarios import parameter_cells

from .config import BUDGET, CONFIRMATION, METHODS, SEEDS, SOURCE_WEIGHTS
from .history_model import predict
from .kernel import covariance
from .scenarios import confirmation_scenes, ras_predictions
from .session import RiskTestingSession


def selector(connection, method, seed, x, prior, risk_means, collision_means,
             weights, ras, calibration):
    from threadpoolctl import threadpool_limits

    from methods.history_guided_testing.experiment import RemoteOracle
    from methods.history_guided_testing.search import TestingSession, select_session
    from methods.ras_frt_uq.unified import select_from_responses

    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    oracle = RemoteOracle(connection)
    try:
        if method == "main":
            result = select_session(
                TestingSession(x, seed, mode="global_feedback", prior=prior),
                oracle)
        elif method == "ras_frt_uq":
            result = select_from_responses(x, parameter_cells(x), ras, oracle,
                                           seed)
        elif method == "class_rank":
            selected = np.argsort(-(collision_means @ weights),
                                  kind="stable")[:BUDGET].tolist()
            rows = []
            for index in selected:
                observed = oracle.query(index)
                rows.append({
                    "index": index,
                    "query_number": observed.query_number,
                    "continuous_risk": observed.risk
                })
            result = {"selected_indices": selected, "queries": rows}
        else:
            kernel = covariance(x, risk_means, device="cpu")
            mode = {
                "candidate": "dual",
                "risk_only": "risk",
                "calibrated_only": "calibrated"
            }[method]
            session = RiskTestingSession(x,
                                         kernel,
                                         risk_means,
                                         collision_means,
                                         weights,
                                         calibration,
                                         mode=mode,
                                         device="cpu")
            selected = []
            while (index := session.next_index()) is not None:
                observation = oracle.query(index)
                session.observe(observation.risk)
                selected.append(index)
            result = {"selected_indices": selected, "queries": session.records}
        connection.send(("result", result))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def main():
    protocol = read_json(CONFIRMATION / "protocol.json")
    pool_count = protocol["pool_count"]
    tasks = [
        (pool, method) for pool in range(pool_count) for method in METHODS
        if not (CONFIRMATION / f"pool_{pool}" / f"{method}.json").exists()
    ]
    if not tasks:
        print("All confirmation sequences already saved", flush=True)
        return
    torch.set_num_threads(1)
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    history = subset(history, train)
    calibration = {
        int(key): value
        for key, value in protocol["calibration"].items()
    }
    pools = {}
    for pool in sorted({pool for pool, _ in tasks}):
        seed = SEEDS[pool % len(SEEDS)]
        scenes = confirmation_scenes(pool)
        x = np.asarray([s["numeric_input"] for s in scenes])
        write_json(CONFIRMATION / f"pool_{pool}" / "scenarios.json", scenes)
        risks, collisions = predict(x, seed)
        weights = np.asarray(SOURCE_WEIGHTS)
        prior = historical_risk(history, x)
        ras = ras_predictions(x, seed)
        pools[pool] = (scenes, x, prior, risks, collisions, weights, ras)
    context = mp.get_context("spawn")
    lock = threading.Lock()
    cache, pending = {}, {}
    for pool in range(pool_count):
        path = CONFIRMATION / f"pool_{pool}" / "measurements.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                cache[(pool, row["index"])] = row
    print("Fresh locked confirmation; cached physical calls:",
          len(cache),
          flush=True)
    with ProcessPoolExecutor(max_workers=12, mp_context=context) as simulator:

        def query(pool, index):
            key = (pool, index)
            with lock:
                if key in cache:
                    return cache[key]
                if key not in pending:
                    pending[key] = simulator.submit(
                        measure_scene, (pools[pool][0][index], TARGET))
                future = pending[key]
            row = {**future.result(), "index": index}
            with lock:
                if key not in cache:
                    cache[key] = row
                    with (CONFIRMATION / f"pool_{pool}" /
                          "measurements.jsonl").open(
                              "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    if len(cache) % 200 == 0:
                        print("Fresh physical calls:", len(cache), flush=True)
            return row

        def run_one(arguments):
            pool, method = arguments
            path = CONFIRMATION / f"pool_{pool}" / f"{method}.json"
            scenes, x, prior, risks, collisions, weights, ras = pools[pool]
            seed = SEEDS[pool % len(SEEDS)]
            parent, child = context.Pipe()
            process = context.Process(target=selector,
                                      args=(child, method, seed, x, prior,
                                            risks, collisions, weights, ras,
                                            calibration))
            process.start()
            child.close()
            disclosed = set()
            try:
                while True:
                    message, value = parent.recv()
                    if message == "query":
                        if value in disclosed or len(
                                disclosed
                        ) >= BUDGET or not 0 <= value < len(x):
                            raise ValueError(
                                "Duplicate, over-budget or invalid query")
                        disclosed.add(value)
                        row = query(pool, value)
                        parent.send((None, bool(row["collision"])) if method ==
                                    "ras_frt_uq" else (float(row["risk"]),
                                                       None))
                    elif message == "result":
                        assert len(value["selected_indices"]) == BUDGET
                        assert set(value["selected_indices"]) == disclosed
                        write_json(
                            path, {
                                **value, "pool":
                                pool,
                                "seed":
                                seed,
                                "method":
                                method,
                                "feedback":
                                protocol["feedback"]
                                ["ras_frt_uq" if method ==
                                 "ras_frt_uq" else "other_methods"]
                            })
                        print("Fresh sequence locked:",
                              pool,
                              method,
                              flush=True)
                        break
                    else:
                        raise RuntimeError(value)
            finally:
                parent.close()
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join()

        with ThreadPoolExecutor(max_workers=12) as workers:
            list(workers.map(run_one, tasks))
    write_json(
        CONFIRMATION / "physical_cost.json", {
            "unique_new_simulator_calls": len(cache),
            "simulator_seconds": sum(r["elapsed_s"] for r in cache.values()),
            "selector_queries": pool_count * len(METHODS) * BUDGET,
            "all_sequences_locked": True
        })
    print("Fresh physical confirmation complete", flush=True)


if __name__ == "__main__":
    main()
