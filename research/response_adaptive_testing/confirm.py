"""Run sealed selectors; the parent discloses one permitted observation per query."""
import multiprocessing as mp
import time

import numpy as np
import torch

from methods.history_guided_testing.experiment import RemoteOracle
from methods.history_guided_testing.history import historical_risk, load_history, split_indices, subset
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.scenarios import parameter_cells
from methods.history_guided_testing.search import TestingSession, select_session
from methods.ras_frt_uq.unified import select_from_responses
from research.history_response_testing.calibration import calibrators
from research.history_response_testing.config import SOURCE_WEIGHTS
from research.history_response_testing.history_model import predict
from research.history_response_testing.kernel import covariance
from research.history_response_testing.scenarios import ras_predictions
from research.history_response_testing.session import RiskTestingSession

from .config import BUDGET, SEEDS
from .confirmation import CONFIRMATION, METHODS, REPLICATES, verify_lock
from .develop import metrics
from .session import AdaptiveTestingSession


def select_task(connection, task):
    method, seed, x, prediction, ras, prior, options = task
    oracle = RemoteOracle(connection)
    if method == "ras_frt_uq":
        return select_from_responses(x, parameter_cells(x), ras, oracle, seed)
    if method == "frozen_original":
        return select_session(
            TestingSession(x, seed, mode="global_feedback", prior=prior),
            oracle)
    if method == "class_rank":
        score = prediction[1] @ np.asarray(SOURCE_WEIGHTS)
        selected = np.argsort(-score, kind="stable")[:BUDGET].tolist()
        records = []
        for index in selected:
            observation = oracle.query(index)
            records.append({
                "index": index,
                "continuous_risk": observation.risk
            })
        return {"selected_indices": selected, "queries": records}
    if method == "previous_best":
        session = RiskTestingSession(x, covariance(x, prediction[0]),
                                     *prediction, np.asarray(SOURCE_WEIGHTS),
                                     calibrators())
    elif method == "candidate":
        session = AdaptiveTestingSession(x, *prediction, options)
    else:
        raise ValueError(f"Unknown confirmation method {method}")
    selected = []
    while (index := session.next_index()) is not None:
        session.observe(oracle.query(index).risk)
        selected.append(index)
    return {"selected_indices": selected, "queries": session.records}


def selector_worker(connection):
    torch.set_num_threads(1)
    try:
        while True:
            message, payload = connection.recv()
            if message == "stop":
                return
            if message != "task":
                raise ValueError(f"Unexpected selector message {message}")
            connection.send(("result", select_task(connection, payload)))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def main():
    torch.set_num_threads(1)
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    allowed_history = subset(history, train)
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=selector_worker,
                                              args=(child, ))
    process.start()
    child.close()
    try:
        for profile in protocol["profiles"]:
            for replicate in range(REPLICATES):
                folder = CONFIRMATION / profile["name"] / f"pool_{replicate}"
                bank = np.load(folder / "responses.npz")
                x = bank["x"]
                prior = historical_risk(allowed_history, x)
                for seed in SEEDS:
                    if all((folder / "selection" /
                            f"{method}_{seed}.json").exists()
                           for method in METHODS):
                        continue
                    prediction, ras = predict(x,
                                              seed), ras_predictions(x, seed)
                    for method in METHODS:
                        path = folder / "selection" / f"{method}_{seed}.json"
                        if path.exists():
                            continue
                        started = time.perf_counter()
                        # No target risk, collision array, or target-profile parameter enters this payload.
                        parent.send(
                            ("task", (method, seed, x, prediction, ras, prior,
                                      protocol["candidate_options"])))
                        disclosed, observations = [], []
                        while True:
                            message, value = parent.recv()
                            if message == "query":
                                index = int(value)
                                if not 0 <= index < len(
                                        x) or index in disclosed or len(
                                            disclosed) == BUDGET:
                                    raise ValueError(
                                        "Duplicate, invalid, or excess confirmation query"
                                    )
                                disclosed.append(index)
                                if method == "ras_frt_uq":
                                    collision = bool(bank["collision"][index])
                                    parent.send((None, collision))
                                    observations.append({
                                        "index": index,
                                        "collision": collision
                                    })
                                else:
                                    risk = float(bank["risk"][index])
                                    parent.send((risk, None))
                                    observations.append({
                                        "index": index,
                                        "continuous_risk": risk
                                    })
                            elif message == "result":
                                if value[
                                        "selected_indices"] != disclosed or len(
                                            disclosed) != BUDGET:
                                    raise ValueError(
                                        "Selector result differs from disclosed query sequence"
                                    )
                                result = {
                                    **value,
                                    **metrics(bank["collision"], disclosed), "observations":
                                    observations,
                                    "method":
                                    method,
                                    "seed":
                                    seed,
                                    "selector_elapsed_s":
                                    time.perf_counter() - started,
                                    "feedback":
                                    "queried binary collision"
                                    if method == "ras_frt_uq" else
                                    "queried continuous risk",
                                    "isolation":
                                    "separate selector process; full target arrays retained in parent"
                                }
                                write_json(path, result)
                                print("CONFIRMED SELECTOR",
                                      profile["name"],
                                      replicate,
                                      seed,
                                      method,
                                      round(
                                          result["mean_cumulative_collisions"],
                                          3),
                                      result["F200"],
                                      flush=True)
                                break
                            else:
                                raise RuntimeError(f"Selector failed: {value}")
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Selector process exited with code {process.exitcode}")


if __name__ == "__main__":
    main()
