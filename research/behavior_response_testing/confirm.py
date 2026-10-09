"""Select through a pipe that discloses exactly one allowed target response."""
import multiprocessing as mp
import time

import numpy as np
import torch

from methods.history_guided_testing.experiment import RemoteOracle
from methods.history_guided_testing.history import historical_risk, load_history, split_indices, subset
from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import predict
from research.history_response_testing.kernel import covariance
from research.history_response_testing.scenarios import ras_predictions
from research.history_response_testing.session import RiskTestingSession
from research.response_adaptive_testing.confirm import select_task as select_frozen
from research.response_adaptive_testing.develop import metrics

from .confirmation import CONFIRMATION, METHODS, REPLICATES, verify_lock
from .log_probability import LogProbabilityTestingSession
from .model import BehaviorResponseModel
from .prediction import response_tables
from .session import BUDGET
from .train import OUTPUT, SEEDS

FROZEN_METHODS = ("previous_best", "frozen_original", "ras_frt_uq")


def select_task(connection, task):
    method = task["method"]
    if method in FROZEN_METHODS:
        return select_frozen(
            connection, (method, task["seed"], task["x"], task["prediction"],
                         task["ras"], task["prior"], None))
    oracle = RemoteOracle(connection)
    if method == "matched_collision_gp":
        risk, collision = task["prediction"]
        session = RiskTestingSession(task["x"],
                                     covariance(task["x"], risk),
                                     risk,
                                     collision,
                                     np.full(risk.shape[1], 1 / risk.shape[1]),
                                     task["calibrators"],
                                     mode="calibrated")
    elif method in ("candidate", "discrete_history", "fixed_behavior"):
        risk, logits = task["prediction"]
        session = LogProbabilityTestingSession(task["x"], risk, logits,
                                               task["discrepancy"])
    else:
        raise ValueError(f"Unknown behavioral confirmation method {method}")
    if method == "fixed_behavior":
        selected = torch.argsort(session.log_safe_probabilities(),
                                 stable=True)[:BUDGET].tolist()
        queries = [{
            "index": index,
            "continuous_risk": oracle.query(index).risk
        } for index in selected]
        return {"selected_indices": selected, "queries": queries}
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


def disclose(parent, task, bank):
    parent.send(("task", task))
    disclosed, observations = [], []
    while True:
        message, value = parent.recv()
        if message == "query":
            index = int(value)
            if not 0 <= index < len(bank["x"]) or index in disclosed or len(
                    disclosed) == BUDGET:
                raise ValueError("Duplicate, invalid or excess target query")
            disclosed.append(index)
            if task["method"] == "ras_frt_uq":
                collision = bool(bank["collision"][index])
                parent.send((None, collision))
                observations.append({"index": index, "collision": collision})
            else:
                risk = float(bank["risk"][index])
                parent.send((risk, None))
                observations.append({"index": index, "continuous_risk": risk})
        elif message == "result":
            if value["selected_indices"] != disclosed or len(
                    disclosed) != BUDGET:
                raise ValueError(
                    "Selector sequence differs from disclosed queries")
            return {
                **value,
                **metrics(bank["collision"], disclosed),
                **{
                    f"F{n}":
                    int(np.count_nonzero(bank["collision"][disclosed[:n]]))
                    for n in (10, 30, 150)
                }, "observations": observations
            }
        else:
            raise RuntimeError(f"Selector failed: {value}")


def main():
    torch.set_num_threads(1)
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    options = protocol["candidate"]
    grid = np.asarray(options["continuous_behaviors"], dtype=np.float32)
    historical = np.asarray(options["historical_behaviors"], dtype=np.float32)
    calibrators = {
        int(key): value
        for key, value in options["risk_calibrators"].items()
    }
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    allowed_history = subset(history, train)
    predictors = {}
    for seed in SEEDS:
        states = torch.load(OUTPUT / "models" / f"predictor_{seed}.pt",
                            map_location="cuda",
                            weights_only=True)
        models = []
        for family in (0, 1):
            model = BehaviorResponseModel().cuda().eval()
            model.load_state_dict(states[str(family)])
            models.append(model)
        predictors[seed] = models
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=selector_worker, args=(child, ))
    process.start()
    child.close()
    try:
        for profile in protocol["profiles"]:
            for replicate in range(REPLICATES):
                folder = CONFIRMATION / profile["name"] / f"pool_{replicate}"
                with np.load(folder / "responses.npz") as bank:
                    x = bank["x"]
                    prior = historical_risk(allowed_history, x)
                    for seed in SEEDS:
                        if all((folder / "selection" /
                                f"{m}_{seed}.json").exists() for m in METHODS):
                            continue
                        original, ras = predict(x, seed), ras_predictions(
                            x, seed)
                        grid_prediction = tuple(
                            value.cpu().numpy() for value in response_tables(
                                x, grid, predictors[seed], return_logits=True))
                        history_tensors = response_tables(x,
                                                          historical,
                                                          predictors[seed],
                                                          return_logits=True)
                        matched_prediction = (
                            history_tensors[0].cpu().numpy().T,
                            history_tensors[1].sigmoid().cpu().numpy().T)
                        history_prediction = tuple(
                            value.cpu().numpy() for value in history_tensors)
                        for method in METHODS:
                            path = folder / "selection" / f"{method}_{seed}.json"
                            if path.exists():
                                continue
                            # Only scenario coordinates and frozen historical predictions enter the worker.
                            task = {"method": method, "seed": seed, "x": x}
                            if method in FROZEN_METHODS:
                                task.update(prediction=original,
                                            ras=ras,
                                            prior=prior)
                            elif method == "matched_collision_gp":
                                task.update(prediction=matched_prediction,
                                            calibrators=calibrators)
                            else:
                                task.update(
                                    prediction=history_prediction if method
                                    == "discrete_history" else grid_prediction,
                                    discrepancy=options["risk_discrepancy"][
                                        str(seed)])
                            started = time.perf_counter()
                            result = disclose(parent, task, bank)
                            result.update(
                                method=method,
                                seed=seed,
                                selector_elapsed_s=time.perf_counter() -
                                started,
                                feedback="queried collision" if method
                                == "ras_frt_uq" else "queried continuous risk",
                                isolation=
                                "Separate selector process; full target arrays and target parameters retained in parent"
                            )
                            write_json(path, result)
                            print("BEHAVIOR CONFIRMED SELECTOR",
                                  profile["name"],
                                  replicate,
                                  seed,
                                  method,
                                  result["F200"],
                                  flush=True)
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
