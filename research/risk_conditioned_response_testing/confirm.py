"""Run isolated selectors and disclose one permitted response per query."""
import multiprocessing as mp
import time

import numpy as np
import torch

from methods.history_guided_testing.experiment import RemoteOracle
from methods.history_guided_testing.history import (
    historical_risk, load_history, split_indices, subset)
from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.log_probability import (
    LogProbabilityTestingSession)
from research.behavior_response_testing.model import BehaviorResponseModel
from research.behavior_response_testing.prediction import response_tables
from research.behavior_response_testing.train import OUTPUT as MODEL_OUTPUT
from research.history_response_testing.history_model import predict
from research.history_response_testing.kernel import covariance
from research.history_response_testing.scenarios import ras_predictions
from research.history_response_testing.session import RiskTestingSession
from research.response_adaptive_testing.confirm import (
    select_task as select_frozen)
from research.response_adaptive_testing.develop import metrics

from .confirmation import (BUDGET, CONFIRMATION, METHODS, REPLICATES,
                           SEEDS, verify_lock)
from .risk_session import RiskConditionedTestingSession


FROZEN_METHODS = ("previous_best", "frozen_original", "ras_frt_uq")
ROUND_TWO = CONFIRMATION.parents[2] / "behavior_response_testing" / "results" / "confirmation"


def select_task(connection, task):
    method = task["method"]
    if method in FROZEN_METHODS:
        return select_frozen(
            connection,
            (method, task["seed"], task["x"], task["original"],
             task["ras"], task["prior"], None))
    oracle = RemoteOracle(connection)
    if method == "matched_collision_gp":
        risk, collision = task["historical_prediction"]
        calibrators = {
            int(key): value
            for key, value in task["matched_calibrators"].items()
        }
        session = RiskTestingSession(
            task["x"], covariance(task["x"], risk), risk, collision,
            np.full(risk.shape[1], 1 / risk.shape[1]),
            calibrators, mode="calibrated")
    elif method == "behavior_posterior":
        risk, logits = task["grid_prediction"]
        session = LogProbabilityTestingSession(
            task["x"], risk, logits, task["discrepancy"])
    elif method in ("risk_conditioned", "collision_only_ablation"):
        risk, logits = task["grid_prediction"]
        session = RiskConditionedTestingSession(
            task["x"], risk, logits, task["discrepancy"],
            task["decoder"],
            collision_only=(method == "collision_only_ablation"),
            budget=BUDGET)
    else:
        raise ValueError(f"Unknown round-three method: {method}")
    selected = []
    while (index := session.next_index()) is not None:
        session.observe(oracle.query(index).risk)
        selected.append(index)
    return {"selected_indices": selected, "queries": session.records}


def selector_worker(connection):
    torch.set_num_threads(1)
    try:
        while True:
            message, task = connection.recv()
            if message == "stop":
                return
            if message != "task":
                raise ValueError(f"Unexpected worker message: {message}")
            connection.send(("result", select_task(connection, task)))
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
    states_by_seed, behavior_models = {}, {}
    for seed in SEEDS:
        states_by_seed[seed] = torch.load(
            MODEL_OUTPUT / "models" / f"predictor_{seed}.pt",
            map_location="cuda", weights_only=True)
        models = []
        for family in (0, 1):
            model = BehaviorResponseModel().cuda().eval()
            model.load_state_dict(states_by_seed[seed][str(family)])
            models.append(model)
        behavior_models[seed] = models
    candidate = protocol["candidate"]
    grid = np.asarray(candidate["continuous_behaviors"], dtype=np.float32)
    historical = np.asarray(
        read_json(ROUND_TWO / "protocol.json")["candidate"][
            "historical_behaviors"], dtype=np.float32)

    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=selector_worker, args=(child, ))
    process.start()
    child.close()
    try:
        for profile in protocol["profiles"]:
            for replicate in range(REPLICATES):
                folder = (CONFIRMATION / profile["name"] /
                          f"pool_{replicate}")
                with np.load(folder / "responses.npz") as bank:
                    x = bank["x"].copy()
                    target_risk = bank["risk"].copy()
                    target_collision = bank["collision"].copy()
                prior = historical_risk(allowed_history, x)
                for seed in SEEDS:
                    original = predict(x, seed)
                    ras = ras_predictions(x, seed)
                    grid_prediction = tuple(
                        value.cpu().numpy() for value in response_tables(
                            x, grid, behavior_models[seed], return_logits=True))
                    historical_table = response_tables(
                        x, historical, behavior_models[seed],
                        return_logits=True)
                    matched_prediction = (
                        historical_table[0].cpu().numpy().T,
                        historical_table[1].sigmoid().cpu().numpy().T)
                    for method in METHODS:
                        path = folder / "selection" / f"{method}_{seed}.json"
                        if path.exists():
                            continue
                        task = {
                            "method": method,
                            "seed": seed,
                            "x": x,
                        }
                        if method in FROZEN_METHODS:
                            task.update(original=original, ras=ras, prior=prior)
                        elif method == "matched_collision_gp":
                            task.update(
                                historical_prediction=matched_prediction,
                                matched_calibrators=candidate[
                                    "matched_collision_calibrators"])
                        elif method == "behavior_posterior":
                            task.update(
                                grid_prediction=grid_prediction,
                                discrepancy=candidate["risk_discrepancy"][
                                    str(seed)])
                        else:
                            task.update(
                                grid_prediction=grid_prediction,
                                discrepancy=candidate["risk_discrepancy"][
                                    str(seed)],
                                decoder=candidate[
                                    "risk_conditioned_decoder"][str(seed)])
                        started = time.perf_counter()
                        parent.send(("task", task))
                        disclosed, observations = [], []
                        while True:
                            message, value = parent.recv()
                            if message == "query":
                                index = int(value)
                                if (not 0 <= index < len(x) or
                                        index in disclosed or
                                        len(disclosed) == BUDGET):
                                    raise ValueError(
                                        "Duplicate, invalid, or excess query")
                                disclosed.append(index)
                                if method == "ras_frt_uq":
                                    collision = bool(target_collision[index])
                                    parent.send((None, collision))
                                    observations.append({
                                        "index": index,
                                        "collision": collision})
                                else:
                                    risk = float(target_risk[index])
                                    parent.send((risk, None))
                                    observations.append({
                                        "index": index,
                                        "continuous_risk": risk})
                            elif message == "result":
                                if (value["selected_indices"] != disclosed or
                                        len(disclosed) != BUDGET):
                                    raise ValueError(
                                        "Worker sequence differs from queried disclosures")
                                result = {
                                    **value,
                                    **metrics(target_collision, disclosed),
                                    "observations": observations,
                                    "method": method,
                                    "seed": seed,
                                    "selector_elapsed_s":
                                    time.perf_counter() - started,
                                    "feedback": "queried binary collision"
                                    if method == "ras_frt_uq" else
                                    "queried continuous risk",
                                    "isolation": "separate worker; target parameters and full response arrays retained in parent",
                                }
                                folder.joinpath("selection").mkdir(
                                    parents=True, exist_ok=True)
                                write_json(path, result)
                                print("R3 SELECTOR", profile["name"],
                                      replicate, seed, method,
                                      result["F200"], flush=True)
                                break
                            else:
                                raise RuntimeError(
                                    f"Selector worker failed: {value}")
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
                f"Selector worker exited with {process.exitcode}")


if __name__ == "__main__":
    main()
