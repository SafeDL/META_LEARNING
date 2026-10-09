"""Run controlled risk-only policies on the fixed development holdout."""
import hashlib
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from methods.history_guided_testing.experiment import RemoteOracle
from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.prediction import behavior_grid
from research.response_adaptive_testing.develop import metrics

from .config import BUDGET, COHORT, METHODS, MODELS, RESULTS, ROOT, SEEDS
from .model import FailureDecoder, FrozenResponseBackbone
from .planner import BudgetLookaheadSession
from .session import MetaTestingSession


EVALUATION = RESULTS / "development_holdout"


def verify_or_lock():
    """Prevent cached sequences from being reused after changing a learner."""
    path = EVALUATION / "lock.json"
    if path.exists():
        sealed = read_json(path)
        for name, digest in sealed["sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Held-out evaluation inputs changed: {name}")
        return
    inputs = list(ROOT.glob("*.py")) + [RESULTS / "protocol.json"]
    inputs += [RESULTS / "models" / f"decoder_{mode}_{seed}.pt"
               for seed in SEEDS for mode in ("supervised", "meta")]
    inputs += [RESULTS / "models" / f"training_{seed}.json" for seed in SEEDS]
    inputs += [MODELS / f"predictor_{seed}.pt" for seed in SEEDS]
    inputs += [COHORT / "protocol.json"]
    write_json(path, {
        "role": "Fixed development evaluation; data previously disclosed",
        "sha256": {os.path.relpath(item, ROOT): hashlib.sha256(
            item.read_bytes()).hexdigest() for item in inputs},
    })


def selector_task(connection, task):
    oracle = RemoteOracle(connection)
    arguments = (task["x"], task["risk_prediction"], task["collision_logits"],
                 task["discrepancy"], task["decoder"])
    if task["method"].endswith("greedy"):
        session = MetaTestingSession(*arguments, budget=BUDGET)
    else:
        session = BudgetLookaheadSession(
            *arguments, budget=BUDGET,
            constrained=task["method"] != "meta_unconstrained")
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
                raise ValueError(f"Unexpected selector message: {message}")
            connection.send(("result", selector_task(connection, task)))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def run_disclosures(parent, task, risk, collision):
    parent.send(("task", task))
    disclosed, observations = [], []
    started = time.perf_counter()
    while True:
        message, value = parent.recv()
        if message == "query":
            index = int(value)
            if (not 0 <= index < len(risk) or index in disclosed
                    or len(disclosed) == BUDGET):
                raise ValueError("Invalid, repeated or excess selector query")
            disclosed.append(index)
            feedback = float(risk[index])
            parent.send((feedback, None))
            observations.append({"index": index, "continuous_risk": feedback})
        elif message == "result":
            if value["selected_indices"] != disclosed or len(disclosed) != BUDGET:
                raise ValueError("Selected sequence differs from disclosures")
            measured = metrics(collision, disclosed)
            measured.update({f"F{count}": measured["curve"][count - 1]
                             for count in (10, 30, 150)})
            return {**value, **measured, "observations": observations,
                    "selector_elapsed_s": time.perf_counter() - started,
                    "method": task["method"], "seed": task["seed"],
                    "feedback": "one queried continuous risk only",
                    "isolation": "child receives predictions and coordinates; parent retains parameters and outcomes"}
        else:
            raise RuntimeError(f"Selector worker failed: {value}")


def load_models():
    result = {}
    for seed in SEEDS:
        states = torch.load(MODELS / f"predictor_{seed}.pt",
                            map_location="cuda", weights_only=True)
        backbone = FrozenResponseBackbone(states).cuda()
        weight, bias = backbone.initial_head()
        decoders = {}
        for mode in ("supervised", "meta"):
            fitted = torch.load(RESULTS / "models" / f"decoder_{mode}_{seed}.pt",
                                map_location="cuda", weights_only=True)
            decoder = FailureDecoder(weight, bias, fitted["center"], fitted["scale"]).cuda()
            decoder.load_state_dict(fitted)
            decoders[mode] = decoder.eval()
        result[seed] = backbone, decoders
    return result


def main():
    torch.set_num_threads(1)
    protocol = read_json(RESULTS / "protocol.json")
    verify_or_lock()
    models = load_models()
    discrepancy = read_json(COHORT / "protocol.json")["candidate"]["risk_discrepancy"]
    grid = torch.as_tensor(behavior_grid(), device="cuda")
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=selector_worker, args=(child,))
    process.start()
    child.close()
    try:
        for name in protocol["split"]["development_holdout"]:
            for replicate in range(2):
                with np.load(COHORT / name / f"pool_{replicate}" / "responses.npz") as bank:
                    x, risk, collision = (bank[key].copy()
                                          for key in ("x", "risk", "collision"))
                coordinates = torch.as_tensor(x, dtype=torch.float32, device="cuda")
                for seed in SEEDS:
                    backbone, decoders = models[seed]
                    predictions = {mode: tuple(value.cpu().numpy() for value in
                                               backbone.tables(coordinates, grid, decoder))
                                   for mode, decoder in decoders.items()}
                    for method in METHODS:
                        path = EVALUATION / name / f"pool_{replicate}" / f"{method}_{seed}.json"
                        if path.exists():
                            continue
                        mode = "meta" if method.startswith("meta") else "supervised"
                        decoder = decoders[mode]
                        task = {
                            "method": method, "seed": seed, "x": x,
                            "risk_prediction": predictions[mode][0],
                            "collision_logits": predictions[mode][1],
                            "discrepancy": discrepancy[str(seed)],
                            "decoder": {key: getattr(decoder, key).detach().cpu().tolist()
                                        for key in ("slope", "center", "scale")},
                        }
                        result = run_disclosures(parent, task, risk, collision)
                        write_json(path, result)
                        print("META DEVELOPMENT", name, replicate, seed, method,
                              result["F200"], result["selector_elapsed_s"], flush=True)
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Selector worker exited with {process.exitcode}")


if __name__ == "__main__":
    main()
