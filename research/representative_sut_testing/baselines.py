"""Retained baselines on available shared banks; no retraining or target access."""
import multiprocessing as mp
import hashlib
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from methods.history_guided_testing.baseline import METHODS as STANDARD, select
from methods.history_guided_testing.experiment import RemoteOracle
from methods.history_guided_testing.history import historical_risk, load_history, split_indices, subset
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.scenarios import parameter_cells
from research.behavior_response_testing.model import BehaviorResponseModel
from research.behavior_response_testing.prediction import response_tables
from research.behavior_response_testing.train import OUTPUT as MODEL_OUTPUT
from research.history_response_testing.history_model import predict
from research.history_response_testing.scenarios import ras_predictions
from research.risk_conditioned_response_testing.confirm import ROUND_TWO, select_task
from research.risk_conditioned_response_testing.confirmation import CONFIRMATION

from .config import BUDGET, OUTPUT, POOL_SEEDS, SEEDS, SUT_IDS
from .evaluation import metrics


RETAINED = (
    *STANDARD, "ras_frt_uq", "frozen_original", "previous_best",
    "risk_conditioned", "behavior_posterior", "matched_collision_gp",
)


def verify_baseline_inputs(protocol):
    root = Path(__file__).resolve().parents[2]
    lock = read_json(CONFIRMATION / "lock.json")
    if hashlib.sha256((CONFIRMATION / "protocol.json").read_bytes()).hexdigest() != lock["protocol_sha256"]:
        raise ValueError("Frozen baseline protocol differs from its lock")
    archived_document = "docs/Attachment_Paper_Review_and_Research_Directions.md"
    verified, documentation = [], []
    with ZipFile(CONFIRMATION / "locked_inputs.zip") as archive:
        for name, expected in protocol["sha256"].items():
            path = root / name
            if path.exists():
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"Frozen baseline input changed: {name}")
                verified.append(name)
            elif name == archived_document:
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError("Archived historical review does not match frozen inputs")
                documentation.append(name)
            else:
                raise FileNotFoundError(f"Missing frozen baseline runtime input: {name}")
    write_json(OUTPUT / "baseline_input_audit.json", {
        "current_inputs_matched": len(verified), "archived_documentation_matched": documentation,
        "runtime_input_changes": [],
        "scope": "One removed historical Markdown review is checked in its original archive; all runtime inputs must match current files",
    })


def worker(connection):
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            while True:
                message, task = connection.recv()
                if message == "stop":
                    return
                if task["method"] in STANDARD:
                    result = select(task["method"], task["x"], parameter_cells(task["x"]),
                                    RemoteOracle(connection), task["seed"],
                                    history_scores=task["prior"], cell_count=512)
                else:
                    result = select_task(connection, task)
                connection.send(("result", result))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def task_predictions(x, seed, candidate, historical):
    original = predict(x, seed)
    ras = ras_predictions(x, seed)
    states = torch.load(MODEL_OUTPUT / "models" / f"predictor_{seed}.pt",
                        map_location="cuda", weights_only=True)
    models = []
    for family in (0, 1):
        model = BehaviorResponseModel().cuda().eval()
        model.load_state_dict(states[str(family)])
        models.append(model)
    grid = np.asarray(candidate["continuous_behaviors"], dtype=np.float32)
    grid_prediction = tuple(value.cpu().numpy() for value in
                            response_tables(x, grid, models, return_logits=True))
    history_table = response_tables(x, historical, models, return_logits=True)
    return {
        "original": original, "ras": ras, "grid_prediction": grid_prediction,
        "historical_prediction": (history_table[0].cpu().numpy().T,
                                  history_table[1].sigmoid().cpu().numpy().T),
        "matched_calibrators": candidate["matched_collision_calibrators"],
        "discrepancy": candidate["risk_discrepancy"][str(seed)],
        "decoder": candidate["risk_conditioned_decoder"][str(seed)],
    }


def run_disclosures(connection, task, bank):
    connection.send(("task", task))
    selected, observations = [], []
    while True:
        message, value = connection.recv()
        if message == "query":
            index = int(value)
            if index not in range(len(bank["x"])) or index in selected or len(selected) >= BUDGET:
                raise ValueError("Invalid, duplicate or over-budget baseline query")
            selected.append(index)
            if task["method"] == "ras_frt_uq":
                pair = None, bool(bank["collision"][index])
            else:
                pair = float(bank["risk"][index]), None
            observations.append({"index": index, "risk": pair[0], "collision": pair[1]})
            connection.send(pair)
        elif message == "result":
            if value["selected_indices"] != selected or len(selected) != BUDGET:
                raise ValueError("Baseline result differs from disclosure sequence")
            return {**value, **metrics(bank["collision"], selected),
                    "disclosures": observations}
        else:
            raise RuntimeError(value)


def main():
    torch.set_num_threads(1)
    frozen_protocol = read_json(CONFIRMATION / "protocol.json")
    verify_baseline_inputs(frozen_protocol)
    candidate = frozen_protocol["candidate"]
    historical = np.asarray(read_json(ROUND_TWO / "protocol.json")["candidate"][
        "historical_behaviors"], dtype=np.float32)
    history = load_history()
    training, _ = split_indices(next(iter(history.values()))["x"])
    allowed = subset(history, training)
    write_json(OUTPUT / "baseline_protocol.json", {
        "methods": RETAINED, "budget": BUDGET, "seeds": SEEDS,
        "scene_pools": "same coordinates and physical banks as the new method",
        "retraining": False,
        "feedback": {method: "queried collision only" if method == "ras_frt_uq"
                     else "queried risk only" for method in RETAINED},
        "source_information": {
            "standard_and_original": "six original sources; frozen historical split",
            "behavior_models": "additional 48 public IDM/FVDM configurations; engineering reference",
            "mechanism_controls": "reported separately with identical source data and both feedback values",
        },
        "working_model_limit": "behavior hypotheses describe IDM/FVDM, not VI/MCTS/PPO controllers",
        "stage": "development; missing banks are listed rather than imputed",
    })
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=worker, args=(child,))
    process.start()
    child.close()
    missing = []
    try:
        for pool_id in range(len(POOL_SEEDS)):
            available = []
            for sut_id in SUT_IDS:
                path = OUTPUT / f"pool_{pool_id}" / sut_id / "responses.npz"
                if path.exists():
                    available.append(sut_id)
                else:
                    missing.append({"pool": pool_id, "sut_id": sut_id})
            if not available:
                continue
            with np.load(OUTPUT / f"pool_{pool_id}" / available[0] / "responses.npz") as bank:
                x = bank["x"].copy()
            prior = historical_risk(allowed, x)
            for seed in SEEDS:
                pending = any(not (OUTPUT / f"pool_{pool_id}" / sut_id / "baselines" /
                                   f"{method}_{seed}.json").exists()
                              for sut_id in available for method in RETAINED)
                if not pending:
                    continue
                predictions = task_predictions(x, seed, candidate, historical)
                for sut_id in available:
                    folder = OUTPUT / f"pool_{pool_id}" / sut_id
                    with np.load(folder / "responses.npz") as bank:
                        if not np.array_equal(x, bank["x"]):
                            raise ValueError("Cross-SUT candidate coordinates differ")
                        for method in RETAINED:
                            path = folder / "baselines" / f"{method}_{seed}.json"
                            if path.exists():
                                continue
                            task = {"method": method, "seed": seed, "x": x,
                                    "prior": prior, **predictions}
                            started = time.perf_counter()
                            result = run_disclosures(parent, task, bank)
                            result.update(method=method, seed=seed,
                                          selector_elapsed_s=time.perf_counter() - started)
                            write_json(path, result)
                            print("BASELINE", pool_id, sut_id, seed, method,
                                  result["F200"], flush=True)
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Baseline selector exited {process.exitcode}")
    write_json(OUTPUT / "baseline_progress.json", {
        "status": "incomplete_missing_banks" if missing else "complete",
        "missing_banks": missing, "expected_runs": len(POOL_SEEDS) * len(SUT_IDS) * len(SEEDS) * len(RETAINED),
        "rerun_after_measurement": bool(missing),
    })


if __name__ == "__main__":
    main()
