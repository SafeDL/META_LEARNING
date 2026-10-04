"""Run methods with identical budgets and disclose only their requested feedback."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import torch

from methods.ras_frt_uq.unified import select as select_ras
from .baseline import METHODS, select
from .config import BUDGET, ROOT, SEEDS
from .history import historical_risk, load_history, split_indices, subset
from .io import read_json, write_json
from .kernel import MODES
from .scenarios import parameter_cells
from .search import TestingSession, select_session


ABLATIONS = {"adaptive": "Adaptive multiscale", "fixed_multiscale": "Fixed multiscale",
             "single_scale": "Single scale", "global_feedback": "Untapered feedback",
             "mixed_acquisition": "Mixed QD/risk acquisition", "single_source": "Single-source prior",
             "constant_prior": "Constant prior"}
COMPARISON_METHODS = {**METHODS, "ras_frt_uq": "RAS-FRT-UQ"}


def choose_kernel():
    destination = ROOT / "models/selection.json"
    if destination.exists():
        return read_json(destination)
    errors = {mode: [] for mode in MODES}
    for seed in SEEDS:
        state = torch.load(ROOT / "models" / f"seed_{seed}.pt", weights_only=True)
        for mode in MODES:
            errors[mode].append(state["validation_rmse"][mode])
    means = {mode: float(np.mean(values)) for mode, values in errors.items()}
    selected = min(means, key=means.get)
    value = {"active_kernel": selected, "historical_validation_rmse": means,
             "per_seed_validation_rmse": errors,
             "selection_rule": "lowest five-seed mean historical group-balanced validation RMSE",
             "kernel_target_labels_used": False,
             "final_target_labels_used_for_selection": False}
    value["active_acquisition"] = "risk_only"
    value["acquisition_selection_rule"] = "fixed pure risk acquisition"
    write_json(destination, value)
    print("Historical validation selected:", selected, means, flush=True)
    return value


class RemoteOracle:
    def __init__(self, connection):
        self.connection = connection
        self.count = 0

    def query(self, index):
        self.connection.send(("query", int(index)))
        risk, collision = self.connection.recv()
        self.count += 1
        return SimpleNamespace(risk=risk, valid_risk=risk is not None, collision=collision,
                               scenario_id=f"target:{index}", query_number=self.count)


def selector_worker(connection, method, seed, x, prior, active):
    try:
        torch.set_num_threads(1)
        oracle = RemoteOracle(connection)
        if method == "ras_frt_uq":
            result = select_ras(x, parameter_cells(x), oracle, seed, budget=BUDGET)
        elif method in METHODS:
            result = select(method, x, parameter_cells(x), oracle, seed,
                            budget=BUDGET, history_scores=prior, cell_count=512)
        else:
            mode = method if method in MODES else active
            if method == "single_source":
                history = load_history()
                train, _ = split_indices(next(iter(history.values()))["x"])
                prior = historical_risk(subset({"idm_reactive": history["idm_reactive"]}, train), x)
            elif method == "constant_prior":
                history = load_history()
                train, _ = split_indices(next(iter(history.values()))["x"])
                prior = np.full(len(x), np.mean([source["risk"][train].mean() for source in history.values()]))
            use_risk_only = method != "mixed_acquisition"
            session = TestingSession(x, seed, mode=mode, prior=prior, risk_only=use_risk_only)
            result = select_session(session, oracle)
        connection.send(("result", result))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def run_one(arguments):
    method, seed, x, risk, collision, prior, active = arguments
    destination = ROOT / "comparison" / f"{method}_{seed}.json"
    if destination.exists():
        return
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(
        target=selector_worker, args=(child, method, seed, x, prior, active))
    process.start()
    child.close()
    disclosed = set()
    try:
        while True:
            message, value = parent.recv()
            if message == "query":
                if value in disclosed or len(disclosed) >= BUDGET:
                    raise ValueError("duplicate query or exhausted testing budget")
                disclosed.add(value)
                if method == "ras_frt_uq":
                    parent.send((None, bool(collision[value])))
                else:
                    parent.send((float(risk[value]), None))
            elif message == "result":
                if len(value["selected_indices"]) != BUDGET or set(value["selected_indices"]) != disclosed:
                    raise ValueError("selection and risk disclosure counts differ")
                write_json(destination, {**value, "method_id": method, "seed": seed,
                           "feedback": ("queried binary collision only" if method == "ras_frt_uq"
                                        else "queried continuous risk only"), "budget": BUDGET})
                print("Comparison complete:", method, seed, flush=True)
                break
            else:
                raise RuntimeError(value)
    finally:
        parent.close()
        if process.is_alive():
            process.join(timeout=10)
        if process.is_alive():
            process.terminate()
        process.join()


def compare():
    selection = choose_kernel()
    active = selection["active_kernel"]
    bank = np.load(ROOT / "target/responses.npz")
    x, risk, collision = bank["x"], bank["risk"], bank["collision"]
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    prior = historical_risk(subset(history, train), x)
    methods = ["bas", "rf_bo", *[name for name in COMPARISON_METHODS if name not in ("bas", "rf_bo")],
               "main", *[name for name in ABLATIONS if name != active]]
    write_json(ROOT / "comparison/settings.json", {"methods": methods, "seeds": SEEDS,
               "budget": BUDGET, "active_kernel": active,
               "initialization": "GP/RF baselines share 10 random initial queries; all count toward 200",
               "feedback": {"ras_frt_uq": "queried binary collision only",
                            "other_methods": "queried continuous risk only"}})
    tasks = [(method, seed, x, risk, collision, prior, active) for seed in SEEDS for method in methods]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for _ in executor.map(run_one, tasks):
            pass
    print("All comparisons complete", flush=True)


if __name__ == "__main__":
    compare()
