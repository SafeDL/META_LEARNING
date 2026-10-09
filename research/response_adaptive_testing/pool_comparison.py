"""Development comparisons with selector feedback separated from evaluation truth."""
import time
from types import SimpleNamespace

import numpy as np
import torch

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

from .config import BUDGET, OUTPUT, SEEDS
from .develop import metrics, run_one


class FeedbackOracle:

    def __init__(self, bank, binary=False):
        self.bank, self.binary, self.selected = bank, binary, []

    def query(self, index):
        index = int(index)
        if (not 0 <= index < len(self.bank["x"]) or index in self.selected
                or len(self.selected) == BUDGET):
            raise ValueError("Duplicate query or exhausted budget")
        self.selected.append(index)
        risk = None if self.binary else float(self.bank["risk"][index])
        collision = bool(
            self.bank["collision"][index]) if self.binary else None
        return SimpleNamespace(risk=risk,
                               collision=collision,
                               valid_risk=not self.binary,
                               query_number=len(self.selected),
                               scenario_id=str(index))


def previous_selection(method, bank, seed, prediction, calibration, ras=None):
    oracle = FeedbackOracle(bank, binary=method == "ras_frt_uq")
    x, risks, collisions = bank["x"], *prediction
    if method == "ras_frt_uq":
        result = select_from_responses(x, parameter_cells(x), ras, oracle,
                                       seed)
        selected = result["selected_indices"]
    elif method == "frozen_original":
        history = load_history()
        train, _ = split_indices(next(iter(history.values()))["x"])
        prior = historical_risk(subset(history, train), x)
        result = select_session(
            TestingSession(x, seed, mode="global_feedback", prior=prior),
            oracle)
        selected = result["selected_indices"]
    elif method == "class_rank":
        selected = np.argsort(-(collisions @ np.asarray(SOURCE_WEIGHTS)),
                              kind="stable")[:BUDGET].tolist()
        for index in selected:
            oracle.query(index)
    else:
        session = RiskTestingSession(x, covariance(x, risks),
                                     risks, collisions,
                                     np.asarray(SOURCE_WEIGHTS), calibration)
        selected = []
        while (index := session.next_index()) is not None:
            session.observe(oracle.query(index).risk)
            selected.append(index)
    if selected != oracle.selected:
        raise ValueError("Saved choices differ from disclosed feedback")
    return {
        **metrics(bank["collision"], selected), "selected_indices": selected,
        "feedback": "binary collision" if oracle.binary else "continuous risk"
    }


def main():
    torch.set_num_threads(1)
    screen = read_json(OUTPUT / "development" / "summary.json")
    best = sorted(screen,
                  key=lambda r: (r["F200"], r["mean_cumulative_collisions"]),
                  reverse=True)[:4]
    calibration = calibrators()
    records = []
    for pool in sorted((OUTPUT / "development_pools").glob("*/responses.npz")):
        bank = np.load(pool)
        name = pool.parent.name
        for seed in SEEDS:
            prediction = predict(bank["x"], seed)
            for method in ("previous_best", "class_rank", "frozen_original",
                           "ras_frt_uq"):
                path = pool.parent / "comparison" / f"{method}_{seed}.json"
                if path.exists():
                    result = read_json(path)
                else:
                    started = time.perf_counter()
                    ras = ras_predictions(
                        bank["x"], seed) if method == "ras_frt_uq" else None
                    result = previous_selection(method, bank, seed, prediction,
                                                calibration, ras)
                    result["elapsed_s"] = time.perf_counter() - started
                    write_json(path, result)
                records.append({
                    "pool": name,
                    "seed": seed,
                    "method": method,
                    **{
                        key: result[key]
                        for key in ("mean_cumulative_collisions", "F200", "pool_collisions", "recall")
                    }
                })
                print("POOL",
                      name,
                      seed,
                      method,
                      round(result["mean_cumulative_collisions"], 3),
                      result["F200"],
                      result["pool_collisions"],
                      flush=True)
            for number, item in enumerate(best):
                for horizon in (0, 1, 5):
                    method = f"joint_{number}_{horizon}"
                    options = {
                        **item["options"], "lookahead": horizon > 0,
                        "horizon": horizon
                    }
                    path = pool.parent / "comparison" / f"{method}_{seed}.json"
                    if path.exists():
                        result = read_json(path)
                    else:
                        result = run_one(bank["x"], bank["risk"],
                                         bank["collision"], prediction,
                                         options)
                        write_json(path, result)
                    records.append({
                        "pool": name,
                        "seed": seed,
                        "method": method,
                        **{
                            key: result[key]
                            for key in ("mean_cumulative_collisions", "F200", "pool_collisions", "recall")
                        }
                    })
                    print("POOL",
                          name,
                          seed,
                          method,
                          round(result["mean_cumulative_collisions"], 3),
                          result["F200"],
                          result["pool_collisions"],
                          flush=True)
    write_json(OUTPUT / "development_pools" / "summary.json", records)


if __name__ == "__main__":
    main()
