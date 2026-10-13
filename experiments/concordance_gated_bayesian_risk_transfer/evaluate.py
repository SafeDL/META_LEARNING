"""Disclose only requested R values to an isolated selector worker."""
import multiprocessing as mp
import time
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiments.driving_benchmark.config import OUTPUT, SUT_IDS
from methods.concordance_gated_bayesian_risk_transfer import select
from methods.historical_expert_mixture.experts import HistoricalExperts
from methods.history_guided_testing.history import load_history, split_indices, subset
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.train import load_kernel

from .config import BUDGET, INITIAL_QUERIES, POLICY, ROOT, SEEDS


class RiskOracle:
    def __init__(self, connection):
        self.connection = connection

    def query(self, index):
        self.connection.send(("query", int(index)))
        return SimpleNamespace(risk=self.connection.recv())


def worker(connection):
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            while True:
                message, task = connection.recv()
                if message == "stop":
                    return
                result = select(**task, oracle=RiskOracle(connection))
                connection.send(("result", result))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def disclose_risk(connection, task, risk):
    connection.send(("task", task))
    selected, disclosures = [], []
    while True:
        message, value = connection.recv()
        if message == "query":
            index = int(value)
            if index not in range(len(risk)) or index in selected or len(selected) >= BUDGET:
                raise ValueError("Invalid or repeated paid risk query")
            selected.append(index)
            disclosures.append({"index": index, "risk": float(risk[index]), "collision": None})
            connection.send(float(risk[index]))
        elif message == "result":
            if value["selected_indices"] != selected or len(selected) != BUDGET:
                raise ValueError("Selector result differs from actual paid queries")
            for reported, actual in zip(value["queries"], disclosures):
                if reported["index"] != actual["index"] or reported["risk"] != actual["risk"]:
                    raise ValueError("Reported risk differs from the disclosure")
            return {**value, "disclosures": disclosures}
        else:
            raise RuntimeError(value)


def main():
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    experts = HistoricalExperts(subset(history, train))
    write_json(ROOT / "protocol.json", {
        "method": "CG-BRT: Concordance-Gated Bayesian Risk Transfer",
        "positive_rule": "measured R > 0.5, strictly greater",
        "selection_seeds": list(SEEDS), "suts": list(SUT_IDS),
        "pools": 2, "budget": BUDGET, "initial_queries": INITIAL_QUERIES,
        "historical_sources": list(experts.history), "source_components": 6,
        "online_feedback": "queried R only; native collision withheld from selector",
        "reference": "Gaussian GP with original pooled historical mean and shared kernel",
        "initialization": "same original ten-query risk acquisition, included in budget",
        "decision": "risk-threshold posterior set inclusion and correlated replacement loss",
        "budget_phase_switching": False,
        "neutral_mixture_component": False, "conditional_event_model": False,
        "new_physical_calls": 0,
    })
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=worker, args=(child,))
    process.start()
    child.close()
    completed, new_runs = 0, 0
    try:
        with threadpool_limits(limits=1):
            for seed in SEEDS:
                for pool in (0, 1):
                    with np.load(OUTPUT / f"pool_{pool}/{SUT_IDS[0]}/responses.npz") as bank:
                        x = bank["x"].copy()
                    prediction = experts.predict(x)
                    with torch.no_grad():
                        covariance = load_kernel(seed, "global_feedback").covariance(x).numpy()
                    task = {
                        "x": x, "covariance": covariance, "priors": prediction.mean,
                        "probability": prediction.group_weights,
                        "fixed_prior": prediction.fixed_mean(), "seed": seed,
                    }
                    for sut in SUT_IDS:
                        path = ROOT / f"driving/pool_{pool}/{sut}/{POLICY}_{seed}.json"
                        with np.load(OUTPUT / f"pool_{pool}/{sut}/responses.npz") as bank:
                            np.testing.assert_array_equal(x, bank["x"])
                            if path.exists():
                                result = read_json(path)
                            else:
                                started = time.perf_counter()
                                result = disclose_risk(parent, task, bank["risk"])
                                result.update(method=POLICY, selector_elapsed_s=time.perf_counter() - started)
                                write_json(path, result)
                                new_runs += 1
                        reference = read_json(OUTPUT / f"pool_{pool}/{sut}/baselines/frozen_original_{seed}.json")
                        if result["selected_indices"][:INITIAL_QUERIES] != reference["selected_indices"][:INITIAL_QUERIES]:
                            raise ValueError("Shared initialization differs from the frozen benchmark")
                        completed += 1
                        write_json(ROOT / "status.json", {
                            "status": "running" if completed < 60 else "complete",
                            "completed_trajectories": completed, "planned_trajectories": 60,
                            "new_selector_runs": new_runs, "new_physical_calls": 0,
                        })
                        print(f"CG-BRT {completed}/60 seed={seed} pool={pool} sut={sut}", flush=True)
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join()
        if process.exitcode != 0:
            raise RuntimeError("Risk-only selector worker failed")


if __name__ == "__main__":
    main()
