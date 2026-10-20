"""Selectors execute in a process that cannot access unqueried target responses."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from benchmark.data import bank_path
from benchmark.risk_channel import RiskOracle, disclose_risk
from methods.baselines.cg_brt import select as original_select
from methods.baselines.dart import select
from methods.baselines.dart.encoder import fixed_context
from methods.baselines.dart.train import load_encoder
from methods.baselines.hem.experts import HistoricalExperts
from methods.baselines.hgt.history import load_history, split_indices, subset
from methods.baselines.hgt.io import write_json
from methods.baselines.hgt.train import load_kernel

from .config import ROOT, SEEDS, SUT_IDS, VARIANTS, pools, variants


def worker(connection):
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            while True:
                message, task = connection.recv()
                if message == "stop":
                    return
                policy = task.pop("policy")
                selector = original_select if policy == "cg_brt" else select
                result = selector(**task, oracle=RiskOracle(connection))
                connection.send(("result", result))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def trace_path(stage, pool, sut, policy, seed):
    if stage == "development" and policy == "cg_brt":
        return (
            ROOT.parent
            / "cg_brt"
            / "driving"
            / f"pool_{pool}"
            / sut
            / f"cg_brt_{seed}.json"
        )
    return ROOT / stage / f"pool_{pool}" / sut / f"{policy}_{seed}.json"


def main(stage="development", selection_seeds=SEEDS):
    if stage == "confirmation":
        from .freeze import verify_freeze

        verify_freeze()
    torch.set_num_threads(1)
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    sources = subset(history, train)
    experts = HistoricalExperts(sources)
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=worker, args=(child,))
    process.start()
    child.close()
    planned = (
        len(pools(stage)) * len(selection_seeds) * len(SUT_IDS) * len(variants(stage))
    )
    completed = 0
    try:
        with threadpool_limits(limits=1):
            for seed in selection_seeds:
                context = fixed_context(sources, seed)
                encoders = {
                    mode: load_encoder(
                        seed, mode, "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    for mode in ("decision", "conventional")
                }
                for pool in pools(stage):
                    with np.load(bank_path(pool, SUT_IDS[0])) as bank:
                        x = bank["x"].copy()
                    prediction = experts.predict(x)
                    with torch.no_grad():
                        covariance = (
                            load_kernel(seed, "global_feedback").covariance(x).numpy()
                        )
                        learned = {
                            mode: model.covariance(x, context, covariance).cpu().numpy()
                            for mode, model in encoders.items()
                        }
                    common = {
                        "x": x,
                        "covariance": covariance,
                        "priors": prediction.mean,
                        "probability": prediction.group_weights,
                        "fixed_prior": prediction.fixed_mean(),
                        "seed": seed,
                    }
                    for sut in SUT_IDS:
                        with np.load(bank_path(pool, sut)) as bank:
                            np.testing.assert_array_equal(x, bank["x"])
                            risk = bank["risk"].copy()
                        for policy in variants(stage):
                            path = trace_path(stage, pool, sut, policy, seed)
                            if not path.exists():
                                task = {**common, "policy": policy, **VARIANTS[policy]}
                                if policy != "cg_brt":
                                    mode = (
                                        "conventional"
                                        if policy == "conventional_training"
                                        else "decision"
                                    )
                                    task["proposal_covariance"] = (
                                        covariance
                                        if policy == "physical_kernel"
                                        else learned[mode]
                                    )
                                started = time.perf_counter()
                                result = disclose_risk(parent, task, risk, process)
                                result.update(
                                    method=policy,
                                    stage=stage,
                                    pool=pool,
                                    sut=sut,
                                    seed=seed,
                                    selector_elapsed_s=time.perf_counter() - started,
                                )
                                write_json(path, result)
                            completed += 1
                            if len(selection_seeds) == len(SEEDS):
                                write_json(
                                    ROOT / "status.json",
                                    {
                                        "stage": stage,
                                        "status": (
                                            "complete"
                                            if completed == planned
                                            else "running"
                                        ),
                                        "completed": completed,
                                        "planned": planned,
                                        "scope_seeds": list(selection_seeds),
                                    },
                                )
                            print(
                                f"EVALUATE {stage} {completed}/{planned} {policy} pool={pool} sut={sut} seed={seed}",
                                flush=True,
                            )
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join()
        if process.exitcode != 0:
            raise RuntimeError("Isolated selector worker failed")


def evaluate_seed(arguments):
    stage, seed = arguments
    main(stage, (seed,))


def run_parallel(stage="development", selection_seeds=SEEDS):
    planned = (
        len(pools(stage)) * len(selection_seeds) * len(SUT_IDS) * len(variants(stage))
    )
    write_json(
        ROOT / "status.json",
        {
            "stage": stage,
            "status": "running",
            "scope_seeds": list(selection_seeds),
            "planned": planned,
        },
    )
    with ProcessPoolExecutor(
        max_workers=3, mp_context=mp.get_context("spawn")
    ) as executor:
        list(executor.map(evaluate_seed, [(stage, seed) for seed in selection_seeds]))
    write_json(
        ROOT / "status.json",
        {
            "stage": stage,
            "status": "complete",
            "scope_seeds": list(selection_seeds),
            "completed": planned,
            "planned": planned,
        },
    )


if __name__ == "__main__":
    run_parallel()
