"""Existing response banks remain in the evaluator, outside selector processes."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from methods.baselines.cg_brt import select as cg_select
from methods.baselines.dart.evaluation.config import bank_root, SUT_IDS
from methods.baselines.dart.evaluation.evaluate import trace_path as dart_trace_path
from methods.baselines.dart.selection import select as dart_select
from methods.baselines.hgt.config import GROUPS
from methods.baselines.hgt.history import load_history, split_indices, subset
from methods.baselines.hgt.io import write_json
from benchmark.risk_channel import RiskOracle, disclose_risk

from ..config import ROOT, SEEDS, TRAIN_MODES, VARIANTS
from ..data import assemble
from methods.common.prior import visible_history
from ..selection import select
from ..train import load_value
from .protocol import freeze


def bank_path(pool, sut):
    return (
        bank_root("confirmation" if pool == 2 else "development")
        / f"pool_{pool}"
        / sut
        / "responses.npz"
    )


def trace_path(pool, sut, seed, variant, condition="complete"):
    if condition == "complete" and variant in ("cg_brt", "previous"):
        return dart_trace_path(
            "confirmation" if pool == 2 else "development",
            pool,
            sut,
            "cg_brt" if variant == "cg_brt" else "full",
            seed,
        )
    stage = "main" if condition == "complete" else "stress"
    return ROOT / stage / condition / f"pool_{pool}" / sut / f"{variant}_{seed}.json"


def case_variants(pool, condition):
    if condition == "complete":
        return (
            ("cg_brt", "previous", *VARIANTS)
            if pool != 2
            else ("cg_brt", "previous", "full")
        )
    matched = (
        "no_archive_covariance"
        if condition == "sparse"
        else (
            "global_sources"
            if condition in ("cutin_gap", "brake_gap")
            else "no_target_component"
        )
    )
    return ("cg_brt", "previous", "full", matched)


def cases():
    standard = [(pool, seed, "complete") for pool in (0, 1, 2) for seed in SEEDS]
    stress = [
        (0, seed, condition)
        for seed in SEEDS
        for condition in (
            "sparse",
            "cutin_gap",
            "brake_gap",
            *("without_" + g for g in GROUPS),
        )
    ]
    return standard + stress


def selector_worker(connection):
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            while True:
                command, task = connection.recv()
                if command == "stop":
                    return
                variant = task.pop("policy")
                selector = (
                    cg_select
                    if variant == "cg_brt"
                    else dart_select if variant == "previous" else select
                )
                if task.get("value_model") is not None:
                    task["value_model"].to(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                result = selector(**task, oracle=RiskOracle(connection))
                connection.send(("result", result))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def evaluate_case(case):
    pool, seed, condition = case
    variants = case_variants(pool, condition)
    if all(
        trace_path(pool, sut, seed, variant, condition).exists()
        for sut in SUT_IDS
        for variant in variants
    ):
        return case
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        history = load_history()
        train, _ = split_indices(next(iter(history.values()))["x"])
        sources = visible_history(subset(history, train), condition, seed)
        with np.load(bank_path(pool, SUT_IDS[0])) as bank:
            x = bank["x"].copy()
        common, prior = assemble(x, sources, seed)
        networks = {mode: load_value(seed, mode) for mode in TRAIN_MODES}
        parent, child = mp.get_context("spawn").Pipe()
        process = mp.get_context("spawn").Process(target=selector_worker, args=(child,))
        process.start()
        child.close()
        try:
            for sut in SUT_IDS:
                with np.load(bank_path(pool, sut)) as bank:
                    risk = bank["risk"].copy()
                    np.testing.assert_array_equal(bank["x"], x)
                for variant in variants:
                    destination = trace_path(pool, sut, seed, variant, condition)
                    if destination.exists():
                        continue
                    task = {**common, "policy": variant}
                    if variant == "previous":
                        task["proposal_covariance"] = prior.enhanced_covariance
                    elif variant != "cg_brt":
                        task.update(
                            transfer_prior=prior,
                            variant=variant,
                            value_model=(
                                None
                                if variant == "no_value_learning"
                                else networks[
                                    variant if variant in TRAIN_MODES else "full"
                                ]
                            ),
                        )
                    started = time.perf_counter()
                    trace = disclose_risk(parent, task, risk, process)
                    trace.update(
                        pool=pool,
                        sut=sut,
                        seed=seed,
                        condition=condition,
                        method=variant,
                        selector_elapsed_s=time.perf_counter() - started,
                        numerical_backend=(
                            "frozen_original"
                            if variant in ("cg_brt", "previous")
                            else "integer_student_cdf"
                        ),
                    )
                    write_json(destination, trace)
                    print(
                        f"TARGET {pool} {seed} {condition} {sut} {variant}", flush=True
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
                raise RuntimeError("LRT worker exited with an error")
    return case


def main():
    freeze()
    planned = cases()
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=12, mp_context=mp.get_context("spawn")
    ) as executor:
        for count, case in enumerate(executor.map(evaluate_case, planned), 1):
            write_json(
                ROOT / "status.json",
                {
                    "stage": "evaluation",
                    "status": "complete" if count == len(planned) else "running",
                    "completed_cases": count,
                    "planned_cases": len(planned),
                    "last_case": list(case),
                    "elapsed_s": time.perf_counter() - started,
                    "physical_calls": 0,
                },
            )


if __name__ == "__main__":
    main()
