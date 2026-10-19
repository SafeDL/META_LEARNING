"""Historical training, frozen stopping decisions and 36 isolated replays."""

import multiprocessing as mp
import platform
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from methods.baselines.hgt.io import read_json, write_json
from methods.common.features import prepare_pool
from benchmark.data import bank_path, measured_seconds
from benchmark.risk_channel import disclose_risk
from benchmark.timing import QueryClock, TimedRiskOracle, now
from methods.common.snapshots import ADDED_SOURCES

from ..adaptive.config import ROOT, SEED, STEPS, THRESHOLDS
from ..adaptive.controller import StopController
from ..adaptive.data import generate
from ..adaptive.train import load_network, train
from ..selection import select
from ..train import load_value
from .by_family import (
    BUDGET,
    FAMILIES,
    INITIAL,
    PROJECT,
    ROOT as FAMILY_ROOT,
    SUT_IDS,
    build_family,
    workspace_for,
)

INTERFACES = {
    "methods/baselines/lrt/selection.py",
    "benchmark/timing.py",
    "benchmark/risk_channel.py",
}


def freeze_interfaces():
    destination = ROOT / "reviewed_interfaces.zip"
    paths = sorted(INTERFACES | ADDED_SOURCES)
    ROOT.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with ZipFile(destination) as archive:
            for relative in paths:
                if archive.read(relative) != (PROJECT / relative).read_bytes():
                    raise ValueError(f"Reviewed stopping interface changed: {relative}")
        return
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for relative in paths:
            archive.write(PROJECT / relative, relative)


def stage(family, name, action):
    path = ROOT / family / "build_cost.json"
    ledger = read_json(path) if path.exists() else {"stages": {}, "physical_calls": 0}
    previous = ledger["stages"].get(name, {})
    if previous.get("status") == "complete":
        return
    started = now()
    ledger["stages"][name] = {
        "status": "running",
        "elapsed_s": previous.get("elapsed_s", 0),
    }
    write_json(path, ledger)
    try:
        action()
    except BaseException as error:
        ledger["stages"][name].update(status="failed", error=repr(error))
        raise
    else:
        ledger["stages"][name]["status"] = "complete"
    finally:
        ledger["stages"][name]["elapsed_s"] += now() - started
        write_json(path, ledger)
    print(
        f"BUILD STOP {family} {name} {ledger['stages'][name]['elapsed_s']:.2f}s",
        flush=True,
    )


def freeze(workspaces):
    paths = sorted(
        p
        for p in (PROJECT / "methods").rglob("*.py")
        if p.name != "report.py" and not p.name.endswith("_report.py")
    )
    for workspace in workspaces:
        paths.extend(
            (
                workspace.kernel_root / "models/seed_11.pt",
                workspace.encoder_root / "models/decision_11.pt",
                workspace.root / "models/full_11.pt",
                ROOT / workspace.root.name / "model.pt",
                ROOT / workspace.root.name / "training.json",
            )
        )
    destination = ROOT / "frozen_inputs.zip"
    if destination.exists():
        with ZipFile(destination) as archive:
            for path in paths:
                if (
                    archive.read(path.relative_to(PROJECT).as_posix())
                    != path.read_bytes()
                ):
                    raise ValueError(f"Adaptive input changed after freezing: {path}")
        return
    write_json(
        ROOT / "protocol.json",
        {
            "seed": SEED,
            "families": FAMILIES,
            "systems": SUT_IDS,
            "pools": [0, 1, 2],
            "maximum_queries": BUDGET,
            "initial_queries": INITIAL,
            "training_updates": STEPS,
            "threshold_candidates": THRESHOLDS,
            "retention": 0.9,
            "formal_trajectories": 36,
            "new_physical_calls": 0,
            "target_data_used_for_selection": False,
            "existing_fixed_trajectories_reused": True,
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "online_workers": 1,
                "torch_threads": 1,
            },
        },
    )
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in paths + [ROOT / "protocol.json"]:
            archive.write(path, path.relative_to(PROJECT).as_posix())


def selector_worker(connection):
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            while True:
                message, task = connection.recv()
                if message == "stop":
                    break
                clock = QueryClock()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                workspace = SimpleNamespace(root=FAMILY_ROOT / task["family"])
                value = load_value(SEED, device=device, root=workspace.root)
                stopper = None
                if not task.get("fixed", False):
                    network, threshold = load_network(workspace, value_model=value)
                    stopper = StopController(network, threshold)
                result = select(
                    **task["common"],
                    transfer_prior=task["prior"],
                    value_model=value,
                    oracle=TimedRiskOracle(connection, clock),
                    clock=clock,
                    stopper=stopper,
                )
                result["timing"] = clock.report()
                connection.send(("result", result))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def evaluate(workspaces):
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=selector_worker, args=(child,))
    process.start()
    child.close()
    try:
        for workspace in workspaces:
            family = workspace.root.name
            for pool in (0, 1, 2):
                cache = prepare_pool(workspace, pool)
                for sut in SUT_IDS:
                    path = (
                        ROOT / family / "trajectories" / f"pool_{pool}" / f"{sut}.json"
                    )
                    if path.exists():
                        continue
                    with np.load(bank_path(pool, sut)) as bank:
                        keep = bank["x"][:, 4] == workspace.family
                        np.testing.assert_array_equal(
                            cache["common"]["x"], bank["x"][keep]
                        )
                        risk, ids = (
                            bank["risk"][keep].copy(),
                            bank["scenario_id"][keep].copy(),
                        )
                    durations = measured_seconds(pool, sut, ids)
                    fixed_path = (
                        workspace.root
                        / "trajectories"
                        / f"pool_{pool}"
                        / sut
                        / "lrt_11.json"
                    )
                    if not fixed_path.exists():
                        fixed = disclose_risk(
                            parent,
                            {
                                "family": family,
                                "common": cache["common"],
                                "prior": cache["prior"],
                                "fixed": True,
                            },
                            risk,
                            process,
                        )
                        indices = fixed["selected_indices"]
                        fixed_found = np.cumsum(risk[indices] > 0.5).tolist()
                        fixed.update(
                            family=family,
                            pool=pool,
                            sut=sut,
                            seed=SEED,
                            method="lrt",
                            positives=int((risk > 0.5).sum()),
                            found=fixed_found,
                            H200=fixed_found[-1],
                            recall=(
                                fixed_found[-1] / int((risk > 0.5).sum())
                                if (risk > 0.5).any()
                                else None
                            ),
                            scenario_ids=ids[indices].tolist(),
                            execution_s=durations[indices].tolist(),
                            physical_calls=0,
                        )
                        write_json(fixed_path, fixed)
                    result = disclose_risk(
                        parent,
                        {
                            "family": family,
                            "common": cache["common"],
                            "prior": cache["prior"],
                        },
                        risk,
                        process,
                        minimum_queries=INITIAL,
                    )
                    indices = result["selected_indices"]
                    fixed = read_json(fixed_path)
                    if (
                        indices != fixed["selected_indices"][: len(indices)]
                        or result["queries"] != fixed["queries"][: len(indices)]
                    ):
                        raise ValueError(
                            "Adaptive selection must preserve the exact fixed-policy prefix"
                        )
                    found = np.cumsum(risk[indices] > 0.5).tolist()
                    result.update(
                        family=family,
                        pool=pool,
                        sut=sut,
                        seed=SEED,
                        positives=int((risk > 0.5).sum()),
                        found=found,
                        H_tau=found[-1],
                        recall=(
                            found[-1] / int((risk > 0.5).sum())
                            if (risk > 0.5).any()
                            else None
                        ),
                        reference_H200=fixed["H200"],
                        scenario_ids=ids[indices].tolist(),
                        execution_s=durations[indices].tolist(),
                        physical_calls=0,
                    )
                    write_json(path, result)
                    print(
                        f"ADAPTIVE {family} {pool} {sut} n={len(indices)} found={found[-1]}/{fixed['H200']}",
                        flush=True,
                    )
                    write_json(
                        ROOT / "status.json",
                        {
                            "status": "running",
                            "stage": "evaluation",
                            "completed_trajectories": len(
                                list(ROOT.glob("*/trajectories/*/*.json"))
                            ),
                            "physical_calls": 0,
                        },
                    )
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join()
        if process.exitcode:
            raise RuntimeError("Adaptive selector worker failed")


def main():
    torch.set_num_threads(1)
    freeze_interfaces()
    workspaces = [workspace_for(family) for family in FAMILIES]
    with threadpool_limits(limits=1):
        for workspace in workspaces:
            required = (
                workspace.kernel_root / "models/seed_11.pt",
                workspace.encoder_root / "models/decision_11.pt",
                workspace.root / "models/full_11.pt",
            )
            if not all(p.exists() for p in required):
                build_family(workspace)
            stage(
                workspace.root.name,
                "historical_trajectories",
                lambda: generate(workspace),
            )
            stage(workspace.root.name, "stopping_network", lambda: train(workspace))
        eligible = {
            w.root.name: read_json(ROOT / w.root.name / "training.json")["selected"][
                "eligible"
            ]
            for w in workspaces
        }
        if not all(eligible.values()):
            write_json(
                ROOT / "status.json",
                {
                    "status": "historical_validation_failed",
                    "eligible": eligible,
                    "physical_calls": 0,
                },
            )
            print(f"Historical validation failed: {eligible}", flush=True)
            return
        freeze(workspaces)
        evaluate(workspaces)
        from .adaptive_report import main as report

        report()
    write_json(
        ROOT / "status.json",
        {"status": "complete", "formal_trajectories": 36, "physical_calls": 0},
    )


if __name__ == "__main__":
    main()
