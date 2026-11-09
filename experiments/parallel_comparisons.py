"""Run independent ADS comparisons concurrently while keeping each task's query stream serial."""

import hashlib
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from controllers.highway.catalog import CURRENT_INDEPENDENT_SUTS

from . import candidate_sensitivity, mechanisms
from .joint_validation import DEFAULT, OUTPUT, freeze_protocol
from .pilot import (
    CONTROLS,
    MAIN_METHODS,
    MECHANISMS,
    fixed_comparison,
    historical_classifiers,
    implementation_contract,
    method_configurations,
)
from .prepare_joint import DATASET_ROOT
from .storage import read_json, write_json


def run_system(root, sut, suite, config, dataset_manifest):
    """Each worker owns one ADS directory and reads already prepared shared models."""
    torch.set_num_threads(1)
    if suite == "sensitivity":
        candidate_sensitivity.run(
            root, systems=(sut,), config=config, dataset_manifest=dataset_manifest
        )
    else:
        fixed_comparison(
            root,
            systems=(sut,),
            suite=suite,
            config=config,
            dataset_manifest=dataset_manifest,
            methods=(*MAIN_METHODS, *CONTROLS) if suite == "fixed" else ("joint", *MECHANISMS),
            reuse_root=OUTPUT / ("pilot" if suite == "fixed" else "fixed"),
            classifier_builder=partial(historical_classifiers, train_missing=False),
            historical_collision_loader=mechanisms.historical_collision_models
            if suite == "mechanisms"
            else None,
        )


def collect_rows(root, protocol):
    """Rebase complete records from atomic worker summaries without moving raw archives."""
    methods = protocol["main_methods"] + protocol["controls"]
    kinds = protocol["config"]["scenarios"]
    seeds = protocol["config"]["algorithm_seeds"]
    systems = protocol["systems"]
    sizes = protocol.get("candidate_sizes")
    expected = {
        (sut["sut_id"], kind, seed, method, size)
        for sut in systems
        for kind in kinds
        for seed in seeds
        for method in methods
        for size in (sizes or [protocol["candidate_counts"][kind]])
    }
    common = (
        "config",
        "candidate_counts",
        "rss_contract",
        "training_contract",
        "runner_contract",
        "dataset_sha256",
        "cpu_torch_threads",
        "main_methods",
        "controls",
    )
    if protocol["suite"] != "sensitivity":
        common += ("baseline_contract", "method_configurations", "candidate_replicate")
    else:
        common += ("candidate_sizes", "candidate_replicate", "initial_pool_size")
    rows, seen, complete = [], set(), True
    for sut in systems:
        batch = root / "tasks" / sut["sut_id"]
        path = batch / "reports/summary.json"
        if not path.exists():
            complete = False
            continue
        summary = read_json(path)
        registration = read_json(batch / "reports/protocol.json")
        if registration["systems"] != [sut] or any(registration[k] != protocol[k] for k in common):
            raise ValueError(f"ADS batch changed the registered comparison: {sut['sut_id']}")
        if sizes:
            for size in sizes:
                nested = batch / str(size) / "reports/protocol.json"
                if nested.exists():
                    child = read_json(nested)
                    if (
                        child["systems"] != [sut]
                        or child["config"] != {**protocol["config"], "candidate_count": size}
                        or child["method_configurations"] != protocol["method_configurations"]
                        or child["baseline_contract"] != protocol["baseline_contract"]
                        or child["initial_pool_size"] != protocol["initial_pool_size"]
                    ):
                        raise ValueError(f"Nested comparison changed its contract: {nested}")
        held = set()
        for row in summary["runs"]:
            key = (
                row["sut_id"],
                row["scenario"],
                row["seed"],
                row["method"],
                row["metrics"]["candidate_count"],
            )
            if row["sut_id"] != sut["sut_id"] or key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate ADS task: {key}")
            held.add(key)
            seen.add(key)
            result_path = (batch / row["result_path"]).resolve()
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            rebased = {**row, "result_path": os.path.relpath(result_path, root).replace("\\", "/")}
            if row.get("reused_from") is not None:
                rebased["reused_from"] = os.path.relpath(batch / row["reused_from"], root).replace(
                    "\\", "/"
                )
            rows.append(rebased)
        expected_ads = {key for key in expected if key[0] == sut["sut_id"]}
        if summary["status"] == "complete" and held != expected_ads:
            raise ValueError(f"Completed ADS batch is missing registered tasks: {sut['sut_id']}")
        complete &= summary["status"] == "complete"
    system_order = {sut["sut_id"]: i for i, sut in enumerate(systems)}
    rows.sort(
        key=lambda row: (
            sizes.index(row["candidate_count"]) if sizes else 0,
            system_order[row["sut_id"]],
            kinds.index(row["scenario"]),
            seeds.index(row["seed"]),
            methods.index(row["method"]),
        )
    )
    return rows, complete and seen == expected


def run(
    suite="fixed",
    root=None,
    *,
    config=None,
    systems=CURRENT_INDEPENDENT_SUTS,
    workers=4,
    dataset_manifest=DATASET_ROOT / "manifest.json",
):
    """Only fixed-history suites may run ADS tasks concurrently; sequential remains ordered."""
    if suite not in ("fixed", "mechanisms", "sensitivity"):
        raise ValueError("Parallel ADS execution requires a fixed-history comparison")
    if workers < 1 or not systems or len({sut.sut_id for sut in systems}) != len(systems):
        raise ValueError("Workers and distinct ADS tasks must be nonempty")
    torch.set_num_threads(1)
    config = config or (
        replace(DEFAULT, scenarios=("front_brake",)) if suite == "sensitivity" else DEFAULT
    )
    root = Path(root) if root is not None else OUTPUT / suite
    workers = min(workers, len(systems)) if torch.device(config.device).type == "cpu" else 1
    methods = (
        (*MAIN_METHODS, *CONTROLS)
        if suite == "fixed"
        else (("joint", *MECHANISMS) if suite == "mechanisms" else MAIN_METHODS)
    )
    extra = {
        "main_methods": [m for m in methods if m in MAIN_METHODS],
        "controls": [m for m in methods if m not in MAIN_METHODS],
        "method_configurations": method_configurations(),
        "baseline_contract": implementation_contract(),
        "candidate_replicate": 30 if suite == "sensitivity" else 10,
        "source_availability": "fixed original historical ADSs; completed tasks remain sealed",
        "execution_layout": {
            "unit": "independent ADS",
            "workers": workers,
            "within_task_queries": "serial",
            "selector_torch_threads": 1,
            "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    if suite == "sensitivity":
        if config.scenarios != ("front_brake",):
            raise ValueError("Candidate sensitivity requires front braking")
        extra.update(
            candidate_sizes=list(candidate_sensitivity.SIZES),
            initial_pool_size=min(candidate_sensitivity.SIZES),
            physical_bank="largest nested pool only",
            historical_classifier_root=os.path.relpath(OUTPUT / "pilot/models", root).replace(
                "\\", "/"
            ),
        )
        registration_config = replace(config, candidate_count=max(candidate_sensitivity.SIZES))
    else:
        count = len(systems) * len(config.scenarios) * len(config.algorithm_seeds)
        extra.update(
            main_runs=count * len(extra["main_methods"]),
            control_runs=count * len(extra["controls"]),
            reuse_root=os.path.relpath(
                OUTPUT / ("pilot" if suite == "fixed" else "fixed"), root
            ).replace("\\", "/"),
        )
        registration_config = config
    protocol = freeze_protocol(
        root, suite, registration_config, systems, dataset_manifest, extra=extra
    )
    if suite == "sensitivity":
        for size in candidate_sensitivity.SIZES:
            view_extra = {key: value for key, value in extra.items() if key != "candidate_sizes"}
            view_extra["historical_classifier_root"] = os.path.relpath(
                OUTPUT / "pilot/models", root / str(size)
            ).replace("\\", "/")
            freeze_protocol(
                root / str(size),
                suite,
                replace(config, candidate_count=size),
                systems,
                dataset_manifest,
                extra=view_extra,
            )
    previous = None

    def record(finished=False):
        nonlocal previous
        rows, complete = collect_rows(root, protocol)
        status = "complete" if finished and complete else "running"
        if (len(rows), status) != previous:
            summary = {"status": status, "runs": rows}
            if suite == "sensitivity":
                summary.update(
                    physical_truth_records=len(systems) * max(candidate_sensitivity.SIZES),
                    logical_queries=sum(row["metrics"]["queries"] for row in rows),
                )
                for size in candidate_sensitivity.SIZES:
                    view = root / str(size)
                    members = []
                    for row in rows:
                        if row["candidate_count"] != size:
                            continue
                        member = {
                            **row,
                            "result_path": os.path.relpath(root / row["result_path"], view).replace(
                                "\\", "/"
                            ),
                        }
                        if row.get("reused_from") is not None:
                            member["reused_from"] = os.path.relpath(
                                root / row["reused_from"], view
                            ).replace("\\", "/")
                        members.append(member)
                    write_json(view / "reports/summary.json", {"status": status, "runs": members})
            write_json(root / "reports/summary.json", summary)
            previous = len(rows), status
            print(f"{suite}: {len(rows)} saved runs, {status}, {workers} ADS workers", flush=True)
        if finished and not complete:
            raise ValueError("ADS workers finished without the complete registered comparison")
        return rows

    if suite == "mechanisms":
        # Prepare shared checkpoints once before any worker can request the same files.
        for kind in config.scenarios:
            mechanisms.historical_collision_models(kind)
    record()
    if workers == 1:
        for sut in systems:
            run_system(root / "tasks" / sut.sut_id, sut, suite, config, dataset_manifest)
            record()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(
                    run_system, root / "tasks" / sut.sut_id, sut, suite, config, dataset_manifest
                ): sut.sut_id
                for sut in systems
            }
            while pending:
                finished, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                for future in finished:
                    sut_id = pending.pop(future)
                    try:
                        future.result()
                    except Exception as error:
                        print(f"{suite}: ADS worker {sut_id} failed: {error!r}", flush=True)
                        raise
                record()
    return record(finished=True)


if __name__ == "__main__":
    run()
