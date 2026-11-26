"""Native historical transfer on directly sampled, development-validated domains."""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from controllers.highway.catalog import FORMAL_HISTORY_SUTS, FORMAL_SUTS, FORMAL_TARGET_SUTS
from controllers.highway.contracts import runtime_contract
from controllers.highway.factory import create
from simulator.highway.timing import timing_contract
from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import DEFAULT as TRAINING_CONFIG
from testing_methods.history_transfer.training import load_model, train

from .executions import ExecutionBank, collect_bank, execution_contract, require_qualified
from experiments.scene_design.check_formal import verify_development
from .formal_scenarios import (
    HISTORY_COUNTS,
    TARGET_COUNT,
    domain_contract,
    prepare_candidates,
    verify_candidates,
)
from .history import (
    calculation_view,
    load_training_archives,
    model_version,
    uniform_record_ids,
)
from .k_shot import CHECKPOINTS, FEEDBACK_COUNT
from .method_contracts import method_contract
from .parallel import (
    COLLECTION_WORKERS,
    initialize_worker,
    prediction_device,
    prepare_processes,
    training_workers,
)
from .historical_responses import response_from_bank, save_verified
from .protocol import ROOT
from .qualification import COURSES, QUALIFICATION_ROOT, qualify, replay_courses
from .scenarios import DOMAINS, FORMAL_SCENARIOS
from .storage import load_archive, read_json, write_json

OUTPUT = ROOT / "results/k_shot_adaptation"
RUN_SCENARIOS = FORMAL_SCENARIOS
SEEDS = (11,)
PREDICTION_METHODS = ("target_only", "pooled_history", "full")
ACTIVE_EXPERIMENTS = ("prediction",)
BUDGET_METHODS = ("full", "target_only", "random", "av_bo_rf_ei", "samota", "scenario_fuzz")
HISTORY_VIEW_LIMIT = 512
HISTORY_VIEW_SEED = 1301


def freeze(path, value):
    value = json.loads(json.dumps(value))
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Registered protocol changed: {path}")
        return value
    write_json(path, value)
    return value


def scenarios_for(root):
    path = Path(root) / "reports/protocol.json"
    return tuple(read_json(path)["scenarios"]) if path.exists() else RUN_SCENARIOS


def register(root=OUTPUT):
    root = Path(root)
    register_history_view_protocol(root)
    return freeze(
        root / "reports/protocol.json",
        {
            "format": "native-history-transfer-direct",
            "timing": timing_contract(),
            "history": [asdict(s) for s in FORMAL_HISTORY_SUTS],
            "targets": [asdict(s) for s in FORMAL_TARGET_SUTS],
            "scenarios": list(RUN_SCENARIOS),
            "domains": {k: DOMAINS[k].to_dict() for k in RUN_SCENARIOS},
            "parameter_domain": domain_contract(),
            "history_counts_per_ads": {k: HISTORY_COUNTS[k] for k in RUN_SCENARIOS},
            "target_count_per_task": TARGET_COUNT,
            "base_physical_executions": 8
            * sum(HISTORY_COUNTS[k] + TARGET_COUNT for k in RUN_SCENARIOS),
            "domain_validation_training_qualification_render_executions": "separately metered",
            "feedback_seeds": SEEDS,
            "k_shot": {
                "checkpoints": CHECKPOINTS,
                "query_count": FEEDBACK_COUNT,
                "evaluation_count": TARGET_COUNT - FEEDBACK_COUNT,
                "methods": PREDICTION_METHODS,
            },
            "active_experiments": list(ACTIVE_EXPERIMENTS),
            "historical_posterior_records_per_source": HISTORY_VIEW_LIMIT,
            "deferred_experiments": ["budget", "controls", "accumulation"],
            "target_selection": "existing previously observed target configurations; frozen before new K-shot",
            "reuse_old_execution_banks": False,
        },
    )


def readiness(root=OUTPUT):
    from .transfer_admission import readiness as method_readiness

    register(root)
    rows = []
    for sut in FORMAL_SUTS:
        try:
            evidence = require_qualified(sut)
            rows.append({"sut_id": sut.sut_id, "ready": True, "evidence": evidence})
        except (FileNotFoundError, ValueError) as error:
            rows.append({"sut_id": sut.sut_id, "ready": False, "reason": str(error)})
    reasons = []
    domain_ready = False
    try:
        verify_development()
        domain_ready = True
    except (FileNotFoundError, ValueError) as error:
        reasons.append(str(error))
    try:
        registered_populations(root)
        candidates_ready = True
    except (FileNotFoundError, ValueError) as error:
        candidates_ready = False
        reasons.append(str(error))
    ads_ready = all(r["ready"] for r in rows)
    if not ads_ready:
        reasons.append("Some ADS capability identities do not match")
    report = {
        "scope": "Separate ADS, physical collection, and method evaluation readiness",
        "ready": False,
        "ads_ready": ads_ready,
        "systems": rows,
        "collection": {
            "ready": ads_ready and domain_ready and candidates_ready,
            "domain_ready": domain_ready,
            "candidates_ready": candidates_ready,
            "reasons": reasons,
        },
        "method_evaluation": method_readiness(root),
    }
    report["ready"] = report["collection"]["ready"] and report["method_evaluation"]["ready"]
    write_json(Path(root) / "reports/readiness.json", report)
    return report


def qualify_systems(root=OUTPUT):
    register(root)
    rows = []
    for sut in FORMAL_SUTS:
        try:

            def policy_factory(sut=sut):
                return create(sut.sut_id)

            contract = runtime_contract(sut)
            report = {
                "sut_id": sut.sut_id,
                "parameters": sut.parameters,
                "runtime_contract": contract,
                **qualify(policy_factory, save_root=QUALIFICATION_ROOT / sut.sut_id),
            }
            write_json(QUALIFICATION_ROOT / sut.sut_id / "qualification.json", report)
            rows.append({"sut_id": sut.sut_id, "qualified": report["qualified"]})
        except (FileNotFoundError, ValueError) as error:
            rows.append({"sut_id": sut.sut_id, "qualified": False, "reason": str(error)})
        print(rows[-1], flush=True)
    write_json(Path(root) / "reports/capability_checks.json", rows)
    return readiness(root)


def replay_capabilities(root=OUTPUT):
    register(root)
    rows = []
    for sut in FORMAL_SUTS:
        directory = QUALIFICATION_ROOT / sut.sut_id
        saved = read_json(directory / "qualification.json")
        if not saved["qualified"] or saved["runtime_contract"] != runtime_contract(sut):
            raise ValueError(f"Cannot reuse changed ADS capability evidence: {sut.sut_id}")
        evidence = replay_courses(directory)
        if evidence["courses"] != saved["courses"]:
            raise ValueError(f"Capability summary differs: {sut.sut_id}")
        require_qualified(sut)
        rows.append({"sut_id": sut.sut_id, "courses": len(COURSES), "passed": True})
        print(rows[-1], flush=True)
    write_json(
        Path(root) / "reports/capability_replay.json",
        {
            "systems": rows,
            "physical_replay_executions": sum(r["courses"] for r in rows),
            "new_response_queries": 0,
            "new_controller_training_runs": 0,
            "original_qualification_and_trajectories_unchanged": True,
        },
    )
    return readiness(root)


def prepare(root=OUTPUT):
    register(root)
    verify_development()
    for kind in scenarios_for(root):
        for split, count in (("history", HISTORY_COUNTS[kind]), ("target", TARGET_COUNT)):
            prepare_candidates(kind, count, split, Path(root) / "candidates" / split / kind)
    registered_populations(root)
    return readiness(root)


def registered_populations(root):
    populations = {}
    for kind in scenarios_for(root):
        for split in ("history", "target"):
            count = HISTORY_COUNTS[kind] if split == "history" else TARGET_COUNT
            populations[split, kind] = verify_candidates(
                Path(root) / "candidates" / split / kind, kind, count, split
            )
        if {s.coordinate() for s in populations["history", kind]} & {
            s.coordinate() for s in populations["target", kind]
        }:
            raise ValueError("Historical and target coordinate populations overlap")
    return populations


def collect(root=OUTPUT, *, workers=COLLECTION_WORKERS, history_only=False, target_only=False):
    root = Path(root)
    if history_only and target_only:
        raise ValueError("Choose either historical or target-only collection")
    if not readiness(root)["collection"]["ready"]:
        raise ValueError(
            "Formal collection prerequisites are incomplete; see reports/readiness.json"
        )
    freeze(
        root / "reports/execution_registration.json",
        {s.sut_id: runtime_contract(s) for s in FORMAL_SUTS},
    )
    # Check the entire input matrix before launching any ADS executions.
    populations = registered_populations(root)
    entries = []
    populations_to_collect = [] if target_only else [("history", FORMAL_HISTORY_SUTS)]
    if not history_only:
        populations_to_collect.append(("target", FORMAL_TARGET_SUTS))
    for split, systems in populations_to_collect:
        for kind in scenarios_for(root):
            for sut in systems:
                bank = collect_bank(
                    sut.sut_id,
                    populations[split, kind],
                    root / "physical" / split / kind / sut.sut_id,
                    workers=workers,
                    current_registration=True,
                )
                if split == "history":
                    path = root / "data/history" / kind / f"{sut.sut_id}.npz"
                    archive = save_verified(path, response_from_bank(bank.root, batch="native"))
                    entries.append(
                        {
                            "scenario": kind,
                            "source_id": sut.sut_id,
                            "role": "history",
                            "path": path.relative_to(root / "data").as_posix(),
                            "records": len(archive.scenarios),
                            "fingerprint": archive_fingerprints([archive])[sut.sut_id],
                        }
                    )
    if not target_only:
        write_json(root / "data/manifest.json", {"status": "complete", "archives": entries})
    else:
        write_json(
            root / "reports/target_collection.json",
            {
                "status": "complete",
                "target_tasks": len(FORMAL_TARGET_SUTS) * len(scenarios_for(root)),
                "expected_executions": len(FORMAL_TARGET_SUTS)
                * len(scenarios_for(root))
                * TARGET_COUNT,
                "used_for_method_development": False,
            },
        )


def scenario_history(root, kind):
    """Publish all registered sources for one completed scene, without target data."""
    root = Path(root)
    if kind not in scenarios_for(root):
        raise ValueError("Unknown formal historical scenario")
    scenes = registered_populations(root)["history", kind]
    entries = []
    for sut in FORMAL_HISTORY_SUTS:
        path = root / "data/history" / kind / f"{sut.sut_id}.npz"
        archive = load_archive(path)
        archive.continuous_margin
        bank = ExecutionBank(root / "physical/history" / kind / sut.sut_id)
        if (
            len(archive.scenarios) != HISTORY_COUNTS[kind]
            or archive.scenarios != scenes
            or bank.candidates != scenes
            or archive.role != "history"
            or archive.source_id != sut.sut_id
            or archive.family != sut.family
            or archive.lineage != sut.lineage
            or archive.provenance["contract"] != bank.manifest["contract"]
            or bank.manifest["contract"] != execution_contract(sut, kind=kind)
            or bank.manifest["environment_seed"] != 81001
        ):
            raise ValueError("Historical scene inputs differ from the registered completed sources")
        entries.append(
            {
                "scenario": kind,
                "source_id": sut.sut_id,
                "role": "history",
                "path": path.name,
                "records": len(archive.scenarios),
                "fingerprint": archive_fingerprints([archive])[sut.sut_id],
                "physical_bank_snapshot": bank.snapshot_fingerprint,
            }
        )
    manifest = root / "data/history" / kind / "manifest.json"
    freeze(manifest, {"status": "complete", "archives": entries})
    return load_training_archives(manifest, kind)


def model_directory(root, kind):
    return Path(root) / "models/adaptive_mixture" / kind


def train_scenario(kind, root=OUTPUT):
    root = Path(root)
    register(root)
    history = scenario_history(root, kind)
    return train(history, model_directory(root, kind), kernel_type=SharedPrivateKernel)


def train_models(root=OUTPUT):
    root = Path(root)
    register(root)
    # Publish shared input manifests before workers access separate model folders.
    for kind in scenarios_for(root):
        scenario_history(root, kind)
    workers = training_workers(TRAINING_CONFIG.device)
    prepare_processes()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_worker,
    ) as pool:
        pending = {pool.submit(train_scenario, kind, root): kind for kind in scenarios_for(root)}
        for future in as_completed(pending):
            future.result()
            print({"scenario": pending[future], "training": "complete"}, flush=True)


def models_and_history(root, kind):
    history = scenario_history(root, kind)
    models = tuple(
        load_model(model_directory(root, kind) / name, kernel_type=SharedPrivateKernel)[0]
        for name in ("target_kernel.pt", "historical_encoder.pt")
    )
    for model in models:
        specification = model.construction_specification
        if specification["scenario"] != kind or specification[
            "history_fingerprints"
        ] != archive_fingerprints(history):
            raise ValueError("Shared model was trained on a different historical source set")
    freeze(
        root / "reports/models/adaptive_mixture" / f"{kind}.json",
        {
            "model_versions": [model_version(m) for m in models],
            "history_fingerprints": archive_fingerprints(history),
        },
    )
    return models, history


def register_history_view_protocol(root):
    return freeze(
        Path(root) / "reports/history_view_protocol.json",
        {
            "selection": "uniform_without_replacement_from_sorted_record_ids",
            "limit_per_source": HISTORY_VIEW_LIMIT,
            "seed": HISTORY_VIEW_SEED,
            "selection_uses_responses": False,
            "shared_by": ["Ours source posteriors"],
        },
    )


def registered_history_views(root, kind, archives):
    protocol = register_history_view_protocol(root)
    if not archives or len({a.source_id for a in archives}) != len(archives):
        raise ValueError("Historical views require distinct nonempty sources")
    inputs = {a.source_id: list(a.provenance["record_ids"]) for a in archives}
    if any(len(inputs[a.source_id]) != len(a.scenarios) for a in archives):
        raise ValueError("Historical record IDs do not align with their archive")
    mapping = {
        name: uniform_record_ids(ids, limit=HISTORY_VIEW_LIMIT, seed=HISTORY_VIEW_SEED)
        for name, ids in inputs.items()
    }
    specification = freeze(
        Path(root) / "reports/history_views" / f"{kind}.json",
        {
            **protocol,
            "scenario": kind,
            "input_record_ids": inputs,
            "selected_record_ids": mapping,
            "history_fingerprints": archive_fingerprints(archives),
        },
    )
    views = [
        calculation_view(a, limit=HISTORY_VIEW_LIMIT, record_ids=mapping[a.source_id])
        for a in archives
    ]
    return views, specification


def pooled_history(views, specification):
    return {
        "x": np.concatenate([a.x for a in views]),
        "collision": np.concatenate([a.collision for a in views]),
        "rss_margin": np.concatenate([a.continuous_margin for a in views]),
        "record_ids": [r for a in views for r in a.provenance["record_ids"]],
        "source_ids": [a.source_id for a in views],
        "view_specification": specification,
    }


def run_k_shot(root=OUTPUT):
    from .transfer_batches import run

    return run("prediction", root)


def prediction_specification(
    bank,
    split,
    seed,
    method,
    base,
    history,
    sources,
    checkpoints,
    online_settings=None,
    pooled_fit_settings=None,
    update_source_weights=True,
):
    return {
        "sut_id": bank.manifest["sut_id"],
        "scenario": bank.candidates[0].kind,
        "feedback_seed": seed,
        "method": method,
        "split": split,
        "checkpoints": list(checkpoints),
        "model": None if method == "target_only" else model_version(base),
        "history_view": history.get("view_specification") if method == "pooled_history" else None,
        "source_snapshot": sources["specification"] if method == "full" else None,
        "pooled_inference": "grouped" if method == "pooled_history" else None,
        "device": prediction_device(method),
        "bank_contract": bank.manifest["contract"],
        "bank_snapshot": bank.snapshot_fingerprint,
        "candidates": [asdict(s) for s in bank.candidates],
        "method_contract": method_contract(method),
        "online_settings": None if online_settings is None else online_settings.to_dict(),
        "pooled_fit_settings": (
            None if pooled_fit_settings is None else pooled_fit_settings.to_dict()
        ),
        "update_source_weights": update_source_weights,
        "model_construction": (
            None if method == "target_only" else getattr(base, "construction_specification", None)
        ),
    }


def run_budget(root=OUTPUT):
    from .transfer_batches import run

    return run("budget", root)


def require_method_evaluation(root):
    state = readiness(root)
    if not state["collection"]["ready"] or not state["method_evaluation"]["ready"]:
        raise ValueError(
            "Formal method evaluation prerequisites remain incomplete; see reports/readiness.json"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "status",
            "qualify",
            "replay-capabilities",
            "check-scene",
            "prepare",
            "collect",
            "collect-history",
            "collect-targets",
            "models",
            "k-shot",
            "settings",
            "report",
            "figures",
        ),
        default="status",
        nargs="?",
    )
    from .report_history_transfer import report
    from experiments.plotting.k_shot import render
    from .transfer_settings import register_settings

    stage = parser.parse_args().stage
    actions = {
        "status": readiness,
        "qualify": qualify_systems,
        "replay-capabilities": replay_capabilities,
        "check-scene": verify_development,
        "prepare": prepare,
        "collect": collect,
        "collect-history": lambda: collect(history_only=True),
        "collect-targets": lambda: collect(target_only=True),
        "models": train_models,
        "k-shot": run_k_shot,
        "settings": register_settings,
        "report": report,
        "figures": lambda: [str(path) for path in render()],
    }
    result = actions[stage]()
    if stage == "check-scene":
        result = {
            key: result[key] for key in ("status", "development_design_passed", "ads_executions")
        }
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
