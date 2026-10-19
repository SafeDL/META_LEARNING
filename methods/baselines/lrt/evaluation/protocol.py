"""Verify frozen decision code and weights; retain the original full snapshot."""

from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from methods.baselines.dart.evaluation.freeze import verify_freeze
from methods.baselines.hgt.io import read_json, write_json
from methods.common.snapshots import current_path, current_source, verify_source

from ..config import (
    ARCHIVES,
    MODELS,
    REFINE_STEPS,
    ROOT,
    SEEDS,
    TRAIN_MODES,
    TRAIN_STEPS,
    VARIANTS,
)

PROJECT = Path(__file__).resolve().parents[4]


def initialize():
    baseline = verify_freeze()
    path = ROOT / "protocol.json"
    if path.exists():
        return read_json(path)
    protocol = {
        "created_at": datetime.now().isoformat(),
        "method": "local transfer and CG-BRT continuation learning",
        "metrics": ["H200", "high_risk_recall_pct"],
        "retention_floor": 0.9,
        "budget": 200,
        "initial_queries": 10,
        "loss_budget": 20,
        "model_set": MODELS,
        "seeds": SEEDS,
        "training_variants": TRAIN_MODES,
        "variants": VARIANTS,
        "training_archives": ARCHIVES,
        "new_physical_calls": 0,
        "existing_pools": [0, 1, 2],
        "all_target_pools_are_previously_observed": True,
        "target_results_select_weights": False,
        "baseline_frozen_sources": len(baseline["decision_source_files"]),
        "stress_interpretation": "Deployment archive visibility changes; pretrained representations remain frozen",
        "guarantee_scope": "Conditional error bounds; empirical validation does not certify target conditional coverage",
    }
    write_json(path, protocol)
    return protocol


def freeze():
    initialize()
    destination = ROOT / "frozen_method.zip"
    package = PROJECT / "methods/baselines/lrt"
    sources = sorted(
        path
        for path in package.glob("*.py")
        if path.name not in ("build.py", "data.py", "train.py")
    )
    sources.extend(
        PROJECT / "methods/common" / name
        for name in ("prior.py", "student.py", "workspace.py")
    )
    weights = [
        ROOT / "models" / f"{mode}_{seed}.pt" for mode in TRAIN_MODES for seed in SEEDS
    ]
    if destination.exists():
        # Training and evaluation may be reorganized. The complete original
        # implementation remains in this archive; deployed decisions stay frozen.
        with ZipFile(destination) as archive:
            entries = {current_path(name): name for name in archive.namelist()}
            for path in sources + weights:
                relative = path.relative_to(PROJECT).as_posix()
                name = entries.get(relative)
                original = current_source(archive.read(name), name) if name else None
                verify_source(PROJECT, relative, original)
        return
    for mode in TRAIN_MODES:
        for seed in SEEDS:
            report = read_json(ROOT / "models" / f"training_{mode}_{seed}.json")
            if (
                report["status"] != "complete"
                or report["executed_steps"] != TRAIN_STEPS + REFINE_STEPS
            ):
                raise ValueError(
                    "Target evaluation requires all historical training updates"
                )
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*.py")) + weights:
            archive.write(path, path.relative_to(PROJECT).as_posix())
        archive.write(ROOT / "protocol.json", "protocol.json")
