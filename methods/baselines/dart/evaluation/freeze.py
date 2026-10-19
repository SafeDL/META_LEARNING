"""Freeze source and selected weights before any independent target measurement."""

from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from benchmark.config import sut_spec
from methods.baselines.dart import config as method_config
from methods.baselines.hgt.io import read_json, write_json
from methods.common.snapshots import current_path, current_source, verify_source

from .config import (
    ACCEPTANCE_METRICS,
    CONFIRMATION_VARIANTS,
    INDEPENDENT_SEEDS,
    ROOT,
    SEEDS,
    SUT_IDS,
)

PROJECT = Path(__file__).resolve().parents[4]


def verify_freeze():
    protocol = read_json(ROOT / "protocol.json")
    extensions = []
    with ZipFile(ROOT / "frozen_inputs.zip") as archive:
        for relative in protocol["decision_source_files"]:
            expected = current_source(archive.read(f"source/{relative}"), relative)
            if verify_source(PROJECT, relative, expected):
                extensions.append(current_path(relative))
        for name in protocol["model_files"]:
            if (ROOT / "models" / name).read_bytes() != archive.read(f"models/{name}"):
                raise ValueError(f"Frozen model changed before confirmation: {name}")
        for relative in protocol["input_files"]:
            if (PROJECT / current_path(relative)).read_bytes() != archive.read(
                f"inputs/{relative}"
            ):
                raise ValueError(
                    f"Frozen historical input or kernel changed: {relative}"
                )
    return {**protocol, "reviewed_api_extensions": extensions}


def main():
    if (ROOT / "protocol.json").exists():
        return verify_freeze()
    source_files = []
    for directory in (
        "methods/baselines/dart",
        "methods/baselines/cg_brt",
        "simulation/highway/envs",
        "controllers/highway",
    ):
        source_files.extend(
            path.relative_to(PROJECT).as_posix()
            for path in (PROJECT / directory).rglob("*.py")
        )
    source_files.extend(
        [
            "methods/baselines/hgt/gp.py",
            "methods/baselines/hgt/kernel.py",
            "methods/baselines/hgt/train.py",
            "methods/baselines/hgt/history.py",
            "methods/baselines/hem/experts.py",
            "methods/baselines/hgt/config.py",
            "methods/baselines/hgt/measurements.py",
            "methods/baselines/hgt/risk.py",
            "benchmark/config.py",
            "benchmark/prepare.py",
            "benchmark/risk_channel.py",
            "simulation/highway/build_spec.py",
            "methods/baselines/dart/requirements.txt",
        ]
    )
    model_files = [
        f"{mode}_{seed}.pt" for seed in SEEDS for mode in ("decision", "conventional")
    ]
    input_files = ["data/history/responses.npz"] + [
        f"results/baselines/hgt/models/seed_{seed}.pt" for seed in SEEDS
    ]
    for seed in SEEDS:
        for mode in ("decision", "conventional"):
            if (
                read_json(ROOT / "models" / f"training_{mode}_{seed}.json")["status"]
                != "complete"
            ):
                raise ValueError(
                    "Confirmation requires all declared training updates to be completed"
                )
    development = read_json(ROOT / "development_comparison.json")
    settings = {
        name: getattr(method_config, name)
        for name in (
            "BUDGET",
            "INITIAL_QUERIES",
            "FANTASIES",
            "LOSS_FRACTION",
            "CANDIDATES_PER_RULE",
            "WIDTH",
            "HEADS",
            "FEATURES",
            "CONTEXT_PER_FAMILY",
            "MAX_AMPLITUDE",
            "INITIAL_AMPLITUDE",
            "SCALE_NODES",
            "TRAINING_STEPS",
            "VALIDATION_INTERVAL",
            "LEARNING_RATE",
            "SUPPORT_COUNTS",
        )
    }
    protocol = {
        "method": "decision-aware risk transfer",
        "settings": settings,
        "source_files": source_files,
        "decision_source_files": [
            relative
            for relative in source_files
            if "/evaluation/" not in relative
            and not relative.startswith("benchmark/")
            and relative != "methods/baselines/dart/train.py"
        ],
        "source_verification_scope": "Frozen decision implementation, selected weights and inputs; evaluation scripts can be reorganized",
        "model_files": model_files,
        "input_files": input_files,
        "independent_pool_seeds": list(INDEPENDENT_SEEDS),
        "suts": list(SUT_IDS),
        "target_builds": [asdict(sut_spec(s)) for s in SUT_IDS],
        "confirmation_variants": list(CONFIRMATION_VARIANTS),
        "selection_seeds": list(SEEDS),
        "physical_calls_planned": 12288,
        "online_feedback": "queried R only",
        "source_only_checkpoint_selection": True,
        "new_pool_used_for_selection": False,
        "evaluation_metrics": list(ACCEPTANCE_METRICS),
        "development_retention": development["retention"],
        "development_effect_accepted": development["effect_accepted"],
        "guarantee_scope": "model-conditional same-history replacement; not full policy domination",
        "confirmation_scope": "new scenes in the same six systems, not new-system generalization",
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "scipy",
                "highway-env",
                "gymnasium",
                "stable-baselines3",
                "cupy-cuda11x",
            )
        },
    }
    write_json(ROOT / "protocol.json", json.loads(json.dumps(protocol)))
    with ZipFile(ROOT / "frozen_inputs.zip", "w", ZIP_DEFLATED) as archive:
        archive.write(ROOT / "protocol.json", "protocol.json")
        for relative in source_files:
            archive.write(PROJECT / relative, f"source/{relative}")
        for name in model_files:
            archive.write(ROOT / "models" / name, f"models/{name}")
        for relative in input_files:
            archive.write(PROJECT / relative, f"inputs/{relative}")
    return verify_freeze()


if __name__ == "__main__":
    main()
