"""Freeze, build, and evaluate the PR-BRVT version-regression MVP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mvr.highway.config import VersionRegressionConfig
from mvr.highway.data.generate_anchor_bank import generate_dual_mode_anchor_bank
from mvr.highway.data.response_bank import build_response_bank
from mvr.highway.data.version_bank import (
    load_versioned_bank,
    make_versioned_bank,
    save_version_metadata,
)
from mvr.highway.diva.version_reference import critical_events
from mvr.highway.experiments.run_version_regression import (
    run_version_regression,
    summarize_version_regression,
    write_replay_artifacts,
)
from mvr.highway.sut.version_lineage import load_version_lineages


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _config(path: Path) -> VersionRegressionConfig:
    config = VersionRegressionConfig(**json.loads(path.read_text(encoding="utf-8")))
    config.validate()
    return config


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "calibration": run_dir / "calibration",
        "evaluation": run_dir / "evaluation",
        "replay": run_dir / "replay",
        "protocol": run_dir / "protocol.frozen.yaml",
        "manifest": run_dir / "manifest.frozen.json",
        "provenance": run_dir / "provenance.json",
    }


def freeze(config_path: Path, manifest_path: Path, run_dir: Path) -> None:
    """Persist exact inputs and hashes without invoking Highway-env."""
    config = _config(config_path)
    lineages = load_version_lineages(manifest_path)
    if len(lineages.versions) != 12 or len(lineages.lineage_ids) != 2:
        raise ValueError("the frozen MVP requires two six-version lineages")
    paths = _paths(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "config_hash": _sha256(config_path),
        "manifest_hash": _sha256(manifest_path),
        "code_commit": _commit(),
    }
    if paths["provenance"].exists():
        current = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        if current["frozen_inputs"] != inputs:
            raise FileExistsError("run directory belongs to different frozen inputs")
        return
    paths["protocol"].write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    paths["manifest"].write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    provenance = {
        "frozen_inputs": inputs,
        "new_highway_episodes": 0,
        "cache_hits": 0,
        "offline_reveal_calls": 0,
        "model_training": "analytic CPU SVD and posterior only",
        "highway_env_version": version("highway-env"),
    }
    paths["provenance"].write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen version-regression protocol in {run_dir}")


def _assert_outcome_semantics(bank, threshold: float) -> None:
    critical = critical_events(bank.collisions, bank.near_misses)
    if not np.array_equal(bank.vulnerability >= threshold, critical):
        raise ValueError("vulnerability threshold must equal the formal critical event")


def build(run_dir: Path) -> None:
    """Build the single calibration and evaluation bank from frozen inputs."""
    paths = _paths(run_dir)
    if not paths["provenance"].is_file():
        raise FileNotFoundError("run freeze before build")
    config = _config(paths["protocol"])
    lineages = load_version_lineages(paths["manifest"])
    profiles = lineages.profile_tuple()
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    for name, count, seed in (
        ("calibration", config.calibration_anchors, config.calibration_seed),
        ("evaluation", config.num_anchors, config.evaluation_seed),
    ):
        directory = paths[name]
        npz_path = directory / "response_bank.npz"
        metadata_path = directory / "version_metadata.json"
        if npz_path.exists() or metadata_path.exists():
            if not (npz_path.is_file() and metadata_path.is_file()):
                raise FileExistsError(f"incomplete existing {name} bank")
            provenance["cache_hits"] += len(profiles) * count
            continue
        directory.mkdir(parents=True, exist_ok=True)
        anchors, modes = generate_dual_mode_anchor_bank(count, seed)
        response = build_response_bank(anchors, seed, modes, profiles)
        _assert_outcome_semantics(response, config.critical_threshold)
        response.save(npz_path)
        bank = make_versioned_bank(
            response,
            lineages,
            seed,
            {"anchor_seed": seed, "mode_generator": "generate_dual_mode_anchor_bank"},
        )
        save_version_metadata(bank, metadata_path)
        provenance["new_highway_episodes"] += len(profiles) * count
        if name == "calibration":
            with (directory / "version_quality.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("version_id", "failure_rate", "mean_vulnerability"))
                writer.writeheader()
                for index, version_meta in enumerate(lineages.versions):
                    writer.writerow({
                        "version_id": version_meta.version_id,
                        "failure_rate": float((response.collisions[index] | response.near_misses[index]).mean()),
                        "mean_vulnerability": float(response.vulnerability[index].mean()),
                    })
    paths["provenance"].write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Built frozen version banks in {run_dir}")


def evaluate(run_dir: Path) -> None:
    """Replay four methods on the frozen evaluation bank without simulation."""
    paths = _paths(run_dir)
    if (paths["replay"] / "summary.json").exists():
        raise FileExistsError("evaluation output exists; refusing to overwrite")
    config = _config(paths["protocol"])
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    bank = load_versioned_bank(
        paths["evaluation"] / "response_bank.npz",
        paths["evaluation"] / "version_metadata.json",
    )
    bank_hash = _sha256(paths["evaluation"] / "response_bank.npz")
    protocol = {
        "protocol_hash": provenance["frozen_inputs"]["config_hash"],
        "bank_hash": bank_hash,
        "code_commit": provenance["frozen_inputs"]["code_commit"],
    }
    rows, traces = run_version_regression(bank, config, protocol)
    write_replay_artifacts(rows, traces, paths["replay"])
    summary = summarize_version_regression(rows)
    summary["protocol"] = protocol
    (paths["replay"] / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    provenance["offline_reveal_calls"] += len(traces)
    paths["provenance"].write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(
        "# PR-BRVT version-regression MVP\n\n"
        f"Status: `{summary['status']}`.\n\n"
        "This is a frozen controlled-controller replay, not a product safety claim.\n",
        encoding="utf-8",
    )
    print(f"Version-regression status: {summary['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--config", type=Path, required=True)
    freeze_parser.add_argument("--manifest", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.config, args.manifest, args.output)
    elif args.command == "build":
        build(args.run_dir)
    else:
        evaluate(args.run_dir)


if __name__ == "__main__":
    main()
