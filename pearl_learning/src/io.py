"""Configuration and deterministic JSON helpers."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping
import yaml


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_config(path: str | Path) -> dict[str, Any]:
    """Read the single project YAML configuration."""
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("configuration root must be a mapping")
    local = dict(value)
    parent = local.pop("extends", None)
    if parent is None:
        return local
    base = read_config(source.parent / str(parent))

    def merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(left)
        for key, item in right.items():
            if isinstance(result.get(key), Mapping) and isinstance(item, Mapping):
                result[key] = merge(result[key], item)
            else:
                result[key] = item
        return result

    return merge(base, local)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    """Hash exact source bytes, distinct from canonical resolved-config hash."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def assert_method_variant_contract(config: Mapping[str, Any], run_name: str, run_kind: str) -> str:
    """Reject mislabeled Vanilla/Structure runs before any environment starts."""
    experiment = dict(config.get("experiment", {}))
    marker = " ".join((
        str(experiment.get("method_variant", "")), str(run_name), str(run_kind),
        str(config.get("project", {}).get("output_root", "")),
    )).lower()
    scenario_enabled = bool(config.get("scenario_representation", {}).get("enabled", False))
    prior_mode = str(config.get("scenario_prior", {}).get("mode", "unit_normal"))
    if "vanilla" in marker:
        if scenario_enabled or prior_mode != "unit_normal":
            raise ValueError("vanilla run requires scenario_representation.enabled=false and scenario_prior.mode=unit_normal")
        return "vanilla"
    if "structure" in marker:
        if not scenario_enabled or prior_mode != "task_conditioned":
            raise ValueError("structure-aware run requires scenario_representation.enabled=true and scenario_prior.mode=task_conditioned")
        return "structure_aware"
    if "mechanism" in marker:
        if scenario_enabled or prior_mode != "unit_normal":
            raise ValueError("mechanism run requires vanilla unit-normal latent settings")
        return "mechanism"
    return str(experiment.get("method_variant", "unspecified"))


def prepare_run_manifest(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    """Atomically establish output provenance and reject unsafe directory reuse."""
    root = Path(output_dir)
    manifest_path = root / "run_manifest.json"
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise FileExistsError(f"run output already exists: {root}; use an explicit compatible resume")
        if not manifest_path.exists():
            raise ValueError("cannot resume an output directory without run_manifest.json")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = ("resolved_config_sha256", "taskbook_hash", "casebook_hashes", "critical_threshold_hash")
        mismatched = [key for key in required if previous.get(key) != manifest.get(key)]
        if mismatched:
            raise ValueError(f"resume provenance mismatch for {mismatched}")
        return
    root.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, dict(manifest))
