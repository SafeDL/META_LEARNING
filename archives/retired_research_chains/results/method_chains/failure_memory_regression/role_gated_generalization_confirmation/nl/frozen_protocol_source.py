"""Freeze a fresh, wider-family test of role-gated contextual adaptation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from methods.failure_memory_regression import bidirectional as fbrt
from sut_algorithms.highway_env.registry import build_spec_factory


ROOT = Path(
    "results/method_chains/failure_memory_regression/"
    "role_gated_generalization_confirmation"
)
SPLIT = "role_gated_generalization_confirmation"
FAMILIES = ("S01", "S02", "S03", "S04", "S05", "S06", "S08", "S09")
METHODS = fbrt.METHODS + (
    "coordinate_context_ucb_pure",
    "directed_context_ucb_pure",
    "directed_context_ucb_loose",
    "coordinate_offset_calibrated_ucb",
    "directed_offset_calibrated_ucb",
    "directed_bootstrap_offset_ucb",
    "coordinate_regression_bootstrap_ucb",
    "directed_regression_bootstrap_ucb",
    "static_role_coverage2",
    "coordinate_role_gated",
    "directed_role_gated_edges",
)
PRIMARY = "coordinate_role_gated"
BASELINES = (
    "static_risk",
    "coordinate_residual",
    "coordinate_offset_calibrated_ucb",
    "directed_offset_calibrated_ucb",
    "directed_bootstrap_offset_ucb",
    "coordinate_regression_bootstrap_ucb",
    "directed_regression_bootstrap_ucb",
    "static_role_coverage2",
    "directed_role_gated_edges",
    "target_only",
)
CHAINS = {
    "nl": ("nl_v0", "nl2_v1", "nl2_v2"),
    "ppo": ("ppo_release_v0", "ppo_release_v1", "ppo_release_v2"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_snapshot(source: Path, destination: Path) -> str:
    content = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != content:
        bank = destination.parent / "full_response_bank.jsonl"
        if bank.exists() and bank.stat().st_size:
            raise ValueError(f"cannot change frozen source after measurement starts: {destination}")
    destination.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _context_key(row: dict) -> tuple[str, str] | None:
    fixed = row.get("fixed_context")
    family = row.get("catalogue_id")
    if fixed is None or family is None:
        return None
    return family, json.dumps(fixed, sort_keys=True, separators=(",", ":"))


def _assert_new_contexts(rows: list[dict], root: Path) -> None:
    base = Path("results/method_chains/failure_memory_regression")
    previous = set()
    for manifest in base.rglob("scenario_manifest.jsonl"):
        if SPLIT in manifest.parts:
            continue
        try:
            manifest.resolve().relative_to(root.resolve())
        except ValueError:
            pass
        else:
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                key = _context_key(json.loads(line))
                if key is not None:
                    previous.add(key)
    candidate = {key for row in rows if (key := _context_key(row)) is not None}
    overlaps = previous & candidate
    if overlaps:
        raise ValueError(f"physical contexts overlap previous manifests: {sorted(overlaps)}")


def freeze_chain(name: str, builds: tuple[str, ...]) -> dict:
    root = ROOT / name
    candidate = fbrt.compile_manifest(SPLIT, resolution=11, families=FAMILIES)
    _assert_new_contexts(candidate, root)
    rows = fbrt.freeze(
        root,
        split=SPLIT,
        resolution=11,
        builds=builds,
        methods=METHODS,
        families=FAMILIES,
        primary_method=PRIMARY,
        primary_baselines=BASELINES,
    )
    project = Path.cwd()
    sources = {
        "selector": Path(fbrt.__file__).resolve(),
        "physical_runner": project / "highway_sim_env/envs/fbrt_unified_env.py",
        "ppo_loader": project / "sut_algorithms/highway_env/ppo_ece.py",
        "build_registry": project / "sut_algorithms/highway_env/registry.py",
        "nl_release_profiles": project / "sut_algorithms/highway_env/nl_release.py",
        "protocol_freezer": Path(__file__).resolve(),
    }
    snapshot_names = {
        "selector": "frozen_selector_source.py",
        "physical_runner": "frozen_physical_runner_source.py",
        "ppo_loader": "frozen_ppo_loader_source.py",
        "build_registry": "frozen_build_registry_source.py",
        "nl_release_profiles": "frozen_nl_release_source.py",
        "protocol_freezer": "frozen_protocol_source.py",
    }
    source_hashes = {key: _freeze_snapshot(source, root / snapshot_names[key])
                     for key, source in sources.items()}
    ppo_hashes = {build: build_spec_factory(build).checkpoint_sha256
                  for build in builds if build.startswith("ppo_release_")}
    config = {
        "status": "frozen before target outcome measurement",
        "split": SPLIT,
        "families": list(FAMILIES),
        "contexts_per_family": 3,
        "grid_resolution": 11,
        "scenes": len(rows),
        "physical_episodes_by_build": {build: len(rows) for build in builds},
        "total_budget": 40,
        "direction_budget": 20,
        "random_repeats": 10,
        "methods": list(METHODS),
        "primary_method": PRIMARY,
        "primary_baselines": list(BASELINES),
        "candidate_selected_from": [
            "regression_bootstrap_confirmation",
            "superiority_replication_confirmation",
        ],
        "selection": (
            "coordinate role-gated adaptive posterior: regression direction probes "
            "one minimum-parent-TTC case per context; improvement direction probes "
            "up to two parent-failure/lead-partner roles in eligible contexts; "
            "then coordinate posterior UCB and discovered-neighbor frontier ranking"
        ),
        "ppo_checkpoint_sha256": ppo_hashes or None,
        "source_sha256": source_hashes,
        "protocol_sha256": _sha256(root / "protocol.json"),
        "manifest_sha256": _sha256(root / "scenario_manifest.jsonl"),
    }
    config_path = root / "selector_config.json"
    content = (json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if config_path.exists() and config_path.read_bytes() != content:
        bank = root / "full_response_bank.jsonl"
        if bank.exists() and bank.stat().st_size:
            raise ValueError(f"cannot change frozen selector config after measurement starts: {config_path}")
    config_path.write_bytes(content)
    return {"chain": name, "scenes": len(rows), "source_sha256": source_hashes,
            "protocol_sha256": config["protocol_sha256"],
            "manifest_sha256": config["manifest_sha256"]}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    chains = [freeze_chain(name, builds) for name, builds in CHAINS.items()]
    audit = {"status": "frozen before target outcome measurement", "chains": chains}
    (ROOT / "freeze_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(chains, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
