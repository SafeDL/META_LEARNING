"""Freeze the stage-two, no-coverage posterior UCB confirmation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from methods.failure_memory_regression import bidirectional as fbrt
from sut_algorithms.highway_env.registry import build_spec_factory


ROOT = Path("results/method_chains/failure_memory_regression/contextual_confirmation")
SPLIT = "contextual_confirmation"
FAMILIES = ("S01", "S02", "S08")
METHODS = fbrt.METHODS + (
    "coordinate_context_ucb_pure",
    "directed_context_ucb_pure",
    "directed_context_ucb_loose",
)
PRIMARY = "directed_context_ucb_loose"
BASELINES = (
    "static_risk",
    "coordinate_residual",
    "coordinate_context_ucb_pure",
    "directed_context_ucb_pure",
    "target_only",
)
CHAINS = {
    "nl": ("nl_v0", "nl2_v1", "nl2_v2"),
    "ppo": ("ppo_release_v0", "ppo_release_v1", "ppo_release_v2"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_file(source: Path, destination: Path) -> str:
    source_bytes = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source_bytes:
            bank = destination.parent / "full_response_bank.jsonl"
            if bank.exists() and bank.stat().st_size:
                raise ValueError(f"cannot change frozen source after measurement begins: {destination}")
            destination.write_bytes(source_bytes)
    else:
        destination.write_bytes(source_bytes)
    return hashlib.sha256(source_bytes).hexdigest()


def _assert_new_contexts(rows: list[dict], output_root: Path) -> None:
    base = Path("results/method_chains/failure_memory_regression")

    def key(row: dict) -> tuple[str, str] | None:
        fixed = row.get("fixed_context")
        family = row.get("catalogue_id")
        if fixed is None or family is None:
            return None
        return family, json.dumps(fixed, sort_keys=True, separators=(",", ":"))

    previous = set()
    for manifest in base.rglob("scenario_manifest.jsonl"):
        if SPLIT in manifest.parts:
            continue
        if manifest.resolve().is_relative_to(output_root.resolve()):
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = key(json.loads(line))
                if item is not None:
                    previous.add(item)
    overlaps = previous & {item for row in rows if (item := key(row)) is not None}
    if overlaps:
        raise ValueError(f"new physical contexts overlap prior manifests: {sorted(overlaps)}")


def freeze_chain(name: str, builds: tuple[str, ...]) -> dict:
    root = ROOT / name
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
    _assert_new_contexts(rows, root)
    project = Path.cwd()
    source_paths = {
        "selector": Path(fbrt.__file__).resolve(),
        "physical_runner": project / "highway_sim_env/envs/fbrt_unified_env.py",
        "ppo_loader": project / "sut_algorithms/highway_env/ppo_ece.py",
        "build_registry": project / "sut_algorithms/highway_env/registry.py",
        "nl_release_profiles": project / "sut_algorithms/highway_env/nl_release.py",
        "protocol_freezer": Path(__file__).resolve(),
    }
    source_hashes = {}
    snapshot_names = {
        "selector": "frozen_selector_source.py",
        "physical_runner": "frozen_physical_runner_source.py",
        "ppo_loader": "frozen_ppo_loader_source.py",
        "build_registry": "frozen_build_registry_source.py",
        "nl_release_profiles": "frozen_nl_release_source.py",
        "protocol_freezer": "frozen_protocol_source.py",
    }
    for key, source in source_paths.items():
        source_hashes[key] = _freeze_file(source, root / snapshot_names[key])

    ppo_hashes = {
        build: build_spec_factory(build).checkpoint_sha256
        for build in builds if build.startswith("ppo_release_")
    }
    selector_config = {
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
        "selection": (
            "one context-local target-risk posterior with fixed parent-risk offset; "
            "regression uses deterministic +1 posterior standard deviation, "
            "improvement uses the posterior-mean complement; no forced context "
            "probe or neighborhood-frontier override"
        ),
        "method_parameters": {
            "coordinate_context_ucb_pure": {
                "context_intercept_prior_variance": 1.0,
                "regression_ucb_standard_deviations": 1.0,
                "improvement_score": "posterior mean complement",
            },
            "directed_context_ucb_pure": {
                "context_intercept_prior_variance": 1.0,
                "regression_ucb_standard_deviations": 1.0,
                "improvement_score": "posterior mean complement",
            },
            "directed_context_ucb_loose": {
                "context_intercept_prior_variance": 4.0,
                "regression_ucb_standard_deviations": 1.0,
                "improvement_score": "posterior mean complement",
            },
        },
        "ppo_checkpoint_sha256": ppo_hashes or None,
        "source_sha256": source_hashes,
        "protocol_sha256": _sha256(root / "protocol.json"),
        "manifest_sha256": _sha256(root / "scenario_manifest.jsonl"),
    }
    config_path = root / "selector_config.json"
    config_bytes = (json.dumps(selector_config, ensure_ascii=False, sort_keys=True,
                               indent=2) + "\n").encode("utf-8")
    if config_path.exists():
        if config_path.read_bytes() != config_bytes:
            bank = root / "full_response_bank.jsonl"
            if bank.exists() and bank.stat().st_size:
                raise ValueError(f"cannot change frozen selector config after measurement begins: {config_path}")
            config_path.write_bytes(config_bytes)
    else:
        config_path.write_bytes(config_bytes)
    return {"chain": name, "scenes": len(rows),
            "source_sha256": source_hashes,
            "protocol_sha256": selector_config["protocol_sha256"]}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    results = [freeze_chain(name, builds) for name, builds in CHAINS.items()]
    (ROOT / "freeze_audit.json").write_text(
        json.dumps({"status": "frozen before target outcome measurement",
                    "chains": results}, ensure_ascii=False, sort_keys=True,
                   indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
