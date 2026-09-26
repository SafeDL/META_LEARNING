"""Freeze the proposed age080 interaction candidate bank without simulation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from methods.failure_memory_regression.catalogue import (
    INTERACTION_HOLDOUT_CATALOGUE, compile_interaction_scenarios,
)
from sut_algorithms.highway_env.registry import build_spec_factory


ROOT = Path("results/method_chains/failure_memory_regression/interaction_holdout")
CATALOGUE = INTERACTION_HOLDOUT_CATALOGUE
SEEDS = tuple(range(4179910, 4179918))
BUILDS = ("mobil_ref_v2", "mobil_rear_guard_off_v2", "mobil_rear_state_age080")
CODE_PATHS = (
    "highway_sim_env/envs/fbrt_unified_env.py",
    "highway_sim_env/envs/fbrt_scripted_vehicle.py",
    "sut_algorithms/highway_env/regression_builds.py",
    "sut_algorithms/highway_env/registry.py",
    "sut_algorithms/highway_env/fbrt_adapters.py",
    "methods/failure_memory_regression/catalogue.py",
    "methods/failure_memory_regression/selector.py",
    "methods/failure_memory_regression/interaction.py",
)


def prepare_holdout(root: Path = ROOT) -> dict:
    root = Path(root)
    manifest_path = root / "protocol.json"
    if root.exists() and not manifest_path.is_file() and any(root.iterdir()):
        raise ValueError("holdout directory has files but no frozen protocol")
    per_seed = {}
    contexts = None
    for seed in SEEDS:
        cases = compile_interaction_scenarios(root / f"seed{seed}", seed, CATALOGUE)
        if len(cases) != 128:
            raise ValueError(f"expected 128 fixed candidates for seed {seed}")
        seed_contexts = sorted({case["context_id"] for case in cases})
        if contexts is None:
            contexts = seed_contexts
        elif seed_contexts != contexts:
            raise ValueError("source/target contexts changed across holdout seeds")
        protocol = json.loads((root / f"seed{seed}" / "protocol.json").read_text(
            encoding="utf-8"))
        per_seed[str(seed)] = protocol["candidate_sha256"]
    manifest = {
        "schema": "fbrt-interaction-age080-holdout-v1",
        "status": "candidates_frozen_no_physical_episodes",
        "seeds": list(SEEDS),
        "candidates_per_seed": 128,
        "planned_case_build_episodes": len(SEEDS) * 128 * len(BUILDS),
        "candidate_sha256_by_seed": per_seed,
        "context_ids": contexts,
        "catalogue_sha256": hashlib.sha256(CATALOGUE.read_bytes()).hexdigest(),
        "build_fingerprints": {build: build_spec_factory(build).fingerprint
                               for build in BUILDS},
        "source_code_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for path in CODE_PATHS},
        "target_outcome_filtering": False,
    }
    if manifest_path.is_file():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if recorded != manifest:
            raise ValueError("holdout protocol changed after freezing")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False,
                                            sort_keys=True, indent=2) + "\n",
                                 encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = prepare_holdout()
    print(json.dumps({key: result[key] for key in (
        "schema", "seeds", "candidates_per_seed", "planned_case_build_episodes",
        "status")}, ensure_ascii=False))
