"""Frozen clean-source robustness of heterogeneous 20 Hz CoRe transfer."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from highway_sim_env.envs.cutin_env import CutInScenario
from methods.core_mine import heterogeneous20_replication as base
from methods.core_mine import sparse_sut_experiment as sparse
from methods.core_mine.fvdm_revision_pilot import BUILDS
from methods.core_mine.idm_revision_pilot import _episode


ROOT = Path("results/method_chains/core_mine/studies/fvdm_source_robustness")
SEEDS = (20320405, 20320419, 20320503)
SOURCES = ("idm_mobil", "fvdm_ref")


def _configure() -> None:
    """Reuse the frozen target selectors against a separate source bank."""
    base.ROOT = ROOT
    base.SEEDS = SEEDS
    base.SOURCES = SOURCES


def _source_job(args: tuple[int, str, np.ndarray, np.ndarray,
                            np.ndarray]) -> tuple[str, dict[str, np.ndarray]]:
    seed, sut, anchors, modes, controls = args
    if sut == "idm_mobil":
        return base._source_job(args)
    if sut != "fvdm_ref":
        raise ValueError(sut)
    values: dict[str, list] = {field: [] for field in base.FIELDS}
    for index, (anchor, mode, control) in enumerate(zip(anchors, modes, controls,
                                                        strict=True)):
        scenario = CutInScenario(float(anchor[0]), float(anchor[1]), str(mode),
                                 float(control[0]), float(control[1]))
        result = _episode(BUILDS["fvdm_ref"], scenario, seed + index)
        result["min_distance"] = result.pop("min_clearance")
        for field in base.FIELDS:
            values[field].append(result[field])
    return sut, {field: np.asarray(items) for field, items in values.items()}


def build_sources(seed: int, workers: int) -> dict:
    sparse.configure_proposal("v8_source_safe")
    anchors, modes, controls, regimes = sparse.sparse_scenarios(seed)
    if len(modes) != 320 or len(np.unique(modes)) != 5:
        raise RuntimeError("frozen proposal must contain five modes x 64")
    directory = ROOT / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "source_bank.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if not (np.array_equal(old["anchors"], anchors)
                    and np.array_equal(old["modes"], modes)
                    and np.array_equal(old["controls"], controls)
                    and tuple(old["source_names"].astype(str)) == SOURCES
                    and int(old["decision_frequency_hz"]) == 20):
                raise RuntimeError("existing source bank differs from frozen protocol")
        print(f"reused clean sources seed={seed}", flush=True)
    else:
        jobs = [(seed, sut, anchors, modes, controls) for sut in SOURCES]
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            batches = dict(pool.map(_source_job, jobs))
        np.savez_compressed(
            path, anchors=anchors, modes=modes, controls=controls, regimes=regimes,
            source_names=np.asarray(SOURCES),
            source_profiles=np.asarray(("highway-env IDM/MOBIL", "SM-Strong-FVDM")),
            decision_frequency_hz=20, physics_frequency_hz=20,
            **{field: np.stack([batches[sut][field] for sut in SOURCES])
               for field in base.FIELDS},
        )
        print(f"executed clean sources seed={seed}: 640 episodes", flush=True)
    with np.load(path, allow_pickle=False) as bank:
        event = bank["ego_collision"] | bank["near_miss"]
        eligible = ~event.any(axis=0) & bank["completed"].all(axis=0)
        per_mode = {str(mode): int(np.sum(eligible & (bank["modes"] == mode)))
                    for mode in np.unique(bank["modes"])}
    output = {"seed": seed, "sources": SOURCES,
              "source_profiles": ("highway-env IDM/MOBIL", "SM-Strong-FVDM"),
              "target": base.TARGET, "source_episodes": 640,
              "decision_frequency_hz": 20, "physics_frequency_hz": 20,
              "eligible_candidates": int(eligible.sum()),
              "eligible_by_mode": per_mode,
              "passed": bool(eligible.sum() >= 50 and all(per_mode.values())),
              "source_bank_sha256": base._sha(path)}
    (directory / "qualification.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output), flush=True)
    return output


def main() -> None:
    _configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "targets", "analyze", "all"),
                        default="sources")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage in ("sources", "all"):
        for seed in SEEDS:
            build_sources(seed, args.workers)
    if args.stage in ("targets", "all"):
        base._gate_all()
        for seed in SEEDS:
            for method, branch, residual in base.METHODS:
                base.run_campaign(seed, method, branch, residual)
    if args.stage in ("analyze", "all"):
        base.analyze()


if __name__ == "__main__":
    main()
