"""Frozen 20 Hz FVDM-family B=50 source-safe regression validation."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from highway_sim_env.envs.cutin_env import CutInScenario
from methods.core_mine.data import CachedTask, _features, response_value
from methods.core_mine.development_mode_calibration import _run
from methods.core_mine.fvdm_revision_pilot import BUILDS
from methods.core_mine.idm_revision_pilot import _episode
from methods.core_mine.idm_revision_validation import (
    MODE_BOUNDS, SOURCE_FIELDS, _diverse_campaign, _metrics, _pool, _sha,
    _write_csv,
)
from methods.core_mine.source_safe_development import campaign


ROOT = Path("results/method_chains/core_mine/studies/fvdm_revision_validation")
QUALIFICATION_SEED = 20301121
VALIDATION_SEEDS = (20301205, 20301219, 20310109)
TARGETS = ("fvdm_delay05", "fvdm_brake3", "fvdm_delay05_brake3")
METHODS = ("ModeQuantile-Static", "HistoryMargin-Static",
           "RiskDiverse-Static", "HistoryMargin-Residual",
           "TargetOnly-Residual", "RandomSafe")


def _source_job(args: tuple[int, int, list[CutInScenario]]) -> list[dict]:
    seed, start, scenarios = args
    return [{"index": start + offset,
             **_episode(BUILDS["fvdm_ref"], scene, seed + start + offset)}
            for offset, scene in enumerate(scenarios)]


def build_source(seed: int, workers: int) -> dict:
    pool = _pool(seed)
    root = ROOT / str(seed)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "source_bank.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if not np.array_equal(old["anchors"],
                                  [[scene.initial_gap, scene.relative_speed]
                                   for scene in pool]):
                raise RuntimeError("existing FVDM source bank differs from frozen proposal")
        print(f"reused FVDM source seed={seed}", flush=True)
    else:
        jobs = [(seed, offset * 128, pool[offset * 128:(offset + 1) * 128])
                for offset in range(len(MODE_BOUNDS))]
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            groups = list(executor.map(_source_job, jobs))
        rows = sorted((row for group in groups for row in group),
                      key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(640)):
            raise RuntimeError("FVDM source proposal index mismatch")
        np.savez_compressed(path,
                            anchors=np.asarray([[scene.initial_gap, scene.relative_speed]
                                                for scene in pool]),
                            modes=np.asarray([scene.mode for scene in pool]),
                            controls=np.asarray([[scene.timing, scene.intensity]
                                                 for scene in pool]),
                            source_name="SM-Strong-FVDM", physics_frequency_hz=20,
                            **{field: np.asarray([row[field] for row in rows])
                               for field in SOURCE_FIELDS})
        print(f"executed FVDM source seed={seed}: 640 episodes", flush=True)
    with np.load(path, allow_pickle=False) as bank:
        safe = bank["completed"] & ~bank["ego_collision"] & ~bank["near_miss"]
        counts = {mode: int(np.sum(safe & (bank["modes"] == mode)))
                  for mode in MODE_BOUNDS}
        selected = np.concatenate([np.flatnonzero(safe & (bank["modes"] == mode))[:64]
                                   for mode in MODE_BOUNDS])
        selected_data = {key: bank[key][selected].copy()
                         for key in ("anchors", "modes", "controls")}
        source_data = {field: bank[field][selected].copy()
                       for field in SOURCE_FIELDS}
    passed = all(value >= 64 for value in counts.values())
    gate = {"seed": seed, "source_only_gate": True, "source_episodes": 640,
            "source_safe_by_mode": counts, "selected_candidates": len(selected),
            "passed": passed, "source_bank_sha256": _sha(path)}
    (root / "gate.json").write_text(json.dumps(gate, indent=2) + "\n",
                                    encoding="utf-8")
    if passed:
        candidate_path = root / "candidate_pool.npz"
        if candidate_path.exists():
            with np.load(candidate_path, allow_pickle=False) as old:
                if not np.array_equal(old["original_indices"], selected):
                    raise RuntimeError("existing FVDM candidate set differs from source gate")
        else:
            np.savez_compressed(candidate_path, original_indices=selected,
                                source_bank_sha256=gate["source_bank_sha256"],
                                **selected_data,
                                **{f"source_{field}": value
                                   for field, value in source_data.items()})
    print(json.dumps(gate, indent=2), flush=True)
    return gate


def _target_job(args: tuple[int, str, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray]) -> tuple[str, dict]:
    seed, target, indices, anchors, modes, controls = args
    values = {field: [] for field in SOURCE_FIELDS}
    for original, anchor, mode, control in zip(indices, anchors, modes, controls,
                                               strict=True):
        scene = CutInScenario(float(anchor[0]), float(anchor[1]), str(mode),
                              float(control[0]), float(control[1]))
        result = _episode(BUILDS[target], scene, seed + int(original))
        for field in SOURCE_FIELDS:
            values[field].append(result[field])
    return target, {field: np.asarray(items) for field, items in values.items()}


def build_targets(seed: int, workers: int) -> Path:
    root = ROOT / str(seed)
    gate = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    if not gate["passed"]:
        raise RuntimeError("FVDM target execution requires passed source gate")
    path = root / "target_bank.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if str(old["source_bank_sha256"]) != gate["source_bank_sha256"]:
                raise RuntimeError("existing FVDM target provenance mismatch")
        print(f"reused FVDM targets seed={seed}", flush=True)
        return path
    with np.load(root / "candidate_pool.npz", allow_pickle=False) as source:
        indices, anchors, modes, controls = (source[key].copy() for key in
                                            ("original_indices", "anchors",
                                             "modes", "controls"))
    jobs = [(seed, target, indices, anchors, modes, controls) for target in TARGETS]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        batches = dict(executor.map(_target_job, jobs))
    np.savez_compressed(path, original_indices=indices, anchors=anchors,
                        modes=modes, controls=controls,
                        target_names=np.asarray(TARGETS),
                        source_bank_sha256=gate["source_bank_sha256"],
                        **{field: np.stack([batches[target][field] for target in TARGETS])
                           for field in SOURCE_FIELDS})
    print(f"executed FVDM targets seed={seed}: {len(TARGETS) * len(indices)} episodes",
          flush=True)
    return path


def _task(seed: int, target: str) -> CachedTask:
    root = ROOT / str(seed)
    with np.load(root / "candidate_pool.npz", allow_pickle=False) as source, \
         np.load(root / "target_bank.npz", allow_pickle=False) as bank:
        target_index = list(bank["target_names"].astype(str)).index(target)
        anchors, modes, controls = (source[key].copy() for key in
                                    ("anchors", "modes", "controls"))
        source_event = (source["source_ego_collision"]
                        | source["source_near_miss"])[None, :]
        source_collision = source["source_ego_collision"][None, :].copy()
        source_ttc = source["source_min_ttc"][None, :].copy()
        target_event = bank["ego_collision"][target_index] | bank["near_miss"][target_index]
        target_collision = bank["ego_collision"][target_index].copy()
        target_ttc = bank["min_ttc"][target_index].copy()
    if source_event.any():
        raise RuntimeError("FVDM candidate set contains historical event")
    features, dimensions = _features(anchors, modes, controls)
    return CachedTask(seed, f"Target-{target}-fvdm-revision", "versioned", target,
                      anchors, modes, controls, features, dimensions,
                      response_value(source_ttc, source_event, source_collision),
                      source_event, source_collision,
                      response_value(target_ttc, target_event, target_collision),
                      target_event, target_collision, target_ttc, target_event.copy())


def _check_gates() -> None:
    for seed in (QUALIFICATION_SEED, *VALIDATION_SEEDS):
        gate_path = ROOT / str(seed) / "gate.json"
        if not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8"))["passed"]:
            raise RuntimeError(f"FVDM source-only gate missing or failed: {seed}")


def analyze() -> dict:
    _check_gates()
    rows = []
    for seed in VALIDATION_SEEDS:
        for target in TARGETS:
            task = _task(seed, target)
            settings = (
                ("ModeQuantile-Static", lambda: _run(task, "ModeQuantile-Static")),
                ("HistoryMargin-Static", lambda: campaign(task, "mean", 0.0,
                                                           residual=False)),
                ("RiskDiverse-Static", lambda: _diverse_campaign(task)),
                ("HistoryMargin-Residual", lambda: campaign(task, "mean", 0.0,
                                                             residual=True)),
                ("TargetOnly-Residual", lambda: campaign(task, "target", 0.0,
                                                          residual=True)),
            )
            for method, run in settings:
                rows.append(_metrics(task, {"method": method, "repeat": 0, **run()}))
            for repeat in range(10):
                rows.append(_metrics(task, {"method": "RandomSafe", "repeat": repeat,
                                            **campaign(task, None, 0.0, repeat=repeat)}))
            print(f"FVDM campaigns completed seed={seed} target={target}", flush=True)
    _write_csv(ROOT / "records.csv", rows)
    metric_names = ("NewHistoricalFailureCount", "CollisionCount", "NovelFailureModes",
                    "EarlyNovelAUC", "CVS")
    unit_rows = []
    for seed in VALIDATION_SEEDS:
        for target in TARGETS:
            for method in METHODS:
                subset = [row for row in rows if row["seed"] == seed
                          and row["heterogeneity"] == target and row["method"] == method]
                unit_rows.append({"seed": seed, "target": target, "method": method,
                                  "pool_new_failures": int(subset[0]["pool_new_failures"]),
                                  "pool_ego_collisions": int(subset[0]["pool_ego_collisions"]),
                                  **{metric: float(np.mean([row[metric] for row in subset]))
                                     for metric in metric_names}})
    unit = {(row["seed"], row["target"], row["method"]): row for row in unit_rows}
    summary = {method: {metric: float(np.mean([row[metric] for row in unit_rows
                if row["method"] == method])) for metric in metric_names}
                for method in METHODS}
    by_target = {target: {method: {metric: float(np.mean([row[metric]
                 for row in unit_rows if row["target"] == target and row["method"] == method]))
                 for metric in metric_names} for method in METHODS} for target in TARGETS}
    rng = np.random.default_rng(20310201)
    paired = {}
    for baseline in METHODS[1:]:
        paired[baseline] = {}
        for metric in ("NewHistoricalFailureCount", "CollisionCount", "CVS"):
            diff = np.asarray([[unit[(seed, target, METHODS[0])][metric]
                                - unit[(seed, target, baseline)][metric]
                                for target in TARGETS] for seed in VALIDATION_SEEDS])
            sample = np.empty(10000)
            for draw in range(len(sample)):
                seeds = rng.integers(0, len(VALIDATION_SEEDS), len(VALIDATION_SEEDS))
                targets = rng.integers(0, len(TARGETS),
                                       (len(VALIDATION_SEEDS), len(TARGETS)))
                sample[draw] = diff[seeds[:, None], targets].mean()
            paired[baseline][metric] = {
                "mean_difference": float(diff.mean()),
                "bootstrap_95_low": float(np.quantile(sample, .025)),
                "bootstrap_95_high": float(np.quantile(sample, .975)),
                "paired_unit_differences": diff.tolist(),
            }
    positive = all(paired[baseline]["NewHistoricalFailureCount"]["mean_difference"] > 0
                   for baseline in ("HistoryMargin-Static", "RiskDiverse-Static"))
    output = {"budget": 50, "qualification_seed": QUALIFICATION_SEED,
              "validation_seeds": VALIDATION_SEEDS, "target_builds": TARGETS,
              "family_generalization_mean_gate_passed": positive,
              "summary": summary, "by_target": by_target, "paired": paired,
              "unit_rows": unit_rows}
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"summary": summary, "paired": paired,
                      "family_generalization_mean_gate_passed": positive}, indent=2),
          flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("qualification", "sources", "targets",
                                           "analyze", "all"), default="qualification")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage in ("qualification", "all"):
        gate = build_source(QUALIFICATION_SEED, args.workers)
        if args.stage == "all" and not gate["passed"]:
            return
    if args.stage in ("sources", "all"):
        path = ROOT / str(QUALIFICATION_SEED) / "gate.json"
        if not path.exists() or not json.loads(path.read_text(encoding="utf-8"))["passed"]:
            raise RuntimeError("FVDM qualification source gate not passed")
        for seed in VALIDATION_SEEDS:
            gate = build_source(seed, args.workers)
            if not gate["passed"]:
                return
    if args.stage in ("targets", "all"):
        _check_gates()
        for seed in VALIDATION_SEEDS:
            build_targets(seed, args.workers)
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
