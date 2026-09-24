"""Frozen, same-family IDM configuration-regression study at B=50."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import qmc, rankdata

from highway_env_benchmark.envs.cutin_env import CutInScenario
from method_chains.core_mine.acquisition import choose, rbf_by_mode
from method_chains.core_mine.config import SUPPORT_BUDGET
from method_chains.core_mine.data import CachedTask, _features, response_value
from method_chains.core_mine.idm_revision_pilot import BUILDS, _episode
from method_chains.core_mine.metrics import record_metrics
from method_chains.core_mine.oracle import CacheOracle
from method_chains.core_mine.source_safe_development import campaign


ROOT = Path("results/method_chains/core_mine/studies/idm_revision_validation")
QUALIFICATION_SEED = 20291230
VALIDATION_SEEDS = (20300109, 20300123, 20300206)
TARGETS = ("idm_delay07", "idm_brake3", "idm_delay07_brake3")
MODE_BOUNDS = {
    "fast_intrusion": ((10.0, 50.0), (-8.0, -2.0)),
    "cutin_braking": ((10.0, 50.0), (-7.0, -1.0)),
    "lead_braking": ((12.0, 50.0), (-7.0, -1.0)),
    "stop_and_go": ((10.0, 50.0), (-6.0, -1.0)),
    "slow_lead_following": ((12.0, 50.0), (-8.0, -2.0)),
}
SOURCE_FIELDS = ("ego_collision", "background_collision", "near_miss",
                 "completed", "min_ttc", "min_clearance")
METHODS = ("HistoryMargin-Residual", "HistoryMargin-Static",
           "TargetOnly-Residual", "RiskDiverse-Static", "RandomSafe")
PAIRED_BASELINES = METHODS[1:]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def _pool(seed: int) -> list[CutInScenario]:
    pool = []
    for offset, (mode, (gap_bounds, speed_bounds)) in enumerate(MODE_BOUNDS.items()):
        unit = qmc.Sobol(d=4, scramble=True, seed=seed + offset).random_base2(m=7)
        for row in unit:
            pool.append(CutInScenario(
                float(gap_bounds[0] + row[0] * (gap_bounds[1] - gap_bounds[0])),
                float(speed_bounds[0] + row[1] * (speed_bounds[1] - speed_bounds[0])),
                mode, float(.05 + .90 * row[2]), float(.05 + .90 * row[3])))
    return pool


def _source_mode_job(args: tuple[int, int, list[CutInScenario]]) -> list[dict]:
    seed, start, scenarios = args
    return [{"index": start + offset,
             **_episode(BUILDS["idm_ref"], scenario, seed + start + offset)}
            for offset, scenario in enumerate(scenarios)]


def build_source(seed: int, workers: int) -> dict:
    pool = _pool(seed)
    root = ROOT / str(seed)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "source_bank.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if (not np.array_equal(data["anchors"], [s.as_array()[:2] for s in pool])
                    or tuple(data["modes"].astype(str)) != tuple(s.mode for s in pool)):
                raise RuntimeError("existing source bank does not match frozen proposal")
        print(f"reused source bank seed={seed}", flush=True)
    else:
        jobs = [(seed, offset * 128, pool[offset * 128:(offset + 1) * 128])
                for offset in range(len(MODE_BOUNDS))]
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            batches = list(executor.map(_source_mode_job, jobs))
        rows = sorted((row for batch in batches for row in batch),
                      key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(640)):
            raise RuntimeError("source bank index mismatch")
        np.savez_compressed(path,
                            anchors=np.asarray([[s.initial_gap, s.relative_speed] for s in pool]),
                            modes=np.asarray([s.mode for s in pool]),
                            controls=np.asarray([[s.timing, s.intensity] for s in pool]),
                            source_name="idm_ref", physics_frequency_hz=20,
                            **{field: np.asarray([row[field] for row in rows])
                               for field in SOURCE_FIELDS})
        print(f"executed source seed={seed}: 640 episodes", flush=True)
    with np.load(path, allow_pickle=False) as bank:
        safe = bank["completed"] & ~bank["ego_collision"] & ~bank["near_miss"]
        counts = {mode: int(np.sum(safe & (bank["modes"] == mode)))
                  for mode in MODE_BOUNDS}
        selected = np.concatenate([np.flatnonzero(safe & (bank["modes"] == mode))[:64]
                                   for mode in MODE_BOUNDS])
        source_outcomes = {field: bank[field][selected].copy()
                           for field in SOURCE_FIELDS}
        anchors = bank["anchors"][selected].copy()
        modes = bank["modes"][selected].copy()
        controls = bank["controls"][selected].copy()
    passed = all(count >= 64 for count in counts.values())
    output = {"seed": seed, "source_only_gate": True,
              "source_episodes": 640, "source_safe_by_mode": counts,
              "selected_candidates": len(selected), "passed": passed,
              "source_bank_sha256": _sha(path)}
    (root / "gate.json").write_text(json.dumps(output, indent=2) + "\n",
                                    encoding="utf-8")
    if passed:
        candidate_path = root / "candidate_pool.npz"
        if candidate_path.exists():
            with np.load(candidate_path, allow_pickle=False) as old:
                if not np.array_equal(old["original_indices"], selected):
                    raise RuntimeError("existing candidate pool differs from frozen source gate")
        else:
            np.savez_compressed(candidate_path, original_indices=selected,
                                anchors=anchors, modes=modes, controls=controls,
                                source_bank_sha256=output["source_bank_sha256"],
                                **{f"source_{field}": value for field, value in
                                   source_outcomes.items()})
    print(json.dumps(output, indent=2), flush=True)
    return output


def _target_job(args: tuple[int, str, np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray]) -> tuple[str, dict[str, np.ndarray]]:
    seed, target, indices, anchors, modes, controls = args
    values: dict[str, list] = {field: [] for field in SOURCE_FIELDS}
    for original_index, anchor, mode, control in zip(indices, anchors, modes, controls,
                                                      strict=True):
        scenario = CutInScenario(float(anchor[0]), float(anchor[1]), str(mode),
                                 float(control[0]), float(control[1]))
        result = _episode(BUILDS[target], scenario, seed + int(original_index))
        for field in SOURCE_FIELDS:
            values[field].append(result[field])
    return target, {field: np.asarray(items) for field, items in values.items()}


def build_targets(seed: int, workers: int) -> Path:
    root = ROOT / str(seed)
    gate = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    if not gate["passed"]:
        raise RuntimeError("target execution requires source-only gate")
    source_path = root / "candidate_pool.npz"
    path = root / "target_bank.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as bank:
            if str(bank["source_bank_sha256"]) != gate["source_bank_sha256"]:
                raise RuntimeError("target bank source provenance mismatch")
        print(f"reused target bank seed={seed}", flush=True)
        return path
    with np.load(source_path, allow_pickle=False) as source:
        indices, anchors, modes, controls = (source[key].copy() for key in
                                            ("original_indices", "anchors", "modes", "controls"))
    jobs = [(seed, target, indices, anchors, modes, controls) for target in TARGETS]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        batches = dict(executor.map(_target_job, jobs))
    np.savez_compressed(path, original_indices=indices, anchors=anchors,
                        modes=modes, controls=controls,
                        target_names=np.asarray(TARGETS),
                        source_bank_sha256=gate["source_bank_sha256"],
                        **{field: np.stack([batches[target][field] for target in TARGETS])
                           for field in SOURCE_FIELDS})
    print(f"executed target seed={seed}: {len(TARGETS) * len(indices)} episodes", flush=True)
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
        source_collision = source["source_ego_collision"][None, :]
        source_ttc = source["source_min_ttc"][None, :]
        target_event = bank["ego_collision"][target_index] | bank["near_miss"][target_index]
        target_collision = bank["ego_collision"][target_index].copy()
        target_ttc = bank["min_ttc"][target_index].copy()
    if source_event.any():
        raise RuntimeError("candidate pool contains historical event")
    features, dimensions = _features(anchors, modes, controls)
    return CachedTask(seed, f"Target-{target}-idm-revision", "versioned", target,
                      anchors, modes, controls, features, dimensions,
                      response_value(source_ttc, source_event, source_collision),
                      source_event, source_collision,
                      response_value(target_ttc, target_event, target_collision),
                      target_event, target_collision, target_ttc, target_event.copy())


def _diverse_campaign(task: CachedTask) -> dict:
    oracle = CacheOracle(task)
    n = task.count
    risk = (rankdata(task.source_y.mean(axis=0), method="average") - 1) / (n - 1)
    all_indices = np.arange(n)
    while len(oracle.revealed) < 50:
        if oracle.revealed:
            similarity = rbf_by_mode(task.features, task.modes, all_indices,
                                     np.asarray(oracle.revealed), length=.20)
            novelty = 1.0 - similarity.max(axis=1)
        else:
            novelty = np.ones(n)
        scores = .5 * risk + .5 * novelty
        index = choose(scores, oracle.revealed, task.modes, SUPPORT_BUDGET)
        oracle.reveal(index)
    return record_metrics(task, oracle.revealed, 50)


def _metrics(task: CachedTask, record: dict) -> dict:
    indices = np.asarray([int(value) for value in record["queried_indices"].split(";")])
    if len(indices) != 50 or len(np.unique(indices)) != 50:
        raise RuntimeError("campaign did not charge 50 unique queries")
    events = task.target_event[indices]
    if int(record["NewHistoricalFailureCount"]) != int(events.sum()):
        raise RuntimeError("recorded target event count mismatch")
    return {**record, "NovelFailureModes": len(set(task.modes[indices][events].astype(str))),
            "EarlyNovelAUC": float(np.cumsum(events).sum() / (50 * 51 / 2)),
            "pool_new_failures": int(task.target_event.sum()),
            "pool_ego_collisions": int(task.target_collision.sum()),
            "new_failure_recall": float(events.sum() / task.target_event.sum())
            if task.target_event.any() else None}


def analyze() -> dict:
    rows = []
    for seed in VALIDATION_SEEDS:
        for target in TARGETS:
            task = _task(seed, target)
            for method, branch, residual in (
                ("HistoryMargin-Residual", "mean", True),
                ("HistoryMargin-Static", "mean", False),
                ("TargetOnly-Residual", "target", True),
            ):
                row = campaign(task, branch, 0.0, residual=residual)
                rows.append(_metrics(task, {"method": method, "repeat": 0, **row}))
            rows.append(_metrics(task, {"method": "RiskDiverse-Static", "repeat": 0,
                                        **_diverse_campaign(task)}))
            for repeat in range(10):
                row = campaign(task, None, 0.0, repeat=repeat)
                rows.append(_metrics(task, {"method": "RandomSafe", "repeat": repeat,
                                            **row}))
            print(f"campaigns completed seed={seed} target={target}", flush=True)
    _write_csv(ROOT / "records.csv", rows)
    unit_rows = []
    metric_names = ("NewHistoricalFailureCount", "CollisionCount", "NovelFailureModes",
                    "EarlyNovelAUC", "CVS", "new_failure_recall")
    for seed in VALIDATION_SEEDS:
        for target in TARGETS:
            for method in METHODS:
                subset = [row for row in rows if row["seed"] == seed
                          and row["heterogeneity"] == target and row["method"] == method]
                unit_rows.append({"seed": seed, "target": target, "method": method,
                                  "pool_new_failures": int(subset[0]["pool_new_failures"]),
                                  **{metric: float(np.mean([float(row[metric]) for row in subset]))
                                     if subset[0][metric] is not None else None
                                     for metric in metric_names}})
    summary = {method: {metric: float(np.mean([row[metric] for row in unit_rows
                                              if row["method"] == method]))
                        for metric in metric_names if metric != "new_failure_recall"}
               for method in METHODS}
    by_target = {target: {method: {metric: float(np.mean([row[metric]
                    for row in unit_rows if row["target"] == target
                    and row["method"] == method]))
                    for metric in metric_names if metric != "new_failure_recall"}
                    for method in METHODS} for target in TARGETS}
    # Resample seed clusters and target builds within seeds. RandomSafe has
    # already been averaged over its ten orders inside each paired unit.
    unit = {(row["seed"], row["target"], row["method"]): row for row in unit_rows}
    rng = np.random.default_rng(20300301)
    paired = {}
    for baseline in PAIRED_BASELINES:
        paired[baseline] = {}
        for metric in ("NewHistoricalFailureCount", "CollisionCount", "CVS"):
            differences = np.asarray([[unit[(seed, target, METHODS[0])][metric]
                                       - unit[(seed, target, baseline)][metric]
                                       for target in TARGETS] for seed in VALIDATION_SEEDS])
            samples = np.empty(10000)
            for draw in range(len(samples)):
                seed_indices = rng.integers(0, len(VALIDATION_SEEDS), len(VALIDATION_SEEDS))
                target_indices = rng.integers(0, len(TARGETS),
                                              (len(VALIDATION_SEEDS), len(TARGETS)))
                samples[draw] = np.mean(differences[seed_indices[:, None], target_indices])
            paired[baseline][metric] = {
                "mean_difference": float(differences.mean()),
                "bootstrap_95_low": float(np.quantile(samples, .025)),
                "bootstrap_95_high": float(np.quantile(samples, .975)),
                "paired_unit_differences": differences.tolist(),
            }
    output = {"budget": 50, "qualification_seed": QUALIFICATION_SEED,
              "validation_seeds": VALIDATION_SEEDS, "target_builds": TARGETS,
              "summary": summary, "by_target": by_target, "paired": paired,
              "unit_rows": unit_rows}
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"summary": summary, "by_target": by_target}, indent=2), flush=True)
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
        qual = ROOT / str(QUALIFICATION_SEED) / "gate.json"
        if not qual.exists() or not json.loads(qual.read_text(encoding="utf-8"))["passed"]:
            raise RuntimeError("validation requires passed independent qualification")
        for seed in VALIDATION_SEEDS:
            gate = build_source(seed, args.workers)
            if not gate["passed"]:
                return
    if args.stage in ("targets", "all"):
        for seed in VALIDATION_SEEDS:
            build_targets(seed, args.workers)
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
