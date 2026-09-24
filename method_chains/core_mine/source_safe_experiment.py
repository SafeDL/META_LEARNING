"""Frozen B=50 historical blind-spot study with corrected safety labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from method_chains.core_mine import sparse_sut_experiment as sparse
from method_chains.core_mine.source_safe_development import campaign


PROPOSAL = "v8_source_safe"
METHODS = (
    ("HistoryMargin-Residual", "mean", True),
    ("HistoryMargin-Static", "mean", False),
    ("TargetOnly-Residual", "target", True),
    ("CoRe-Residual", "composition", True),
)
PRIMARY_TARGETS = ("mcts_cv", "vi_ttc")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def qualification(workers: int) -> dict:
    seed = sparse.QUALIFICATION_SEEDS[0]
    bank = sparse.load_or_build(seed, workers)
    rows = []
    for task in sparse.tasks_from_bank(bank, seed):
        eligible = ~task.source_event.any(axis=0)
        failures = eligible & task.target_event
        rows.append({"target": task.heterogeneity, "eligible_candidates": int(eligible.sum()),
                     "eligible_modes": sorted(set(task.modes[eligible].astype(str))),
                     "source_safe_target_failures": int(failures.sum()),
                     "failure_modes": sorted(set(task.modes[failures].astype(str))),
                     "ego_collisions_among_failures": int((failures & task.target_collision).sum())})
    lookup = {row["target"]: row for row in rows}
    gates = {
        "all_targets_have_50_eligible_in_five_modes": all(
            row["eligible_candidates"] >= 50 and len(row["eligible_modes"]) == 5 for row in rows),
        "mcts_has_5_failures_in_2_modes": lookup["mcts_cv"]["source_safe_target_failures"] >= 5
        and len(lookup["mcts_cv"]["failure_modes"]) >= 2,
        "vi_has_10_failures_in_2_modes": lookup["vi_ttc"]["source_safe_target_failures"] >= 10
        and len(lookup["vi_ttc"]["failure_modes"]) >= 2,
    }
    output = {"proposal": PROPOSAL, "seed": seed, "rows": rows, "gates": gates,
              "passed": all(gates.values())}
    path = sparse.ROOT / "qualification" / "gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return output


def _campaign_rows(task) -> list[dict]:
    rows = []
    for method, branch, residual in METHODS:
        result = campaign(task, branch, 0.0, residual=residual)
        rows.append({"method": method, "repeat": 0, **result})
    for repeat in range(10):
        result = campaign(task, None, 0.0, repeat=repeat)
        rows.append({"method": "RandomSafe", "repeat": repeat, **result})
    for row in rows:
        if row["CriticalCount"] != row["NewHistoricalFailureCount"]:
            raise RuntimeError("an eligible target event must be historically new")
    return rows


def _bootstrap(rows: list[dict], method: str, baseline: str, metric: str) -> dict[str, float]:
    lookup = {(row["seed"], row["target"], row["method"]): row for row in rows}
    seeds = sparse.VALIDATION_SEEDS
    differences = np.asarray([[lookup[seed, target, method][metric] - lookup[seed, target, baseline][metric]
                               for target in PRIMARY_TARGETS] for seed in seeds], dtype=float)
    rng = np.random.default_rng(20260924)
    draws = np.empty(5000)
    for index in range(len(draws)):
        sampled_seeds = rng.integers(0, len(seeds), len(seeds))
        sampled_targets = rng.integers(0, len(PRIMARY_TARGETS), differences.shape)
        draws[index] = differences[sampled_seeds[:, None], sampled_targets].mean()
    return {"difference": float(differences.mean()), "low": float(np.quantile(draws, .025)),
            "high": float(np.quantile(draws, .975)), "unit_differences": differences.tolist()}


def analyze(records: list[dict]) -> dict:
    banks = {}
    for seed in sparse.VALIDATION_SEEDS:
        with np.load(sparse.BANK_DIR / f"sparse_sut_bank_{seed}.npz", allow_pickle=False) as bank:
            banks[seed] = {key: bank[key].copy() for key in ("sut_names", "modes", "ego_collision", "near_miss")}
    by_unit: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for record in records:
        row = dict(record)
        seed, target = int(row["seed"]), str(row["heterogeneity"])
        bank = banks[seed]
        target_index = list(bank["sut_names"].astype(str)).index(target)
        indices = np.asarray([int(value) for value in str(row["queried_indices"]).split(";")])
        if len(indices) != 50 or len(set(indices)) != 50:
            raise ValueError("source-safe campaign must make 50 unique queries")
        source_indices = np.arange(len(bank["sut_names"])) != target_index
        source_events = bank["ego_collision"][source_indices][:, indices] | bank["near_miss"][source_indices][:, indices]
        if source_events.any():
            raise ValueError("campaign selected a scenario that was unsafe for a historical controller")
        events = bank["ego_collision"][target_index, indices] | bank["near_miss"][target_index, indices]
        if int(row["NewHistoricalFailureCount"]) != int(events.sum()):
            raise ValueError("selected target events disagree with recorded new failures")
        row["NovelFailureModes"] = len(set(bank["modes"][indices][events].astype(str)))
        row["EarlyNovelAUC"] = float(np.cumsum(events).sum() / (50 * 51 / 2))
        by_unit[seed, target, str(row["method"])].append(row)
    metrics = ("NewHistoricalFailureCount", "CollisionCount", "CVS", "F", "NovelFailureModes", "EarlyNovelAUC")
    unit_rows = []
    for (seed, target, method), repeats in by_unit.items():
        unit_rows.append({"seed": seed, "target": target, "method": method,
                          **{metric: float(np.mean([float(row[metric]) for row in repeats])) for metric in metrics}})
    methods = (*[item[0] for item in METHODS], "RandomSafe")
    summary = {}
    for cohort, targets in (("primary", PRIMARY_TARGETS), ("all", tuple(sparse.ACTIVE_SUTS))):
        summary[cohort] = {method: {metric: float(np.mean([
            row[metric] for row in unit_rows if row["method"] == method and row["target"] in targets]))
            for metric in metrics} for method in methods}
    comparisons = {f"HistoryMargin-Residual vs {baseline}": {
        metric: _bootstrap(unit_rows, "HistoryMargin-Residual", baseline, metric) for metric in metrics
    } for baseline in ("HistoryMargin-Static", "TargetOnly-Residual", "RandomSafe", "CoRe-Residual")}
    return {"budget": 50, "primary_targets": PRIMARY_TARGETS, "validation_seeds": sparse.VALIDATION_SEEDS,
            "summary": summary, "paired_primary": comparisons, "unit_rows": unit_rows}


def _write_analysis(records: list[dict]) -> None:
    report = analyze(records)
    (sparse.ROOT / "analysis50.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Historical blind-spot testing at B=50", "",
             "Primary targets were frozen as MCTS and VI/TTC; all four targets are in `analysis50.json`.", "",
             "| Method | New failures@50 | Ego collisions@50 | Failure modes | Early AUC | CVS@50 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in (*[item[0] for item in METHODS], "RandomSafe"):
        values = report["summary"]["primary"][method]
        lines.append(f"| {method} | {values['NewHistoricalFailureCount']:.2f} | "
                     f"{values['CollisionCount']:.2f} | {values['NovelFailureModes']:.2f} | "
                     f"{values['EarlyNovelAUC']:.3f} | {values['CVS']:.2f} |")
    lines += ["", "Paired hierarchical bootstrap intervals and every target unit are in `analysis50.json`."]
    (sparse.ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def validate(workers: int) -> None:
    gate_path = sparse.ROOT / "qualification" / "gate.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8"))["passed"]:
        raise RuntimeError("v8 validation requires its independent passed qualification")
    records = []
    hashes = {}
    for seed in sparse.VALIDATION_SEEDS:
        bank = sparse.load_or_build(seed, workers)
        bank_path = sparse.BANK_DIR / f"sparse_sut_bank_{seed}.npz"
        hashes[bank_path.name] = hashlib.sha256(bank_path.read_bytes()).hexdigest()
        for task in sparse.tasks_from_bank(bank, seed):
            records.extend(_campaign_rows(task))
            print(f"completed seed={seed} target={task.heterogeneity}", flush=True)
    _write_csv(sparse.ROOT / "validate" / "records.csv", records)
    _write_analysis(records)
    manifest = {"proposal": PROPOSAL, "safety_metric_version": "polygon_clearance_1m_v1",
                "budget": 50, "suts": sparse.ACTIVE_SUTS,
                "qualification_seed": sparse.QUALIFICATION_SEEDS[0],
                "validation_seeds": sparse.VALIDATION_SEEDS,
                "candidates_per_seed": sparse.PER_MODE * len(sparse.MODES),
                "qualification_physical_episodes": len(sparse.ACTIVE_SUTS) * sparse.PER_MODE * len(sparse.MODES),
                "validation_physical_episodes": len(sparse.VALIDATION_SEEDS) * len(sparse.ACTIVE_SUTS)
                * sparse.PER_MODE * len(sparse.MODES),
                "validation_bank_sha256": hashes}
    (sparse.ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("qualification", "validate", "analyze", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sparse.configure_proposal(PROPOSAL)
    if args.stage in {"qualification", "all"}:
        gate = qualification(args.workers)
        if args.stage == "all" and not gate["passed"]:
            return
    if args.stage in {"validate", "all"}:
        validate(args.workers)
    if args.stage == "analyze":
        with (sparse.ROOT / "validate" / "records.csv").open(encoding="utf-8", newline="") as handle:
            _write_analysis(list(csv.DictReader(handle)))


if __name__ == "__main__":
    main()
