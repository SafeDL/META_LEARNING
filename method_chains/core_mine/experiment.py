"""CLI for the pre-registered CoRe-Mine cache experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .acquisition import choose, marginal_scores, verified_novelty_scores
from .config import BANK_DIR, BUDGETS, DEVELOPMENT_BANKS, ROOT, SUPPORT_BUDGET, TOTAL_BUDGET, VALIDATION_BANKS, CoreMineConfig
from .data import CachedTask, load_tasks
from .metrics import record_metrics
from .oracle import CacheOracle
from .posterior import PosteriorModel
from .report import write_report


METHODS = ("FPS-Risk", "FPS-Severity", "FPS-Balanced", "FPS-Marginal", "MeanResidual-Risk", "MeanResidual-Marginal", "CoRe-Risk", "CoRe-Marginal", "TargetOnlyGP-Marginal")
ABLATIONS = ("NoComposition", "NoNull")


def _model(method: str, task: CachedTask, config: CoreMineConfig) -> PosteriorModel:
    if method.startswith("FPS"):
        return PosteriorModel(task, config, "composition", False)
    if method.startswith("MeanResidual"):
        return PosteriorModel(task, config, "mean", True)
    if method.startswith("TargetOnly"):
        return PosteriorModel(task, config, "target", True)
    if method == "NoComposition":
        return PosteriorModel(task, config, "global", True)
    if method == "NoNull":
        return PosteriorModel(task, CoreMineConfig(config.residual_length, config.residual_amplitude, config.observation_noise, config.lambda_, False, True), "composition", True)
    return PosteriorModel(task, config, "composition", True)


def run_campaign(task: CachedTask, method: str, config: CoreMineConfig, repeat: int = 0) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run one target-hidden 50-step campaign; scoring happens only afterwards."""
    started = time.perf_counter(); oracle = CacheOracle(task); trace: list[dict[str, object]] = []
    if method == "Random":
        stable = int(hashlib.sha256(task.target_name.encode("utf-8")).hexdigest()[:8], 16)
        selected = np.random.default_rng(task.seed + repeat * 997 + stable % 997).permutation(task.count)[:TOTAL_BUDGET].tolist()
        for index in selected: oracle.reveal(int(index))
    else:
        model = _model(method, task, config)
        revealed_severities: list[float] = []
        while len(oracle.revealed) < TOTAL_BUDGET:
            prediction = model.predict()
            if method.endswith("Novelty"):
                scores, balanced = verified_novelty_scores(task.features, task.modes, oracle.revealed,
                                                            revealed_severities, prediction["p_event"], config.lambda_), False
            elif method.endswith("Risk"):
                scores, balanced = prediction["p_event"], False
            elif method.endswith("Severity"):
                scores, balanced = 0.5 * prediction["p_event"] + 0.5 * prediction["p_collision"], False
            elif method.endswith("Balanced"):
                scores, balanced = 0.5 * prediction["p_event"] + 0.5 * prediction["p_collision"], True
            else:
                scores, balanced = marginal_scores(task.features, task.modes, oracle.revealed, revealed_severities, prediction["p_event"], prediction["p_collision"], config.lambda_), False
            index = choose(scores, oracle.revealed, task.modes, SUPPORT_BUDGET, balanced)
            outcome = oracle.reveal(index)
            trace.append({"seed": task.seed, "target": task.target_name, "method": method, "step": len(oracle.revealed), "index": index,
                          "p_event": float(prediction["p_event"][index]), "p_collision": float(prediction["p_collision"][index]),
                          "mean": float(prediction["mean"][index]), "variance": float(prediction["variance"][index]), "revealed_y": outcome.y,
                          "revealed_event": outcome.event, "revealed_collision": outcome.collision, "charged": True})
            model.observe(index, outcome)
            revealed_severities.append(outcome.severity)
        selected = oracle.revealed
    elapsed = time.perf_counter() - started
    rows = []
    for budget in BUDGETS:
        row = record_metrics(task, selected, budget)
        row.update({"method": method, "repeat": repeat, "selection_seconds": elapsed})
        rows.append(row)
    return rows, trace


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _tasks(seeds: tuple[int, ...]) -> list[CachedTask]:
    return [task for seed in seeds for task in load_tasks(BANK_DIR / f"response_bank_{seed}.npz", seed)]


def audit() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, object] = {"banks": [], "primary_pool": "five two-vehicle modes; passing_cutin excluded", "event": "ego collision OR ego near miss", "response": "0.25 exp(-max(TTC,0)/3) + 0.5 E + 0.5 C"}
    rows: list[dict[str, object]] = []
    for seed in (*DEVELOPMENT_BANKS, *VALIDATION_BANKS):
        path = BANK_DIR / f"response_bank_{seed}.npz"; tasks = load_tasks(path, seed)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        item = {"seed": seed, "file": str(path), "sha256": digest, "targets": len(tasks), "candidates_per_target": tasks[0].count, "modes": sorted(set(tasks[0].modes))}
        inventory["banks"].append(item)
        for task in tasks:
            rows.append({"seed": seed, "target": task.target_name, "coverage": task.coverage, "heterogeneity": task.heterogeneity,
                         "candidates": task.count, "events": int(task.target_event.sum()), "collisions": int(task.target_collision.sum()),
                         "severity_sum": float(np.where(task.target_collision, 1.0, np.where(task.target_event, .5, 0.)).sum()),
                         "historically_safe_target_failures": int(task.source_safe_target_failure.sum()),
                         "function_distribution": ";".join(f"{m}:{int(np.sum(task.modes == m))}" for m in np.unique(task.modes))})
    inventory["total_target_units"] = len(rows); inventory["python"] = sys.version
    (ROOT / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    _csv(ROOT / "baseline_recheck.csv", rows)
    total_events, total_collisions = sum(int(x["events"]) for x in rows), sum(int(x["collisions"]) for x in rows)
    safe_failures = sum(int(x["historically_safe_target_failures"]) for x in rows)
    (ROOT / "opportunity.md").write_text(
        "# Cache opportunity scan\n\n"
        f"The frozen cache contains {len(rows)} target units, each with 500 eligible two-vehicle scenarios across five functions. "
        f"It contains {total_events} critical events, including {total_collisions} collisions, and {safe_failures} target failures at candidates safe for every source.\n\n"
        "Thus hit counts can saturate while archive coverage still has room: repeated failures in one function/region are discounted by CVS and F. "
        "The historically-safe target failures establish that target-local discrepancies exist, but are reported as a descriptive subgroup rather than a selection input.\n",
        encoding="utf-8")


def develop(max_trials: int) -> CoreMineConfig:
    tasks = _tasks(DEVELOPMENT_BANKS)
    grid = [(0.30, .25, .10), (.15, .15, 0), (.15, .25, .10), (.15, .50, .30), (.30, .15, .10), (.30, .25, 0), (.30, .50, .30), (.60, .15, 0), (.60, .25, .10), (.60, .50, .30), (.15, .50, .10), (.60, .15, .30)][:max_trials]
    trials: list[dict[str, object]] = []
    for trial, (length, amplitude, lam) in enumerate(grid, 1):
        config = CoreMineConfig(length, amplitude, lambda_=lam)
        records = [row for task in tasks for row in run_campaign(task, "CoRe-Marginal", config)[0] if row["budget"] == 20]
        trials.append({"trial": trial, **config.as_dict(), "units": len(records), "mean_CVS20": float(np.mean([r["CVS"] for r in records])), "mean_SeveritySum20": float(np.mean([r["SeveritySum"] for r in records])), "mean_CollisionCount20": float(np.mean([r["CollisionCount"] for r in records]))})
    _csv(ROOT / "trials.csv", trials)
    # Require severity within 90% of the development best before selecting the highest coverage trial.
    best_severity = max(float(row["mean_SeveritySum20"]) for row in trials)
    eligible = [row for row in trials if float(row["mean_SeveritySum20"]) >= .9 * best_severity]
    winner = max(eligible, key=lambda row: (float(row["mean_CVS20"]), float(row["mean_SeveritySum20"]), -int(row["trial"])))
    config = CoreMineConfig(float(winner["residual_length"]), float(winner["residual_amplitude"]), lambda_=float(winner["lambda"]))
    (ROOT / "frozen_config.json").write_text(json.dumps({"selected_trial": winner, "config": config.as_dict(), "selection": "highest development CVS@20 among trials retaining >=90% of best SeveritySum@20"}, indent=2) + "\n", encoding="utf-8")
    # Retain matched development records for every required method under the frozen configuration.
    records, traces = _evaluate(tasks, METHODS, config)
    _csv(ROOT / "develop" / "records.csv", records); _jsonl(ROOT / "develop" / "trajectories.jsonl", traces)
    return config


def _evaluate(tasks: list[CachedTask], methods: tuple[str, ...], config: CoreMineConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []; traces: list[dict[str, object]] = []
    for task in tasks:
        for method in methods:
            result, audit_trace = run_campaign(task, method, config)
            records.extend(result); traces.extend(audit_trace)
        for repeat in range(10):
            result, audit_trace = run_campaign(task, "Random", config, repeat)
            records.extend(result); traces.extend(audit_trace)
    return records, traces


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")


def validate(config: CoreMineConfig) -> None:
    records, traces = _evaluate(_tasks(VALIDATION_BANKS), METHODS, config)
    _csv(ROOT / "validate" / "records.csv", records); _jsonl(ROOT / "trajectories.jsonl", traces)
    _csv(ROOT / "validate" / "records.csv", records)
    # E3: the full candidate is deliberately ablated regardless of its outcome.
    ablations, _ = _evaluate(_tasks(VALIDATION_BANKS), ABLATIONS, config)
    _csv(ROOT / "ablations.csv", ablations)
    _csv(ROOT / "summary.csv", summarize(records + ablations))
    _csv(ROOT / "paired_comparisons.csv", paired_comparisons(records))


def tune_simple_objective() -> dict[str, object]:
    """Second, pre-listed development check for the coverage--severity trade-off."""
    tasks = _tasks(DEVELOPMENT_BANKS)
    baseline = [row for task in tasks for row in run_campaign(task, "FPS-Severity", CoreMineConfig())[0] if row["budget"] == 20]
    baseline_severity = float(np.mean([float(row["SeveritySum"]) for row in baseline]))
    rows = []
    for lambda_ in (0.0, 0.10, 0.30):
        records = [row for task in tasks for row in run_campaign(task, "FPS-Marginal", CoreMineConfig(lambda_=lambda_))[0] if row["budget"] == 20]
        rows.append({"method": "FPS-Marginal", "lambda": lambda_, "units": len(records), "mean_CVS20": float(np.mean([float(row["CVS"]) for row in records])), "mean_SeveritySum20": float(np.mean([float(row["SeveritySum"]) for row in records])), "reference_FPS_SeveritySum20": baseline_severity, "retains_90pct_severity": float(np.mean([float(row["SeveritySum"]) for row in records])) >= .9 * baseline_severity})
    _csv(ROOT / "objective_trials.csv", rows)
    eligible = [row for row in rows if bool(row["retains_90pct_severity"])]
    return max(eligible, key=lambda row: float(row["mean_CVS20"])) if eligible else {"status": "no lambda retained 90% severity"}


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        buckets[str(row["method"]), int(row["budget"]), "overall", "all"].append(row)
        buckets[str(row["method"]), int(row["budget"]), "coverage", str(row["coverage"])].append(row)
        buckets[str(row["method"]), int(row["budget"]), "heterogeneity", str(row["heterogeneity"])].append(row)
    out = []
    metrics = ("CollisionCount", "CriticalCount", "SeveritySum", "CVS", "F", "NewHistoricalFailureCount", "selection_seconds")
    for (method, budget, scope, value), values in buckets.items():
        row: dict[str, object] = {"method": method, "budget": budget, "scope": scope, "value": value, "units": len(values)}
        for metric in metrics: row[f"mean_{metric}"] = float(np.mean([float(item[metric]) for item in values]))
        out.append(row)
    return out


def paired_comparisons(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Paired target-unit bootstrap; never treats individual episodes as independent."""
    lookup = {(int(row["seed"]), str(row["target"]), str(row["method"]), int(row["budget"])): row for row in records if int(row["repeat"]) == 0}
    rng = np.random.default_rng(20261103); rows = []
    for reference in ("FPS-Marginal", "MeanResidual-Marginal", "TargetOnlyGP-Marginal"):
        for budget in BUDGETS:
            differences = []
            for seed, target, method, row_budget in lookup:
                if method != "CoRe-Marginal" or row_budget != budget: continue
                proposed = lookup[seed, target, method, budget]; baseline = lookup[seed, target, reference, budget]
                differences.append((seed, float(proposed["CVS"]) - float(baseline["CVS"]), float(proposed["SeveritySum"]) - float(baseline["SeveritySum"])))
            units = np.asarray([[item[1], item[2]] for item in differences]); sampled = np.empty((2000, 2))
            for draw in range(len(sampled)):
                # Resample seeds, then target units inside each drawn seed.
                values = []
                for seed in rng.choice(np.unique([item[0] for item in differences]), 3, replace=True):
                    within = units[np.asarray([item[0] == seed for item in differences])]
                    values.extend(within[rng.integers(0, len(within), len(within))])
                sampled[draw] = np.mean(values, axis=0)
            rows.append({"method": "CoRe-Marginal", "reference": reference, "budget": budget, "units": len(differences), "mean_CVS_difference": float(units[:, 0].mean()), "CVS_bootstrap_low": float(np.quantile(sampled[:, 0], .025)), "CVS_bootstrap_high": float(np.quantile(sampled[:, 0], .975)), "mean_SeveritySum_difference": float(units[:, 1].mean()), "SeveritySum_bootstrap_low": float(np.quantile(sampled[:, 1], .025)), "SeveritySum_bootstrap_high": float(np.quantile(sampled[:, 1], .975))})
    return rows


def confirm(allow_new_simulation: bool) -> None:
    status = "not executed: cache evidence did not authorize new simulation" if not allow_new_simulation else "not executed: physical confirmation requires an explicitly frozen winning method and controller manifest"
    ROOT.joinpath("confirm").mkdir(parents=True, exist_ok=True)
    _csv(ROOT / "confirm" / "records.csv", [{"status": status, "max_new_episodes": 1250}])


def _write_manifest(config: CoreMineConfig) -> None:
    payload = {
        "python": sys.version, "cuda_available": _cuda_available(),
        "frozen_config": config.as_dict(),
        "label_semantics": {"event": "ego_collision OR ego_near_miss", "collision": "ego_collision", "ttc_missing": "zero continuous base"},
        "candidate_pool": "five two-vehicle modes; passing_cutin excluded", "new_simulation": False,
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "develop", "validate", "confirm", "all"), default="all")
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--max-new-episodes", type=int, default=1250)
    parser.add_argument("--allow-new-simulation", action="store_true")
    args = parser.parse_args(); ROOT.mkdir(parents=True, exist_ok=True)
    config = CoreMineConfig()
    if args.stage in {"audit", "all"}: audit()
    if args.stage in {"develop", "all"}:
        config = develop(args.max_trials)
        tune_simple_objective()
    elif (ROOT / "frozen_config.json").exists():
        values = json.loads((ROOT / "frozen_config.json").read_text(encoding="utf-8"))["config"]; config = CoreMineConfig(values["residual_length"], values["residual_amplitude"], values["observation_noise"], values["lambda"], values["include_null"], values["compositional"])
    if args.stage in {"validate", "all"}: validate(config)
    if args.stage in {"confirm", "all"}: confirm(args.allow_new_simulation)
    _write_manifest(config)
    if args.stage == "all": write_report(ROOT)


if __name__ == "__main__": main()
