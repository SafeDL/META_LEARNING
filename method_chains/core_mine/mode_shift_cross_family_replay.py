"""Retrospective B=50 replay of unchanged ModeShift across IDM/FVDM revisions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from method_chains.core_mine import fvdm_revision_validation as fvdm
from method_chains.core_mine import idm_revision_validation as idm
from method_chains.core_mine.acquisition import choose
from method_chains.core_mine.config import SUPPORT_BUDGET
from method_chains.core_mine.metrics import record_metrics
from method_chains.core_mine.oracle import CacheOracle
from method_chains.core_mine.simple_residual_ablation import corrected_scores


ROOT = Path("results/method_chains/core_mine/studies/mode_shift_cross_family_replay")
METHODS = ("ModeShift-Risk", "HistoryMargin-Static", "ModeQuantile-Static")
FAMILIES = {"IDM": idm, "FVDM": fvdm}


def _quantile_scores(task) -> np.ndarray:
    source = task.source_y.mean(axis=0)
    scores = np.empty(task.count, dtype=float)
    for mode in np.unique(task.modes):
        indices = np.flatnonzero(task.modes == mode)
        ranks = rankdata(source[indices], method="average")
        scores[indices] = (ranks - 1) / max(len(indices) - 1, 1)
    return scores


def replay(task, method: str) -> dict:
    if method not in METHODS or task.source_event.any():
        raise ValueError("unsupported method or unsafe historical candidate")
    oracle = CacheOracle(task)
    responses: list[float] = []
    static = task.source_y.mean(axis=0)
    quantile = _quantile_scores(task)
    while len(oracle.revealed) < 50:
        if method == "ModeShift-Risk":
            scores = corrected_scores(task, oracle.revealed, responses,
                                      "HistoryMargin-ModeShift")
        elif method == "ModeQuantile-Static":
            scores = quantile
        else:
            scores = static
        index = choose(scores, oracle.revealed, task.modes, SUPPORT_BUDGET)
        response = oracle.reveal(index)
        responses.append(response.y)
    record = record_metrics(task, oracle.revealed, 50)
    if (len(oracle.revealed) != 50 or len(set(oracle.revealed)) != 50
            or len(responses) != 50):
        raise RuntimeError("invalid selective B=50 oracle ledger")
    return {"method": method,
            "new_failures": int(record["NewHistoricalFailureCount"]),
            "ego_collisions": int(record["CollisionCount"]),
            "failure_modes": int(len(set(task.modes[np.asarray(oracle.revealed)][
                task.target_event[np.asarray(oracle.revealed)]]))),
            "cvs": float(record["CVS"]),
            "queried_indices": record["queried_indices"]}


def main() -> None:
    records = []
    for family, module in FAMILIES.items():
        for seed in module.VALIDATION_SEEDS:
            gate = json.loads((module.ROOT / str(seed) / "gate.json").read_text(
                encoding="utf-8"))
            if not gate["passed"]:
                raise RuntimeError(f"source-only gate failed {family}/{seed}")
            for target in module.TARGETS:
                task = module._task(seed, target)
                for method in METHODS:
                    records.append({"family": family, "seed": seed,
                                    "target": target,
                                    **replay(task, method)})
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = ("new_failures", "ego_collisions", "failure_modes", "cvs")
    summary = {family: {method: {metric: float(np.mean([
        row[metric] for row in records if row["family"] == family
        and row["method"] == method])) for metric in metrics}
        for method in METHODS} for family in FAMILIES}
    by_target = {family: {target: {method: {metric: float(np.mean([
        row[metric] for row in records if row["family"] == family
        and row["target"] == target and row["method"] == method]))
        for metric in metrics} for method in METHODS}
        for target in module.TARGETS}
        for family, module in FAMILIES.items()}
    keyed = {(row["family"], row["seed"], row["target"], row["method"]): row
             for row in records}
    paired = {family: {baseline: {metric: [[
        keyed[(family, seed, target, "ModeShift-Risk")][metric]
        - keyed[(family, seed, target, baseline)][metric]
        for target in module.TARGETS] for seed in module.VALIDATION_SEEDS]
        for metric in metrics} for baseline in METHODS[1:]}
        for family, module in FAMILIES.items()}
    output = {"schema": "mode_shift_cross_family_replay_v1",
              "budget": 50, "physical_target_episodes_new": 0,
              "retrospective_target_banks_inspected_before_protocol": True,
              "unit_count": 18, "record_count": len(records),
              "methods": METHODS, "summary": summary,
              "by_target": by_target, "paired": paired}
    (ROOT / "analysis50.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "by_target": by_target}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
