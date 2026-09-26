"""Development diagnosis: oracle diversity headroom on a qualification bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from methods.core_mine.config import CoreMineConfig
from methods.core_mine.experiment import run_campaign
from methods.core_mine import sparse_sut_experiment as sparse


def _oracle_cvs(task) -> float:
    best = {}
    for index in range(task.count):
        event = bool(task.target_event[index])
        collision = bool(task.target_collision[index])
        if not event:
            continue
        cell = tuple(np.clip(np.floor(task.features[index, :2] * 4).astype(int), 0, 4))
        key = (str(task.modes[index]), *cell)
        best[key] = max(best.get(key, 0.0), 1.0 if collision else 0.5)
    return float(sum(sorted(best.values(), reverse=True)[:50]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", choices=("v6_corrected_wide", "v7_corrected_balanced"),
                        default="v7_corrected_balanced")
    args = parser.parse_args()
    sparse.configure_proposal(args.proposal)
    seed = sparse.QUALIFICATION_SEEDS[0]
    bank = sparse.load_or_build(seed)
    config = CoreMineConfig(residual_length=.30, residual_amplitude=.15, lambda_=.10)
    rows = []
    for task in sparse.tasks_from_bank(bank, seed):
        methods = {}
        for method in ("MeanResidual-Risk", "MeanResidual-Marginal", "CoRe-Marginal", "FPS-Marginal"):
            records, _ = run_campaign(task, method, config)
            row = next(item for item in records if item["budget"] == 50)
            methods[method] = {"CriticalCount": row["CriticalCount"],
                               "CollisionCount": row["CollisionCount"], "CVS": row["CVS"],
                               "NewHistoricalFailureCount": row["NewHistoricalFailureCount"]}
        risk = methods["MeanResidual-Risk"]
        oracle = _oracle_cvs(task)
        rows.append({"target": task.heterogeneity, "events_in_pool": int(task.target_event.sum()),
                     "event_modes_in_pool": int(len(set(task.modes[task.target_event]))),
                     "source_safe_candidates": int((~task.source_event.any(axis=0)).sum()),
                     "source_safe_target_failures": int(task.source_safe_target_failure.sum()),
                     "risk_events_50": risk["CriticalCount"], "risk_cvs_50": risk["CVS"],
                     "oracle_cvs_50": oracle, "cvs_headroom": oracle - float(risk["CVS"]),
                     "method_results": methods})
    payload = {"proposal": args.proposal, "qualification_seed": seed, "budget": 50,
               "oracle_uses_all_target_labels": True, "rows": rows}
    output = sparse.ROOT / "qualification" / "headroom.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
