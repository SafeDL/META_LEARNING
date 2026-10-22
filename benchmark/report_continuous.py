"""Continuous results remain separate from finite-pool claims and curves."""

import json
from pathlib import Path
import numpy as np

from .collision_metrics import lineage_survival, paired_lineage_bootstrap
from .continuous_selection import METHODS
from .evaluate_continuous import OUTPUT as RUNS, validate_frozen
from .protocol import DEFAULT
from . import report_collision
from .studies import CONTINUOUS_TASKS


METRICS = ("unique_collisions", "discovery_auc", "behavior_coverage")


def main():
    validate_frozen()
    index = json.loads((RUNS / "run_index.json").read_text(encoding="utf-8"))
    rows = index["runs"]
    expected = {(sut, kind, method, seed, env) for sut, kind in CONTINUOUS_TASKS for method in METHODS
                for seed in DEFAULT.algorithm_seeds for env in DEFAULT.environment_seeds}
    observed = [(r["target_id"], r["scenario"], r["method"], r["algorithm_seed"], r["environment_seed"]) for r in rows]
    if index["status"] != "complete" or len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("All continuous algorithms and registered tasks must finish")
    output = RUNS / "report"
    output.mkdir(parents=True, exist_ok=True)
    cost_rows = []
    for row in rows:
        saved = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
        if row["queries"] != DEFAULT.budget or saved["cost"]["actual_physical_calls"] != DEFAULT.budget:
            raise ValueError("Continuous results require a full actual-call budget")
        cost_rows.append({"path": row["path"], **saved["cost"]})
    report = {"status": "complete", "scope": "continuous_extension_only", "runs": len(rows),
              "costs": cost_rows, "methods": {}, "paired_lineage_intervals": []}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        report["methods"][method] = {**{m: float(np.mean([r[m] for r in selected])) for m in METRICS},
                                       "first_collision": lineage_survival(selected, restriction=DEFAULT.budget)}
        if method != "history_response":
            report["paired_lineage_intervals"].extend(paired_lineage_bootstrap(
                rows, "history_response", method, metric) for metric in METRICS)
            contrast = paired_lineage_bootstrap(report_collision.restricted_records(rows), "history_response",
                                                 method, "restricted_first_queries")
            contrast.update(restriction=DEFAULT.budget, lower_is_earlier=True)
            report["paired_lineage_intervals"].append(contrast)
    # Each task is also plotted separately; four representative cases are not the full final matrix.
    for sut, kind in CONTINUOUS_TASKS:
        report_collision.plot_curves([r for r in rows if r["target_id"] == sut and r["scenario"] == kind],
                                     f"{sut}_{kind}", f"Continuous: {sut} / {kind}", destination=output)
        report_collision.plot_survival([r for r in rows if r["target_id"] == sut and r["scenario"] == kind],
                                        f"first_{sut}_{kind}", f"First discovery: {sut} / {kind}", destination=output)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
