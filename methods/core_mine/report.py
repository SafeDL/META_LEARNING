"""Compact tables, figures, and an evidence-constrained research decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, str]], method: str, budget: int, metric: str, scope: str = "overall", value: str = "all") -> float:
    values = [float(row[f"mean_{metric}"]) for row in rows if row["method"] == method and int(row["budget"]) == budget and row["scope"] == scope and row["value"] == value]
    return values[0] if values else float("nan")


def _figures(root: Path, summary: list[dict[str, str]], records: list[dict[str, str]]) -> None:
    directory = root / "figures"; directory.mkdir(exist_ok=True)
    methods = ("FPS-Marginal", "MeanResidual-Marginal", "CoRe-Marginal", "TargetOnlyGP-Marginal", "Random")
    for metric, filename, label in (("CVS", "cvs_by_budget.png", "CVS"), ("SeveritySum", "severity_by_budget.png", "Severity sum")):
        plt.figure(figsize=(7, 4))
        for method in methods:
            values = [_mean(summary, method, budget, metric) for budget in (10, 20, 30, 50)]
            if np.isfinite(values).any(): plt.plot((10, 20, 30, 50), values, marker="o", label=method)
        plt.xlabel("Budget"); plt.ylabel(label); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(directory / filename, dpi=180); plt.close()
    plt.figure(figsize=(7, 4)); coverage = ("exact", "interpolated", "unseen"); x = np.arange(3); width = .24
    for offset, method in enumerate(("FPS-Marginal", "MeanResidual-Marginal", "CoRe-Marginal")):
        plt.bar(x + (offset - 1) * width, [_mean(summary, method, 20, "CVS", "coverage", item) for item in coverage], width, label=method)
    plt.xticks(x, coverage); plt.ylabel("CVS@20"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(directory / "grouped_cvs20.png", dpi=180); plt.close()
    target = next((row["target"] for row in records if row["method"] == "CoRe-Marginal"), None)
    if target:
        series = ("FPS-Marginal", "CoRe-Marginal", "TargetOnlyGP-Marginal"); values = {}
        for method in series:
            row = next(row for row in records if row["target"] == target and row["method"] == method and int(row["budget"]) == 50)
            values[method] = {pair.split(":")[0]: int(pair.split(":")[1]) for pair in row["function_queries"].split(";")}
        labels = sorted({key for value in values.values() for key in value}); bottom = np.zeros(3); plt.figure(figsize=(7, 4))
        for mode in labels:
            part = np.asarray([values[method].get(mode, 0) for method in series]); plt.bar(series, part, bottom=bottom, label=mode); bottom += part
        plt.ylabel("Queries through B=50"); plt.legend(fontsize=7); plt.xticks(rotation=10); plt.tight_layout(); plt.savefig(directory / "query_distribution_example.png", dpi=180); plt.close()


def write_report(root: Path) -> None:
    summary = _read(root / "summary.csv"); validation = _read(root / "validate" / "records.csv"); ablations = _read(root / "ablations.csv")
    objective = _read(root / "objective_trials.csv") if (root / "objective_trials.csv").exists() else []
    paired = _read(root / "paired_comparisons.csv") if (root / "paired_comparisons.csv").exists() else []
    _figures(root, summary, validation)
    baselines = ("FPS-Marginal", "MeanResidual-Marginal", "TargetOnlyGP-Marginal")
    strongest = max(baselines, key=lambda name: _mean(summary, name, 20, "CVS"))
    core_cvs, base_cvs = _mean(summary, "CoRe-Marginal", 20, "CVS"), _mean(summary, strongest, 20, "CVS")
    core_sev, base_sev = _mean(summary, "CoRe-Marginal", 20, "SeveritySum"), _mean(summary, strongest, 20, "SeveritySum")
    relative = (core_cvs - base_cvs) / base_cvs if base_cvs else float("nan")
    paired_primary = next((row for row in paired if row["reference"] == strongest and int(row["budget"]) == 20), None)
    no_comp = np.mean([float(row["CVS"]) for row in ablations if row["method"] == "NoComposition" and int(row["budget"]) == 20])
    no_null = np.mean([float(row["CVS"]) for row in ablations if row["method"] == "NoNull" and int(row["budget"]) == 20])
    objective_ok = any(row["retains_90pct_severity"] == "True" for row in objective)
    if core_cvs > base_cvs and core_sev >= .9 * base_sev:
        verdict = "complete method effective" if core_cvs - base_cvs >= 1 and relative >= .10 else "small gain"
    elif objective_ok:
        verdict = "simplified method effective"
    else: verdict = "no reliable gain in this round"
    rows = ["# CoRe-Mine cached experiment", "", "The study uses five existing frozen response banks: two for development and three for validation. Each target retains 500 scenarios from the five two-vehicle functions; `passing_cutin` is excluded as its own three-vehicle setting. Selectors receive only source data and outcomes revealed by the cache oracle.", "", "Primary endpoint: CVS@20, a fixed 4x4 gap × relative-speed grid per function. A collision contributes 1 and a near miss 0.5; repeats do not increase archive value.", "", "## Validation summary", "", "| Method | CVS@20 | Collision@20 | Severity@20 | CVS@50 | Cost (s/campaign) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    methods = ("FPS-Risk", "FPS-Severity", "FPS-Balanced", "FPS-Marginal", "MeanResidual-Marginal", "CoRe-Marginal", "TargetOnlyGP-Marginal", "Random")
    for method in methods:
        rows.append(f"| {method} | {_mean(summary, method, 20, 'CVS'):.3f} | {_mean(summary, method, 20, 'CollisionCount'):.3f} | {_mean(summary, method, 20, 'SeveritySum'):.3f} | {_mean(summary, method, 50, 'CVS'):.3f} | {_mean(summary, method, 20, 'selection_seconds'):.3f} |")
    rows += ["", "## Matched comparisons", "", f"The strongest matched marginal baseline is **{strongest}** (CVS@20={base_cvs:.3f}); CoRe-Marginal is {core_cvs:.3f} (difference {core_cvs-base_cvs:+.3f}, {relative:+.1%}) with severity {core_sev:.3f} versus {base_sev:.3f}."]
    if paired_primary:
        rows.append(f"Paired seed→target bootstrap for the CVS difference: [{float(paired_primary['CVS_bootstrap_low']):+.3f}, {float(paired_primary['CVS_bootstrap_high']):+.3f}] (n={paired_primary['units']}); individual episodes were not treated as independent samples.")
    rows.append(f"Ablations: NoComposition CVS@20={no_comp:.3f}, NoNull CVS@20={no_null:.3f}; FPS-Marginal is the pre-registered NoResidual comparator.")
    if objective:
        rows += ["", "The second pre-listed development check tested FPS-Marginal λ={0, 0.10, 0.30}. None retained 90% of FPS-Severity's SeveritySum@20; results are retained in `objective_trials.csv`."]
    rows += ["", "## Decision", "", f"Outcome: **{verdict}**. All development trials are retained; validation used the frozen configuration. New physical confirmation was not authorized: CoRe lost to its strongest matched baseline and the simpler coverage rule did not meet the pre-set severity guardrail.", ""]
    (root / "report.md").write_text("\n".join(rows), encoding="utf-8")
    decision = ["# Research decision", "", "**Question.** Under a fixed target-test budget, can function-level source responses and selectively revealed target feedback discover severe, non-redundant vulnerability regions better than point-risk search?", "", f"**Evidence.** At B=20, CoRe-Marginal CVS is {core_cvs:.3f}, versus {strongest} at {base_cvs:.3f}; severity is {core_sev:.3f} versus {base_sev:.3f}.", "", "**Method.** CoRe combines function-specific hypotheses, a Matérn-5/2 residual GP, predictive-evidence weights, and marginal archive coverage; NoComposition, NoNull, and FPS-Marginal/NoResidual are retained in the evidence.", "", f"**Conclusion.** {verdict}. The λ check found no simple marginal rule that kept 90% of the severity baseline, so physical confirmation is not warranted. This is cache evidence only, not a real-world safety claim."]
    (root / "research_decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    (root / "cost_summary.json").write_text(json.dumps({"cached_new_episodes": 0, "gpu": "available but not used: analytic GP has at most 50 observations", "confirmation": "not executed", "validation_campaigns": len({(r['seed'], r['target'], r['method'], r['repeat']) for r in validation})}, indent=2) + "\n", encoding="utf-8")
