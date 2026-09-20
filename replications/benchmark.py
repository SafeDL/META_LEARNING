"""Build and evaluate the common highway-env replication benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from diva_highway_env.data.generate_anchor_bank import (
    FUNCTIONAL_MODES,
    generate_multifunction_anchor_bank,
)
from diva_highway_env.data.response_bank import ResponseBank, build_response_bank
from replications.scenariofuzz_highway_env.scenariofuzz.corpus import (
    ScenarioSpec,
    build_default_corpus,
)
from replications.scenariofuzz_highway_env.scenariofuzz.filter import (
    load_checkpoint,
    predict_scores,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "replications" / "benchmark.yaml"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_bank(config_path: Path) -> Path:
    """Run the project simulator on the complete shared candidate domain."""
    config = _load_config(config_path)
    requested_modes = tuple(str(mode) for mode in config["modes"])
    unknown = set(requested_modes) - set(FUNCTIONAL_MODES)
    if unknown:
        raise ValueError(f"unsupported modes: {sorted(unknown)}")

    per_mode = int(config["candidates_per_mode"])
    all_anchors, all_modes = generate_multifunction_anchor_bank(
        per_mode * len(FUNCTIONAL_MODES), int(config["seed"])
    )
    mask = np.isin(all_modes, requested_modes)
    anchors, modes = all_anchors[mask], all_modes[mask]
    bank = build_response_bank(anchors, int(config["seed"]), modes=modes)
    expected = (len(config["suts"]), per_mode * len(requested_modes))
    if bank.collisions.shape != expected or tuple(bank.sut_names) != tuple(config["suts"]):
        raise RuntimeError("shared response bank does not match the benchmark contract")

    output = ROOT / config["response_bank"]
    bank.save(output)
    counts = {mode: int(np.sum(modes == mode)) for mode in requested_modes}
    manifest = {
        "environment": "project highway-env CutInEnv",
        "seed": int(config["seed"]),
        "modes": list(requested_modes),
        "candidates_per_mode": per_mode,
        "candidate_count": len(anchors),
        "suts": list(bank.sut_names),
        "physical_episode_count": int(bank.collisions.size),
        "mode_counts": counts,
        "data_scope": "complete shared candidate domain",
    }
    output.with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def _metric_rows(
    method: str,
    protocol: str,
    target: str,
    repeat: int,
    order: list[int],
    failures: np.ndarray,
    budgets: list[int],
    target_feedback: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total_failures = int(failures.sum())
    for budget in budgets:
        selected = np.asarray(order[:budget], dtype=int)
        count = int(failures[selected].sum())
        values = {
            "collision_count": float(count),
            "precision": float(count / len(selected)),
            "recall": float(count / total_failures) if total_failures else float("nan"),
        }
        for metric, value in values.items():
            rows.append({
                "task": "failure_discovery",
                "protocol": protocol,
                "method": method,
                "target_sut": target,
                "repeat": repeat,
                "budget": budget,
                "metric": metric,
                "value": value,
                "candidate_count": len(failures),
                "target_feedback": target_feedback,
            })
    return rows


def _adate_rows(
    bank: ResponseBank, result_dir: Path, budgets: list[int]
) -> list[dict[str, object]]:
    trace = _read_csv(result_dir / "target_query_trace.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        if row["response"] == "collision":
            grouped[(row["target_sut"], row["method_variant"])].append(row)
    records: list[dict[str, object]] = []
    for (target, method), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        order = [int(row["anchor_index"]) for row in ordered]
        failures = bank.collisions[bank.index_of(target)]
        feedback = {
            "Uniform-Mixture-H": "none",
            "AdaTE-Mixture-H-staticK4": "first four target outcomes",
            "AdaTE-Mixture-H-sequential": "previous target outcomes",
        }[method]
        protocol = "shared_pool_static" if feedback == "none" else "shared_pool_adaptive"
        records.extend(_metric_rows(
            method,
            protocol,
            target,
            0,
            order,
            failures,
            budgets,
            feedback,
        ))
    return records


def _detour_rows(result_dir: Path, candidate_count: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in _read_csv(result_dir / "metrics.csv"):
        for metric, source in (
            ("collision_count", "failure_count"),
            ("precision", "failure_ratio"),
            ("recall", "recall"),
        ):
            protocol = (
                "shared_pool_within_sut_history"
                if "within-SUT" in row["method"]
                else "shared_pool_static"
            )
            records.append({
                "task": "failure_discovery",
                "protocol": protocol,
                "method": row["method"],
                "target_sut": row["target_sut"],
                "repeat": int(row["seed"]),
                "budget": int(row["checkpoint"]),
                "metric": metric,
                "value": float(row[source]),
                "candidate_count": candidate_count,
                "target_feedback": "none",
            })
    return records


def _scenariofuzz_rows(
    bank: ResponseBank,
    result_dir: Path,
    output: Path,
    budgets: list[int],
) -> list[dict[str, object]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ranking_rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    specs = [
        ScenarioSpec.create(gap, speed, str(mode))
        for (gap, speed), mode in zip(bank.anchors, bank.modes, strict=True)
    ]
    for target in bank.sut_names:
        model_dir = result_dir / "models" / target
        config = _load_config(model_dir / "config.resolved.yaml")
        model, _ = load_checkpoint(model_dir / "source_only_sem.pt", device)
        scores = predict_scores(model, build_default_corpus(config)[0], specs, device)
        order = np.argsort(scores)[::-1].tolist()
        failures = bank.collisions[bank.index_of(target)]
        records.extend(_metric_rows(
            "ScenarioFuzz-SEM-Pool-H",
            "shared_pool_static",
            target,
            0,
            order,
            failures,
            budgets,
            "none",
        ))
        for rank, index in enumerate(order, start=1):
            ranking_rows.append({
                "target_sut": target,
                "rank": rank,
                "scenario_index": index,
                "scenario_id": specs[index].scenario_id,
                "mode": specs[index].mode,
                "predicted_score": float(scores[index]),
                "collision_offline": int(failures[index]),
            })
    _write_csv(output / "scenariofuzz_shared_ranking.csv", ranking_rows)
    return records


def _fst_rows(result_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in _read_csv(result_dir / "metrics_summary.csv"):
        for metric in ("mean_absolute_error", "relative_absolute_error", "signed_bias"):
            records.append({
                "task": "performance_estimation",
                "protocol": "shared_pool_estimation",
                "method": row["method"],
                "target_sut": row["target_sut"],
                "repeat": "aggregate",
                "budget": int(row["budget"]),
                "metric": metric,
                "value": float(row[metric]),
                "candidate_count": "shared",
                "target_feedback": "selected target responses",
            })
    return records


def _summaries(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in records:
        key = (row["task"], row["protocol"], row["method"], row["budget"], row["metric"])
        grouped[key].append(float(row["value"]))
    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        task, protocol, method, budget, metric = key
        array = np.asarray(values, dtype=float)
        rows.append({
            "task": task,
            "protocol": protocol,
            "method": method,
            "budget": budget,
            "metric": metric,
            "mean": float(np.nanmean(array)),
            "standard_deviation": float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0,
            "records": len(array),
        })
    return rows


def _plot_discovery(output: Path, summary: list[dict[str, object]], budget: int) -> None:
    rows = [
        row for row in summary
        if row["task"] == "failure_discovery"
        and row["protocol"] in {"shared_pool_static", "shared_pool_adaptive"}
        and row["metric"] == "recall"
        and int(row["budget"]) == budget
    ]
    rows.sort(key=lambda row: float(row["mean"]), reverse=True)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    names = [str(row["method"]) for row in rows]
    means = [float(row["mean"]) for row in rows]
    errors = [float(row["standard_deviation"]) for row in rows]
    axis.barh(np.arange(len(rows)), means, xerr=errors, color="#3977a8", alpha=0.88)
    axis.set_yticks(np.arange(len(rows)), names)
    axis.invert_yaxis()
    axis.set(xlabel=f"Collision recall at budget {budget}", xlim=(0, 1),
             title="Shared highway-env failure-discovery benchmark")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "failure_discovery_comparison.png", dpi=180)
    plt.close(figure)


def _plot_estimation(output: Path, summary: list[dict[str, object]]) -> None:
    rows = [
        row for row in summary
        if row["task"] == "performance_estimation"
        and row["metric"] == "mean_absolute_error"
    ]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for method in sorted({str(row["method"]) for row in rows}):
        subset = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["budget"]),
        )
        axis.plot(
            [int(row["budget"]) for row in subset],
            [float(row["mean"]) for row in subset],
            marker="o",
            label=method,
        )
    axis.set(xlabel="Target execution budget", ylabel="Mean absolute error",
             title="Shared highway-env performance-estimation benchmark")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "performance_estimation_comparison.png", dpi=180)
    plt.close(figure)


def evaluate(config_path: Path) -> Path:
    config = _load_config(config_path)
    results_root = ROOT / config["results_root"]
    output = results_root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    bank = ResponseBank.load(ROOT / config["response_bank"])
    budgets = [int(value) for value in config["budgets"]]

    records: list[dict[str, object]] = []
    records.extend(_adate_rows(bank, results_root / "adate" / "shared_pool", budgets))
    records.extend(_detour_rows(results_root / "detour", len(bank.anchors)))
    records.extend(_scenariofuzz_rows(
        bank, results_root / "scenariofuzz", output, budgets
    ))
    records.extend(_fst_rows(results_root / "fst"))
    _write_csv(output / "evaluation_records.csv", records)
    summary = _summaries(records)
    _write_csv(output / "method_summary.csv", summary)

    coverage = [
        {
            "method": "AdaTE-Mixture-H",
            "task": "failure_discovery",
            "shared_bank": True,
            "target_feedback": "adaptive",
            "paper_specific_results": "adate/dense and adate/rare_event",
        },
        {
            "method": "DETOUR-static",
            "task": "failure_discovery",
            "shared_bank": True,
            "target_feedback": "none",
            "paper_specific_results": "detour",
        },
        {
            "method": "ScenarioFuzz-SEM-Pool-H",
            "task": "failure_discovery",
            "shared_bank": True,
            "target_feedback": "none",
            "paper_specific_results": "scenariofuzz online campaigns",
        },
        {
            "method": "FST-Similarity-H",
            "task": "performance_estimation",
            "shared_bank": True,
            "target_feedback": "selected responses",
            "paper_specific_results": "fst",
        },
    ]
    _write_csv(output / "coverage_matrix.csv", coverage)
    _plot_discovery(output, summary, max(budgets))
    _plot_estimation(output, summary)

    mode_counts = {
        mode: int(np.sum(np.asarray(bank.modes) == mode)) for mode in config["modes"]
    }
    checks = {
        "shared_bank_has_all_suts": tuple(bank.sut_names) == tuple(config["suts"]),
        "shared_bank_is_balanced": set(mode_counts.values())
        == {int(config["candidates_per_mode"])},
        "shared_bank_has_all_responses": bank.collisions.shape == (
            len(config["suts"]),
            int(config["candidates_per_mode"]) * len(config["modes"]),
        ),
        "all_records_are_finite": all(np.isfinite(float(row["value"])) for row in records),
        "both_tasks_are_present": {row["task"] for row in records}
        == {"failure_discovery", "performance_estimation"},
        "all_methods_are_present": {
            "AdaTE-Mixture-H-sequential",
            "DETOUR-static",
            "ScenarioFuzz-SEM-Pool-H",
            "FST-Similarity-H",
        } <= {row["method"] for row in records},
    }
    validation = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "shared_candidate_count": len(bank.anchors),
        "shared_physical_episode_count": int(bank.collisions.size),
        "evaluation_record_count": len(records),
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if validation["status"] != "passed":
        raise RuntimeError(f"benchmark validation failed: {checks}")

    discovery = [
        row for row in summary
        if row["task"] == "failure_discovery"
        and row["protocol"] in {"shared_pool_static", "shared_pool_adaptive"}
        and row["metric"] == "recall"
        and int(row["budget"]) == max(budgets)
    ]
    estimation = [
        row for row in summary
        if row["task"] == "performance_estimation"
        and row["metric"] == "mean_absolute_error"
        and int(row["budget"]) == max(budgets)
    ]
    lines = [
        "# Highway-env replication benchmark",
        "",
        f"The shared domain contains {len(bank.anchors)} scenarios "
        f"({', '.join(f'{name}: {count}' for name, count in mode_counts.items())}) "
        f"and {bank.collisions.size} actual simulator responses across six SUTs.",
        "",
        "Failure-discovery methods are compared only under the shared candidate pool. "
        "AdaTE includes static and adaptive variants; DETOUR LOSO and the ScenarioFuzz "
        "fixed-pool control are static. DETOUR's within-SUT D1 protocol is retained in "
        "the records but excluded from the primary table because it observes target-SUT "
        "history. "
        "FST remains in the separate performance-estimation task, so its MAE is not mixed "
        "with discovery recall.",
        "",
        f"## Collision recall at B={max(budgets)}",
        "",
        "| Method | Protocol | Mean recall | SD |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in sorted(discovery, key=lambda item: float(item["mean"]), reverse=True):
        lines.append(
            f"| {row['method']} | {row['protocol']} | {float(row['mean']):.3f} "
            f"| {float(row['standard_deviation']):.3f} |"
        )
    lines.extend([
        "",
        f"## Collision-rate estimation MAE at B={max(budgets)}",
        "",
        "| Method | Mean MAE | SD |",
        "| --- | ---: | ---: |",
    ])
    for row in sorted(estimation, key=lambda item: float(item["mean"])):
        lines.append(
            f"| {row['method']} | {float(row['mean']):.4f} "
            f"| {float(row['standard_deviation']):.4f} |"
        )
    lines.extend([
        "",
        "The shared benchmark measures project utility on a common highway-env domain. "
        "Paper-specific conclusions, ablations, and deviations remain in each method's "
        "own final report.",
        "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "evaluate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    destination = (
        build_bank(arguments.config)
        if arguments.action == "build"
        else evaluate(arguments.config)
    )
    print(destination)


if __name__ == "__main__":
    main()
