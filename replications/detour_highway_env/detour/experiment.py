"""Reproducible paper-style static DETOUR-Scenario-H runs.

Selection receives source-SUT histories and candidate inputs only; complete
target outcomes are accessed solely for offline evaluation and replay.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from highway_env_benchmark.data.response_bank import ResponseBank
from replications.detour_highway_env.detour.features import encode_scenarios
from replications.detour_highway_env.detour.retrieve import Retriever
from replications.detour_highway_env.detour.selector import prioritize, select
from replications.detour_highway_env.detour.tree import DetourTree, build_tree
from replications.detour_highway_env.detour.visualize import (plot_branch_statistics,
                                                              plot_discovery_curve,
                                                              plot_road_features, plot_selection,
                                                              plot_sensitivity, plot_tree,
                                                              write_replay_gif)
from replications.detour_highway_env.detour.history import (
    scenario_specs,
    source_history,
    within_sut_split,
)
from replications.detour_highway_env.detour.metrics import discovery_metrics


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tree_for(bank: ResponseBank, target_sut: str,
              failure_oracle: str) -> tuple[DetourTree, tuple, tuple, np.ndarray]:
    candidates = scenario_specs(bank)
    history = source_history(bank, target_sut, failure_oracle)
    return build_tree(encode_scenarios(tuple(item.scenario for item in history)),
                      encode_scenarios(candidates),
                      np.asarray([item.failed for item in history
                                  ])), candidates, history, encode_scenarios(candidates)


def _global_nearest_order_fast(history, candidate_features: np.ndarray, count: int,
                               seed: int) -> list[int]:
    """Global nearest-failure baseline; this deliberately omits the hierarchy."""
    failures = np.asarray([index for index, item in enumerate(history) if item.failed], dtype=int)
    remaining, rng, order = set(range(len(candidate_features))), np.random.default_rng(seed), []
    source = encode_scenarios(tuple(item.scenario for item in history))
    while remaining and len(order) < count:
        selected = int(rng.choice(sorted(remaining))) if not len(failures) else min(
            (float(np.min(np.linalg.norm(source[failures] - candidate_features[index], axis=1))),
             index) for index in remaining)[1]
        remaining.remove(selected)
        order.append(selected)
    return order


def _random_order(candidate_count: int, count: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).permutation(candidate_count)[:count].astype(int).tolist()


def _record_order(rows,
                  metrics_rows,
                  method,
                  target_sut,
                  seed,
                  order,
                  collisions,
                  critical,
                  checkpoints,
                  scenarios=None,
                  reasons=None,
                  visible_counts=None,
                  mode="static"):
    cumulative = 0
    for step, index in enumerate(order, start=1):
        cumulative += int(collisions[index])
        rows.append({
            "method":
            method,
            "method_variant":
            method,
            "task_kind":
            "failure_discovery",
            "mode":
            mode,
            "target_sut":
            target_sut,
            "seed":
            seed,
            "step":
            step,
            "scenario_index":
            index,
            "scenario_id":
            "" if scenarios is None else scenarios[index].scenario_id,
            "selected_reason":
            "baseline" if reasons is None else reasons[step - 1],
            "visible_target_count_before":
            0 if visible_counts is None else visible_counts[step - 1],
            "collision_offline":
            bool(collisions[index]),
            "critical_offline":
            bool(critical[index]),
            "cumulative_failures":
            cumulative
        })
    for checkpoint in checkpoints:
        prefix = order[:min(checkpoint, len(order))]
        result = discovery_metrics(prefix, collisions, critical)
        metrics_rows.append({
            "method": method,
            "target_sut": target_sut,
            "seed": seed,
            "checkpoint": checkpoint,
            **{key: value
               for key, value in result.items() if not key.endswith("curve")}
        })


def run(config: dict, bank_path: Path, output: Path) -> Path:
    """Run paper-style static DETOUR, control baselines, and selection sensitivity."""
    bank, output = ResponseBank.load(bank_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    budget, seeds = min(int(config["target_budget"]),
                        len(bank.anchors)), [int(value) for value in config["random_seeds"]]
    manifest = {
        "method":
        "DETOUR-Scenario-H-static",
        "paper_semantics":
        config["semantics"],
        "task_kind":
        config["task_kind"],
        "bank":
        str(bank_path),
        "within_sut_d1_split": {
            "rule": "sha256(scenario_id) first-8-hex modulo 100 < history_percent",
            "history_percent": config["within_sut_d1_history_percent"]
        },
        "config":
        config,
        "oracle":
        config["failure_oracle"],
        "seeds":
        seeds,
        "target_truth_visibility":
        "offline evaluation only",
        "history_candidate_overlap":
        "same scenario coordinates retain distinct history execution identities"
    }
    try:
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                         text=True).strip()
    except subprocess.SubprocessError:
        manifest["git_commit"] = "unavailable"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    selected_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    stop_rows: list[dict[str, object]] = []
    history_cost_rows: list[dict[str, object]] = []
    first_tree = first_trace = first_specs = first_history = None
    for target_index, target_sut in enumerate(bank.sut_names):
        template, specs, history, candidate_features = _tree_for(bank, target_sut,
                                                                 config["failure_oracle"])
        collisions = bank.collisions[target_index]
        critical = collisions | bank.near_misses[target_index]
        d1_history, d1_candidates, d1_indices = within_sut_split(
            bank, target_sut, config["failure_oracle"],
            int(config["within_sut_d1_history_percent"]))
        history_cost_rows.extend((
            {
                "protocol": "D1-within-SUT-static",
                "target_sut": target_sut,
                "history_executions": len(d1_history),
                "candidate_count": len(d1_candidates),
                "history_label_source": "frozen same-SUT history split"
            },
            {
                "protocol": "D2-LOSO",
                "target_sut": target_sut,
                "history_executions": len(history),
                "candidate_count": len(specs),
                "history_label_source": "other SUTs only"
            },
        ))
        d1_template = build_tree(encode_scenarios(tuple(item.scenario for item in d1_history)),
                                 encode_scenarios(d1_candidates),
                                 np.asarray([item.failed for item in d1_history]))
        d1_features = encode_scenarios(d1_candidates)
        d1_collisions, d1_critical = collisions[d1_indices], critical[d1_indices]
        for seed in seeds:
            tree = copy.copy(template)
            tree.active_candidates = set(range(tree.candidate_count))
            order, traces, decisions = prioritize(Retriever(tree, seed), budget)
            _record_order(selected_rows,
                          metric_rows,
                          "DETOUR-static",
                          target_sut,
                          seed,
                          order,
                          collisions,
                          critical,
                          config["checkpoints"],
                          scenarios=specs)
            for step, trace in enumerate(traces, start=1):
                trace_rows.append({
                    "method": "DETOUR-static",
                    "target_sut": target_sut,
                    "seed": seed,
                    "step": step,
                    "visible_target_count_before": 0,
                    **trace.as_dict()
                })
            stop_rows.extend({
                "method": "DETOUR-static",
                "target_sut": target_sut,
                "seed": seed,
                **asdict(decision)
            } for decision in decisions)
            _record_order(selected_rows,
                          metric_rows,
                          "Random",
                          target_sut,
                          seed,
                          _random_order(len(specs), budget, seed),
                          collisions,
                          critical,
                          config["checkpoints"],
                          scenarios=specs)
            _record_order(selected_rows,
                          metric_rows,
                          "Nearest-Failure-global",
                          target_sut,
                          seed,
                          _global_nearest_order_fast(history, candidate_features, budget, seed),
                          collisions,
                          critical,
                          config["checkpoints"],
                          scenarios=specs)
            d1_tree = copy.copy(d1_template)
            d1_tree.active_candidates = set(range(d1_tree.candidate_count))
            d1_order, d1_traces, d1_decisions = prioritize(Retriever(d1_tree, seed),
                                                           min(budget, len(d1_candidates)))
            _record_order(selected_rows,
                          metric_rows,
                          "DETOUR-within-SUT-static",
                          target_sut,
                          seed,
                          d1_order,
                          d1_collisions,
                          d1_critical,
                          config["checkpoints"],
                          scenarios=d1_candidates)
            _record_order(selected_rows,
                          metric_rows,
                          "Random-within-SUT",
                          target_sut,
                          seed,
                          _random_order(len(d1_candidates), min(budget, len(d1_candidates)), seed),
                          d1_collisions,
                          d1_critical,
                          config["checkpoints"],
                          scenarios=d1_candidates)
            _record_order(selected_rows,
                          metric_rows,
                          "Nearest-Failure-global-within-SUT",
                          target_sut,
                          seed,
                          _global_nearest_order_fast(d1_history, d1_features,
                                                     min(budget, len(d1_candidates)), seed),
                          d1_collisions,
                          d1_critical,
                          config["checkpoints"],
                          scenarios=d1_candidates)
            for step, trace in enumerate(d1_traces, start=1):
                trace_rows.append({
                    "method": "DETOUR-within-SUT-static",
                    "target_sut": target_sut,
                    "seed": seed,
                    "step": step,
                    "visible_target_count_before": 0,
                    **trace.as_dict()
                })
            stop_rows.extend({
                "method": "DETOUR-within-SUT-static",
                "target_sut": target_sut,
                "seed": seed,
                **asdict(decision)
            } for decision in d1_decisions)
            if first_tree is None:
                first_tree, first_trace, first_specs, first_history = tree, traces[
                    0], specs, history
    _write_csv(output / "tree_nodes.csv", first_tree.node_rows())
    _write_csv(output / "history_costs.csv", history_cost_rows)
    _write_csv(output / "selected_order.csv", selected_rows)
    _write_csv(output / "metrics.csv", metric_rows)
    _write_csv(output / "stop_decisions.csv", stop_rows)
    with (output / "retrieval_trace.jsonl").open("w", encoding="utf-8") as stream:
        for row in trace_rows:
            stream.write(json.dumps(row) + "\n")
    sensitivity_rows = []
    grid = config["selection_grid"]
    for target_index, target_sut in enumerate(bank.sut_names):
        template, _specs, _history, _features = _tree_for(bank, target_sut,
                                                          config["failure_oracle"])
        target_collisions = bank.collisions[target_index]
        target_critical = target_collisions | bank.near_misses[target_index]
        for minimum in grid["min_ratios"]:
            for maximum in grid["max_ratios"]:
                min_count = int(np.ceil(float(minimum) * len(_specs)))
                max_count = min(int(np.floor(float(maximum) * len(_specs))), budget)
                for m_neighbors in grid["m_neighbors"]:
                    for w_streak in grid["w_streak"]:
                        for seed in seeds:
                            tree = copy.copy(template)
                            tree.active_candidates = set(range(tree.candidate_count))
                            order, _traces, decisions = select(Retriever(tree, seed), min_count,
                                                               max_count, int(m_neighbors),
                                                               int(w_streak))
                            sensitivity_rows.append({
                                "target_sut": target_sut,
                                "seed": seed,
                                "min_ratio": minimum,
                                "max_ratio": maximum,
                                "min_count": min_count,
                                "max_count": max_count,
                                "m_neighbors": m_neighbors,
                                "w_streak": w_streak,
                                **{
                                    key: value
                                    for key, value in discovery_metrics(
                                        order, target_collisions, target_critical).items() if not key.endswith("curve")
                                }, "stop_reason": decisions[-1].reason
                            })
                            stop_rows.extend({
                                "method": "DETOUR-selection-sensitivity",
                                "target_sut": target_sut,
                                "seed": seed,
                                "min_ratio": minimum,
                                "max_ratio": maximum,
                                "m_neighbors": m_neighbors,
                                "w_streak": w_streak,
                                **asdict(decision)
                            } for decision in decisions)
    _write_csv(output / "stop_decisions.csv", stop_rows)
    _write_csv(output / "sensitivity.csv", sensitivity_rows)
    plot_road_features(output / "detour_01_road_features.png")
    plot_tree(output / "detour_02_dendrogram.png", first_tree, first_trace)
    plot_selection(output / "detour_03_selection_static.png", first_specs, [
        row["selected_index"] for row in trace_rows if row.get("method") == "DETOUR-static"
        and row["target_sut"] == bank.sut_names[0] and row["seed"] == seeds[0]
    ], tuple(item.scenario for item in first_history if item.failed))
    plot_discovery_curve(output / "detour_04_discovery_curve.png", output / "selected_order.csv")
    plot_sensitivity(output / "detour_05_stop_sensitivity.png", output / "sensitivity.csv")
    plot_branch_statistics(output / "detour_06_branch_statistics.png",
                           output / "retrieval_trace.jsonl")
    first_failure = next(index for index, value in enumerate(bank.collisions[0]) if value)
    first_pass = next(index for index, value in enumerate(bank.collisions[0]) if not value)
    replay = {
        "failed":
        write_replay_gif(output / f"detour_case_{first_failure}_failure.gif",
                         bank.sut_names[0],
                         first_specs[first_failure],
                         20260912 + first_failure,
                         node_id="offline-selected-candidate"),
        "passed":
        write_replay_gif(output / f"detour_case_{first_pass}_pass.gif",
                         bank.sut_names[0],
                         first_specs[first_pass],
                         20260912 + first_pass,
                         node_id="offline-selected-candidate")
    }
    (output / "replay_manifest.json").write_text(json.dumps(replay, indent=2), encoding="utf-8")
    aggregate = {
        method: float(
            np.mean([
                float(row["failure_ratio"]) for row in metric_rows
                if row["method"] == method and int(row["checkpoint"]) == budget
            ]))
        for method in {row["method"]
                       for row in metric_rows}
    }
    safe_stops = sum(row["stop_reason"] == "safe_neighbor_streak" for row in sensitivity_rows)
    (output / "report.md").write_text(
        f"# DETOUR-Scenario-H static replication\n\n- implementation_validated: see the DETOUR test suite.\n- mechanism_observed: D1 frozen within-SUT split, D2 LOSO hierarchy traces, per-node count snapshots, branch sampling, road compression, and the full 36-cell stop grid were generated.\n- project_utility_observed: mean collision failure ratio at B={budget}: {aggregate}.\n- history execution costs: `history_costs.csv` records D1/D2 history and candidate counts by target.\n- safe-neighbor early stops in 36-cell grid: {safe_stops}/{len(sensitivity_rows)}.\n- original_numbers_reproduced: no - this is an input-feature highway-env adaptation, not the paper's road-suite experiment.\n\nTarget outcomes were excluded from selection and used only for offline evaluation.\n",
        encoding="utf-8")
    (output / "deviations.md").write_text(
        "# Deviations from original DETOUR\n\nDETOUR-Scenario-H replaces curvature-distance ranking with frozen Cut-in input features because this harness uses straight roads. Road curvature compression is independently tested and plotted. The failure oracle is collision (not lane departure). The no-known-failure case uses declared seeded-uniform fallback. Static selections do not update executed/failure counts.\n",
        encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        type=Path,
                        default=Path("replications/detour_highway_env/configs/detour.yaml"))
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("results/highway_replications/shared/response_bank.npz"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    print(run(config, arguments.bank, arguments.output))


if __name__ == "__main__": main()
