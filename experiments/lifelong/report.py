"""Recompute the paired research criteria after all engineering gates pass."""

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.accumulation import lifelong as accumulation
from experiments.budget import protocol
from experiments.collect_transfer_records import ApprovedCase
from experiments.k_shot import CHECKPOINTS, LOWER_BETTER, METRICS, finite_mean, prediction_metrics
from experiments.lifelong_transfer import (
    BUDGET, CROSS_ARCHITECTURE, FROZEN, IN_FAMILY, NEW_TARGET, OUTPUT, PREDICTION,
    TARGETS, _audit_forecast, _verify_receipt, bank_for,
)
from experiments.lifelong_seals import verify as verify_seals
from experiments.paired_statistics import json_ready, paired_summary
from experiments.paper_results import paper_study
from experiments.protocol import ROOT
from experiments.public_ads import evaluate_e01
from experiments.regression import lifelong_replay as regression
from experiments.regression.dqn_evaluation import TARGET as REVISED_DQN
from experiments.regression.fixed_inputs import VERSIONS, methods_for, query_directory, version_directory
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.storage import read_json, write_json
from testing_methods.history_transfer.lifelong_selection import CONFIGURATIONS

BUDGET_METHODS = (*protocol.METHODS, *CONFIGURATIONS)
PREDICTION_METHODS = ("target_only", "pooled_history", "full", "frozen", "lifelong_full")
REPORT = OUTPUT / "reports"
CONTINUAL_METRICS = ("k0_nll", "k0_recall_50", "nll_init", "nll_all", "c50", "auc100", "c200", "auc")


def _binary_scores(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("Prequery forecasts and paid labels must be aligned and finite")
    bounded = np.clip(probabilities, 1e-12, 1 - 1e-12)
    losses = -np.where(labels == 1, np.log(bounded), np.log1p(-bounded))
    return losses, (probabilities - labels) ** 2


def verify_report_links():
    documents = (
        ROOT / "AGENTS.md", *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "controllers/README.md", ROOT / "experiments/README.md",
        ROOT / "testing_methods/history_transfer/README.md",
    )
    for document in documents:
        links = re.findall(r"\]\(([^\s)]+)\)", document.read_text(encoding="utf-8"))
        for link in links:
            path = link.split("#", 1)[0]
            if path and "://" not in path and not (document.parent / path).exists():
                raise ValueError(f"Missing local link in {document.name}: {link}")


def _origin(experiment, target, method):
    if method in CONFIGURATIONS or target == NEW_TARGET:
        if method == "frozen":
            return FROZEN
        return BUDGET if experiment == "budget" else PREDICTION
    if target in CROSS_ARCHITECTURE:
        if method == "frozen":
            return evaluate_e01.OUTPUT / "weight_adaptation"
        return evaluate_e01.OUTPUT / "budget_discovery" if experiment == "budget" else evaluate_e01.OUTPUT
    if method == "frozen":
        original = ROOT / "results/weight_adaptation"
    else:
        original = protocol.OUTPUT if experiment == "budget" else evaluate_e01.ORIGINAL
    return paper_study(original) if target == REVISED_DQN else original


def read_task(experiment, kind, target, method):
    directory = _origin(experiment, target, method)
    case = ApprovedCase(experiment, target, kind, protocol.SEED, "full" if method == "frozen" else method, None)
    saved = _verify_receipt(directory, case, derived=method == "random")
    if method in CONFIGURATIONS:
        _audit_forecast(saved["result"])
    return saved


def _summary(values, methods, baseline, *, lower_better=False, targets=TARGETS):
    return paired_summary(
        np.asarray(values)[:, :, None, ...], ads_ids=targets,
        ads_families=["In-Family" if t in IN_FAMILY else "Cross-Architecture" for t in targets],
        methods=methods, baseline=baseline, lower_better=lower_better, replicates=10000, seed=1301,
    )


def _strata(values):
    return {
        "in_family_mean": np.nanmean(values[:len(IN_FAMILY)], axis=(0, 1)),
        "cross_architecture_mean": np.nanmean(values[len(IN_FAMILY):], axis=(0, 1)),
    }


def budget_results():
    shape = (len(TARGETS), len(FORMAL_SCENARIOS), len(BUDGET_METHODS))
    values = np.empty((*shape, 2))
    curves = np.empty((*shape, 200), dtype=int)
    regions = np.empty((*shape, 3), dtype=int)
    resources = []
    for s, kind in enumerate(FORMAL_SCENARIOS):
        all_labels = np.asarray([
            read_task("budget", kind, target, "lifelong_full")["evaluation"]["candidate_collisions"]
            for target in TARGETS
        ], dtype=bool)
        in_family_collisions = all_labels[:len(IN_FAMILY)].sum(0)
        masks = (in_family_collisions == 8, (in_family_collisions > 0) & (in_family_collisions < 8),
                 in_family_collisions == 0)
        for a, target in enumerate(TARGETS):
            initial = None
            for m, method in enumerate(BUDGET_METHODS):
                result = read_task("budget", kind, target, method)
                order = result["result"]["selected_indices"]
                if initial is None:
                    initial = order[:20]
                if order[:20] != initial:
                    raise ValueError("Budget comparators have different initialization")
                flags = all_labels[a, order]
                curve = np.cumsum(flags)
                if curve.tolist() != result["metrics"]["discovery_curve"]:
                    raise ValueError("Budget discovery does not reproduce from the paid indices and labels")
                values[a, s, m] = curve[-1], curve.mean()
                curves[a, s, m] = curve
                regions[a, s, m] = [int(np.sum(flags & mask[order])) for mask in masks]
                measured = result["result"].get("resources")
                if measured is not None:
                    resources.append({"target": target, "scenario": kind, "method": method, **measured})
                if regions[a, s, m].sum() != curve[-1]:
                    raise ValueError("Regional discoveries do not add up to total discoveries")
    full, old, target_only, pooled = (BUDGET_METHODS.index(m) for m in (
        "lifelong_full", "full", "target_only", "pooled_history",
    ))
    paired = {m: _summary(values, BUDGET_METHODS, m) for m in protocol.METHODS}
    baseline_names = [method for method in protocol.METHODS if method != "full"]
    ads_gains = np.stack([paired[method]["paired"]["ads_gain"][:, full] for method in baseline_names], axis=1)
    draws = np.random.default_rng(1301).integers(0, len(TARGETS), (10000, len(TARGETS)))
    alpha = 0.05 / (len(baseline_names) * values.shape[-1])
    intervals = np.quantile(ads_gains[draws].mean(axis=1), (alpha / 2, 1 - alpha / 2), axis=0)
    simultaneous = {
        method: {"ci_low": intervals[0, index], "ci_high": intervals[1, index]}
        for index, method in enumerate(baseline_names)
    }
    specific = regions[8:, :, :, 2].mean(axis=(0, 1))
    local = BUDGET_METHODS.index("local_reliability")
    strong = {
        "all_six_baselines_positive_95ci": all(
            (interval["ci_low"] > 0).all() for interval in simultaneous.values()
        ),
        "ads_nonnegative_c200_against_target_only": int(((values[:, :, full, 0] - values[:, :, target_only, 0]).mean(1) >= 0).sum()),
        "online_mc_ppo_local_c200_gain_over_v1": float((values[8, :, local, 0] - values[8, :, old, 0]).mean()),
        "online_mc_ppo_local_c200_gain_over_target_only": float((values[8, :, local, 0] - values[8, :, target_only, 0]).mean()),
        "risk_ucb_added_c200_over_local": float((values[:, :, full, 0] - values[:, :, local, 0]).mean()),
        "mean_c200_gain_over_v1": float(paired["full"]["paired"]["gain"][full, 0]),
        "cross_architecture_c200_gain_over_target_only": float(
            (values[8:, :, full, 0] - values[8:, :, target_only, 0]).mean()
        ),
        "online_mc_ppo_c200_gain_over_v1": float((values[8, :, full, 0] - values[8, :, old, 0]).mean()),
        "specific_region_c200": dict(zip(BUDGET_METHODS, specific, strict=True)),
        "specific_region_gain_ratio_over_v1": specific[full] / specific[old] if specific[old] else None,
        "held_new_ads_c200_gains": {
            "target_only": float((values[-1, :, full, 0] - values[-1, :, target_only, 0]).mean()),
            "pooled_history": float((values[-1, :, full, 0] - values[-1, :, pooled, 0]).mean()),
        },
    }
    return {
        "methods": BUDGET_METHODS, "metrics": ("C200", "discovery_auc"),
        "values": values, "paired": paired, "simultaneous": simultaneous,
        "simultaneous_family_size": len(baseline_names) * values.shape[-1],
        **_strata(values), "curves": curves,
        "region_order": ("all_eight_in_family", "mixed_in_family", "none_in_family"),
        "region_counts": regions, "criteria": strong, "selector_resources": resources,
    }


def prediction_results():
    metrics = METRICS
    values = np.full((len(TARGETS), len(FORMAL_SCENARIOS), len(PREDICTION_METHODS), len(CHECKPOINTS), len(metrics)), np.nan)
    history_mass = np.empty((len(TARGETS), len(FORMAL_SCENARIOS), len(CHECKPOINTS)))
    calibration_methods = ("full", "frozen", "lifelong_full")
    sequential = np.empty((len(TARGETS), len(FORMAL_SCENARIOS), len(calibration_methods), 2))
    for a, target in enumerate(TARGETS):
        for s, kind in enumerate(FORMAL_SCENARIOS):
            split = None
            for m, method in enumerate(PREDICTION_METHODS):
                saved = read_task("prediction", kind, target, method)
                if split is None:
                    split = saved["specification"]["split"]
                if split != saved["specification"]["split"]:
                    raise ValueError("Prediction comparators use different held-out coordinates")
                if saved["result"]["selected_indices"] != split["query_order"]:
                    raise ValueError("Prediction replay changed the registered common feedback")
                for k, row in enumerate(saved["metrics"]):
                    if row["K"] != CHECKPOINTS[k]:
                        raise ValueError("Prediction checkpoint is out of order")
                    values[a, s, m, k] = [row.get(name) for name in metrics]
                if method in calibration_methods:
                    queries = saved["result"]["queries"]
                    labels = [row["feedback"]["collision"] for row in saved["query_ledger"]]
                    losses = _binary_scores([row["prequery_probability"] for row in queries], labels)
                    sequential[a, s, calibration_methods.index(method)] = [score.mean() for score in losses]
                if method == "lifelong_full":
                    for k, snapshot in enumerate(saved["result"]["prediction_snapshots"]):
                        weights = np.asarray(snapshot["component_weights"])
                        history_mass[a, s, k] = weights[:-2].sum(axis=0).mean()
    rss_path = evaluate_e01.OUTPUT / "weight_adaptation/reports/rss_discovery/paired_summary.json"
    rss_coverage = read_json(rss_path)
    if rss_coverage["status"] != "complete" or rss_coverage["scopes"]["overall"]["pairs"] != 12:
        raise ValueError("Public low-RSS slice does not contain the 12 registered pairs")
    few = [i for i, count in enumerate(CHECKPOINTS) if count in (5, 10, 20, 50)]
    average = finite_mean(values[..., few, :], axis=-2)
    full, frozen, v1 = (PREDICTION_METHODS.index(m) for m in ("lifelong_full", "frozen", "full"))
    nll_delta = average[:, :, full, 0] - average[:, :, frozen, 0]
    return {
        "methods": PREDICTION_METHODS, "metrics": metrics, "checkpoints": CHECKPOINTS,
        "values": values, "historical_weight": history_mass,
        "change_from_own_k0": values - values[..., :1, :],
        "by_checkpoint": {
            comparator: _summary(values, PREDICTION_METHODS, comparator, lower_better=LOWER_BETTER)
            for comparator in PREDICTION_METHODS if comparator != "lifelong_full"
        },
        "few_shot_vs_target_only": _summary(average, PREDICTION_METHODS, "target_only", lower_better=LOWER_BETTER),
        "few_shot_vs_pooled": _summary(average, PREDICTION_METHODS, "pooled_history", lower_better=LOWER_BETTER),
        "few_shot_paired": _summary(average, PREDICTION_METHODS, "frozen", lower_better=LOWER_BETTER),
        "few_shot_nll_wins_of_48": int((nll_delta < 0).sum()),
        "cross_architecture_calibration_difference_from_v1_by_checkpoint": (
            values[8:, :, full, :, :2] - values[8:, :, v1, :, :2]
        ).mean(axis=(0, 1)),
        "sequential_calibration": {
            "methods": calibration_methods, "metrics": ("nll", "brier"), "values": sequential,
            "paired": _summary(sequential, calibration_methods, "frozen", lower_better=True),
        },
        "public_rss_coverage": {"evidence": rss_path.relative_to(ROOT).as_posix(), **rss_coverage},
    }


def _continual_scores(saved):
    result = saved["result"]
    order = result["selected_indices"]
    query_rows = result["queries"]
    ledger = saved["query_ledger"]
    labels = np.asarray(saved["evaluation"]["candidate_collisions"], dtype=int)
    observed = labels[order]
    initial = saved["specification"]["initial_indices"]
    if (len(order) != 200 or len(initial) != 20 or order[:20] != initial
            or order != [row["index"] for row in query_rows]
            or order != [row["index"] for row in ledger]
            or observed.tolist() != [row["feedback"]["collision"] for row in ledger]):
        raise ValueError("Continual metrics require the registered paid order and labels")
    scores, _ = _binary_scores([row["prequery_probability"] for row in query_rows], observed)
    curve = np.cumsum(observed)
    if curve.tolist() != saved["metrics"]["discovery_curve"]:
        raise ValueError("Continual discovery does not reproduce from paid responses")
    prior = result["prediction_snapshots"][0]
    if prior["paid_queries"] != 0 or prior["observed_indices"]:
        raise ValueError("The zero-shot metric requires a snapshot before target feedback")
    ids = [row["scenario_id"] for row in saved["specification"]["candidates"]]
    metrics, _ = prediction_metrics(prior["probabilities"], labels, ids, np.arange(len(ids)), top=(50,))
    return {
        "k0_nll": metrics["nll"], "k0_recall_50": metrics["recall_50"],
        "k0_recall_undefined": metrics["recall_undefined"], "nll_init": scores[:20].mean(),
        "nll_all": scores.mean(), "c50": int(curve[49]), "auc100": curve[:100].mean(),
        "c200": int(curve[-1]), "auc": curve.mean(),
        "pool_collisions": int(labels.sum()), "pool_candidates": len(labels),
    }


def _continual_effect(rows):
    if not rows:
        return {}
    ids = [target for target in TARGETS if any(row["target"] == target for row in rows)]
    ads_arrays = []
    for target in ids:
        scene_values = []
        for scenario in FORMAL_SCENARIOS:
            paired_rows = [row for row in rows if row["target"] == target and row["scenario"] == scenario]
            if paired_rows:
                scene_values.append(finite_mean([[row[m] for m in CONTINUAL_METRICS] for row in paired_rows], axis=0))
        ads_arrays.append(finite_mean(scene_values, axis=0))
    gains = np.asarray(ads_arrays)
    values = np.stack([np.zeros_like(gains), gains], axis=1)[:, None, None, :, :]
    summary = paired_summary(
        values, ads_ids=ids, ads_families=["In-Family" if t in IN_FAMILY else "Cross-Architecture" for t in ids],
        methods=("reference", "difference"), baseline="reference", lower_better=False, replicates=10000, seed=1301,
    )["paired"]
    effects = {}
    for i, metric in enumerate(CONTINUAL_METRICS):
        finite = np.asarray([row[metric] for row in rows if row[metric] is not None], dtype=float)
        orientation = -1 if "nll" in metric else 1
        valid_ads = int(summary["valid_ads"][1, i])
        effects[metric] = {
            "mean": summary["gain"][1, i], "ci_low": summary["gain_ci_low"][1, i] if valid_ads > 1 else None,
            "ci_high": summary["gain_ci_high"][1, i] if valid_ads > 1 else None, "ads": valid_ads,
            "interval_undefined": "fewer_than_two_defined_ads" if valid_ads < 2 else None,
            "pairs": len(finite), "wins": int((orientation * finite > 1e-12).sum()),
            "ties": int((np.abs(finite) <= 1e-12).sum()), "losses": int((orientation * finite < -1e-12).sum()),
        }
    return effects


def continual_results(study_names=tuple(accumulation.PARTITION_SEEDS)):
    rows, gates, contrasts, studies = [], [], [], {}
    for study in study_names:
        registration = read_json(accumulation.ROOT / study / "protocol.json")
        regimes = list(registration["regimes"])
        fixed = regimes[0]
        comparisons = [(method, fixed) for method in ("naive_append", "debiased_append", "full_lifelong")]
        comparisons += [("debiased_append", "naive_append"), ("full_lifelong", "debiased_append")]
        if study == "in_family":
            comparisons += [("naive_append", "legacy_fixed"), ("full_lifelong", "legacy_fixed")]
        study_pairs = defaultdict(list)
        for kind in FORMAL_SCENARIOS:
            for sequence in accumulation.sequences(study):
                previous_decision = None
                for order, target in enumerate(sequence.target_ids, 1):
                    key = dict(study=study, scenario=kind, branch=sequence.name, order=order, target=target)
                    point = {}
                    for regime in regimes:
                        saved = accumulation.branch_result(kind, sequence, regime, target)
                        point[regime] = _continual_scores(saved)
                        rows.append({**key, "regime": regime, **point[regime]})
                    if study == "in_family":
                        legacy_saved, _ = accumulation._sealed_in_family(kind, sequence, "fixed", target)
                        point["legacy_fixed"] = _continual_scores(legacy_saved)
                    for left, right in comparisons:
                        delta = {
                            metric: point[left][metric] - point[right][metric]
                            if point[left][metric] is not None and point[right][metric] is not None else None
                            for metric in CONTINUAL_METRICS
                        }
                        row = {**key, "left": left, "right": right, "previous_gate": previous_decision, **delta}
                        contrasts.append(row)
                        study_pairs[left, right].append(row)
                    if order < len(sequence.target_ids):
                        path = (accumulation.ROOT / study / kind / sequence.name / "full_lifelong"
                                / "tasks" / target / "gate.json")
                        receipt = read_json(path)
                        previous_decision = receipt["decision"]
                        gates.append({**key, **{k: receipt[k] for k in (
                            "decision", "validation_old_nll", "validation_new_nll",
                        )}})
        scopes = {}
        for (left, right), paired in study_pairs.items():
            scopes[f"{left}_minus_{right}"] = {
                "all_positions": _continual_effect(paired),
                "after_first": _continual_effect([row for row in paired if row["order"] > 1]),
                "by_order": dict(
                    (str(order), _continual_effect([row for row in paired if row["order"] == order]))
                    for order in sorted({row["order"] for row in paired})
                ),
                "by_gate": {
                    gate: _continual_effect([row for row in paired if row["previous_gate"] == gate])
                    for gate in ("gate_accepted", "gate_rejected_retained_parent")
                },
                "by_family": {
                    family: _continual_effect([row for row in paired if row["order"] > 1 and row["target"] in cohort])
                    for family, cohort in (("in_family", IN_FAMILY), ("cross_architecture", CROSS_ARCHITECTURE))
                },
            }
        studies[study] = {"fixed": fixed, "comparisons": scopes}
    cold_fixed = [row for row in rows if row["study"] == "cold_start" and row["regime"] == "fixed"]
    saturation = {
        "fraction_of_reachable_ceiling": sum(row["c200"] for row in cold_fixed) /
        sum(min(200, row["pool_collisions"]) for row in cold_fixed),
        "by_scenario": {
            scenario: sum(row["c200"] for row in cold_fixed if row["scenario"] == scenario) /
            sum(min(200, row["pool_collisions"]) for row in cold_fixed if row["scenario"] == scenario)
            for scenario in FORMAL_SCENARIOS
        },
    }
    return {
        "rows": rows, "pairs": contrasts, "gates": gates, "studies": studies,
        "metrics": CONTINUAL_METRICS, "cold_start_saturation": saturation,
        "interval": {"unit": "ADS", "replicates": 10000, "seed": 1301, "confidence_level": 0.95},
        "gate_rejection_rate": np.mean([row["decision"] == "gate_rejected_retained_parent" for row in gates]),
    }


def regression_results():
    rows = []
    devices = Counter()
    for policy in regression.POLICIES:
        for kind in FORMAL_SCENARIOS:
            for version in regression.TARGET_VERSIONS:
                name = version_directory(kind, "truth", version, policy).name
                root = regression.ROOT / policy / kind / name
                saved = read_json(root / "result.json.gz")
                _audit_forecast(saved["result"])
                devices[saved["result"]["conditional_inference_device"]] += 1
                previous_version = VERSIONS[VERSIONS.index(version) - 1]
                current = read_json(version_directory(kind, "truth", version, policy) / "sealed.json")
                previous = read_json(version_directory(kind, "truth", previous_version, policy) / "sealed.json")
                labels = [row["collision"] for row in current["feedback"]]
                previous_labels = [row["collision"] for row in previous["feedback"]]
                initial = saved["specification"]["initial_indices"]
                metrics = read_json(root / "paired_metrics.json")
                controls = {method: read_json(query_directory(kind, version, method, policy) / "result.json.gz")
                            for method in methods_for(policy)}
                for method, recorded in {**controls, "lifelong_full": saved}.items():
                    order = regression._checked_order(recorded, current, initial)
                    row = regression.score(previous_labels, labels, order)
                    if row != metrics[method]:
                        raise ValueError(f"Fixed-suite metrics do not reproduce from the paid order in {root}")
                    rows.append(dict(policy=policy, scenario=kind, version=version, method=method, **row))
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["policy"], row["scenario"], row["version"])][row["method"]] = row
    totals = {
        policy: {
            method: sum(
                by_method[method]["new_in_first_100"]
                for (p, _, _), by_method in grouped.items() if p == policy
            )
            for method in ("lifelong_full", "full", "sdc_scissor")
        }
        for policy in regression.POLICIES
    }
    comparisons = {}
    for method in ("full", "detour", "birchler", "sdc_scissor", "spectre"):
        deltas = [
            methods["lifelong_full"]["q90"] - methods[method]["q90"]
            for methods in grouped.values()
            if methods["lifelong_full"]["q90"] is not None and methods[method]["q90"] is not None
        ]
        comparisons[method] = dict(pairs=len(deltas), mean_delta=np.mean(deltas), median_delta=np.median(deltas),
                                   wins=sum(d < 0 for d in deltas),
                                   ties=sum(d == 0 for d in deltas), losses=sum(d > 0 for d in deltas))
    early_gain = {}
    for policy in regression.POLICIES:
        values = np.asarray([
            sum(
                methods["lifelong_full"]["new_in_first_100"] - methods["full"]["new_in_first_100"]
                for (p, k, _), methods in grouped.items() if p == policy and k == kind
            )
            for kind in FORMAL_SCENARIOS
        ])
        generator = np.random.default_rng(1301)
        draws = generator.integers(0, len(values), (10000, len(values)))
        low, high = np.quantile(values[draws].sum(1), (0.025, 0.975))
        early_gain[policy] = {"total": int(values.sum()), "ci_low": low, "ci_high": high}
    baselines = ("detour", "birchler", "sdc_scissor", "spectre")
    joint_q90_wins = sum(
        all(
            methods["lifelong_full"]["q90"] is not None and methods[baseline]["q90"] is not None
            and methods["lifelong_full"]["q90"] < methods[baseline]["q90"]
            for baseline in baselines
        )
        for methods in grouped.values()
    )
    return {
        "rows": rows, "q90": comparisons, "joint_q90_wins": joint_q90_wins,
        "means": {
            method: {metric: finite_mean([row[metric] for row in rows if row["method"] == method])
                     for metric in ("q90", "q90_fraction", "auc_500", "new_in_first_100")}
            for method in ("lifelong_full", "full", *baselines)
        },
        "new_in_first_100": totals, "early_gain": early_gain, "replay_devices": dict(devices),
    }


def driving_cases(budget):
    from experiments.public_ads.mechanism import replay_decisions

    cases = {}
    for target, method, comparator in (
        ("online_mc_ppo", "local_reliability", "full"),
        ("decision_transformer", "lifelong_full", "target_only"),
    ):
        a = TARGETS.index(target)
        m, b = map(BUDGET_METHODS.index, (method, comparator))
        delta = budget["values"][a, :, m, 0] - budget["values"][a, :, b, 0]
        kind = FORMAL_SCENARIOS[int(np.argmax(delta))]
        better = read_task("budget", kind, target, method)
        other = read_task("budget", kind, target, comparator)
        other_set = set(other["result"]["selected_indices"])
        hit = next(row for row in better["query_ledger"]
                   if row["index"] not in other_set and row["feedback"]["collision"])
        index = hit["index"]
        bank = bank_for(kind, target)
        path = bank.execution_path(index)
        native = read_json(path)
        if native["scenario"] != better["specification"]["candidates"][index] or not native["trajectory"][-1]["collision"]:
            raise ValueError("Driving case must match the originally executed candidate and collision")
        forecast = better["result"]["queries"][hit["call"] - 1]
        case = {
            "scenario": kind, "parameters": native["scenario"]["parameters"], "candidate_index": index,
            "query": hit["call"], "method": method, "comparator": comparator, "delta_c200": float(delta.max()),
            "native_path": path.relative_to(ROOT).as_posix(), "native_sha256": protocol.sha(path),
            "native_events": native["events"], "collision_time": native["trajectory"][-1]["time_s"],
            "collision_partner": native["trajectory"][-1]["collision_partner"],
            "prequery_probability": forecast["prequery_probability"],
            "prequery_historical_weight": forecast["prequery_historical_weight"],
            "decisions": replay_decisions(kind, target, native["trajectory"]),
        }
        if target == "online_mc_ppo":
            count = 20
            local = next(s for s in better["result"]["prediction_snapshots"] if s["paid_queries"] == count)
            global_ = next(s for s in other["result"]["prediction_snapshots"] if s["paid_queries"] == count)
            if local["observed_indices"] != global_["observed_indices"]:
                raise ValueError("Spatial comparison requires the same paid initialization")
            case["common_prefix_surface"] = {
                "paid": count, "parameters": [row["parameters"] for row in better["specification"]["candidates"]],
                "v1_probabilities": global_["probabilities"], "local_probabilities": local["probabilities"],
                "local_historical_weight": np.asarray(local["component_weights"][:-2]).sum(axis=0),
                "paid_indices": local["observed_indices"],
            }
        cases[target] = case
    return {"new_physical_executions": 0, "cases": cases}


def generate():
    accumulation.verify_stage_three()
    verify_report_links()
    seals = verify_seals()
    budget = budget_results()
    report = {
        "targets": TARGETS, "scenarios": FORMAL_SCENARIOS,
        "budget": budget, "prediction": prediction_results(), "driving_cases": driving_cases(budget),
        "continual": continual_results(), "regression": regression_results(),
        "sealed_source_files": len(seals["source_files"]),
        "sealed_artifacts": sum(tree["files"] for tree in seals["artifact_trees"].values()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(REPORT / "summary.json", json_ready(report))
    write_readme(report)
    return report


def write_readme(report):
    budget, prediction = report["budget"], report["prediction"]
    continuing, regression_summary = report["continual"], report["regression"]
    live = budget["methods"].index("lifelong_full")
    labels = {
        "full": "Ours v1", "target_only": "Target-only GP", "pooled_history": "Pooled GP",
        "scenario_fuzz": "ScenarioFuzz", "paired_system_active_evaluation": "配对系统",
        "static_prior": "静态先验", "random": "Random",
    }
    lines = [
        "# 12-ADS 持续风险迁移结果",
        "",
        "四场景共 48 个任务，目标 ADS 与场景在配对汇总中等权。总体区间按 ADS 整块重采样，"
        "10,000 次，种子 1301。前八个同范式系统曾参与场景参数域开发核验，开发坐标已隔离。"
        "四个跨范式系统未参与场景设计。",
        "",
        "## 预算内碰撞发现",
        "",
        "| 对照 | ΔC(200) [95% 区间] | ΔDiscovery AUC [95% 区间] |",
        "|---|---:|---:|",
    ]
    for method, statistics in budget["paired"].items():
        paired = statistics["paired"]
        interval = budget["simultaneous"].get(method, {
            "ci_low": paired["gain_ci_low"][live], "ci_high": paired["gain_ci_high"][live],
        })
        effects = [
            f"{paired['gain'][live][i]:+.2f} [{interval['ci_low'][i]:+.2f}, {interval['ci_high'][i]:+.2f}]"
            for i in range(2)
        ]
        lines.append(f"| {labels[method]} | {' | '.join(effects)} |")
    lines.extend(("", "六项领域对照的两项主指标使用 12 项 Bonferroni 同时区间。v1 行使用边际区间。"))
    for label, key in (("同范式 8 ADS", "in_family_mean"), ("跨范式 4 ADS", "cross_architecture_mean")):
        c200, auc = budget[key][live]
        lines.extend(("", f"{label}的 Ours method 平均 C(200) 为 {c200:.2f}，Discovery AUC 为 {auc:.2f}。"))
    criteria = budget["criteria"]
    specific_gain = criteria["specific_region_gain_ratio_over_v1"]
    ratio = "基准分母为零" if specific_gain is None else f"为 v1 的 {specific_gain:.2f} 倍"
    learning = prediction["few_shot_paired"]["paired"]
    calibrated = prediction["methods"].index("lifelong_full")
    cold = continuing["studies"]["cold_start"]["comparisons"]
    relay = continuing["studies"]["cross_architecture"]["comparisons"]
    recovery = continuing["studies"]["in_family"]["comparisons"]
    terminal = relay["full_lifelong_minus_fixed"]["by_order"]["5"]["auc"]
    in_family = recovery["full_lifelong_minus_fixed"]["after_first"]["auc"]
    raw = continuing["pairs"]
    stress = [row for row in raw if row["study"] == "in_family" and row["target"] == "target_linear_mpc"
              and row["left"] == "naive_append" and row["right"] == "legacy_fixed"]
    worst = min(stress, key=lambda row: row["auc"])
    match_keys = ("scenario", "branch", "order", "target")
    corrected = next(row for row in raw if row["study"] == "in_family" and row["left"] == "full_lifelong"
                     and row["right"] == "fixed" and all(row[key] == worst[key] for key in match_keys))
    lines.extend((
        "",
        f"跨范式组相对 Target-only GP 平均多发现 {criteria['cross_architecture_c200_gain_over_target_only']:.2f} "
        f"个碰撞。特有失效区的平均发现数{ratio}，预登记目标为 2 倍。"
        f"Continuous SAC 相对 Target-only GP 和 Pooled GP 各多发现 "
        f"{criteria['held_new_ads_c200_gains']['target_only']:.2f} 和 "
        f"{criteria['held_new_ads_c200_gains']['pooled_history']:.2f} 个碰撞。",
        "",
        "## 校准与持续积累",
        "",
        f"在 K = 5、10、20、50 的少反馈检查点平均后，Ours method 相对冻结权重降低留出 NLL 的"
        f"任务数为 {prediction['few_shot_nll_wins_of_48']}／48。平均 NLL 降幅为 "
        f"{learning['gain'][calibrated][0]:.4f}，95% 区间 "
        f"[{learning['gain_ci_low'][calibrated][0]:.4f}, {learning['gain_ci_high'][calibrated][0]:.4f}]。",
        "",
        "## 冷启动与持续演化",
        "",
        "E05-A 第 2 至 4 轮的差值为左侧方法减右侧方法。NLL 越低越好，"
        "召回和发现指标越高越好。区间按目标 ADS 整块重采样。",
        "",
        "| 指标 | 朴素 − 固定 | 去偏 − 朴素 | 完整 − 去偏 | 完整 − 固定 |",
        "|---|---:|---:|---:|---:|",
    ))
    order = ("naive_append_minus_fixed", "debiased_append_minus_naive_append",
             "full_lifelong_minus_debiased_append", "full_lifelong_minus_fixed")
    for metric, label in zip(CONTINUAL_METRICS, (
        "K=0 NLL", "K=0 Recall@50（百分点）", "NLL_init", "NLL_all",
        "C(50)", "AUC@100", "C(200)", "AUC@200",
    ), strict=True):
        scale = 100 if metric == "k0_recall_50" else 1
        digits = 4 if "nll" in metric else 2
        cells = []
        for name in order:
            point = cold[name]["after_first"][metric]
            mean, low, high = [point[key] * scale for key in ("mean", "ci_low", "ci_high")]
            cells.append(f"{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    fraction = continuing["cold_start_saturation"]["fraction_of_reachable_ceiling"]
    devices = "，".join(f"{device.upper()} {count} 条" for device, count in
                         sorted(regression_summary["replay_devices"].items()))
    lines.extend((
        "",
        f"冷启动固定两来源在 200 次查询内达到可达碰撞上限的 {fraction:.1%}。"
        f"全套持续更新的内部拒绝率为 {continuing['gate_rejection_rate']:.1%}。",
        "",
        f"E05-B 末端 PPO 在 {terminal['wins']}／{terminal['pairs']} 个位置提高 AUC@200，"
        f"平均差值为 {terminal['mean']:+.3f}。这些位置来自同一 ADS 的四场景正反序。",
        "",
        "E05-C 固定增强选择器用 32 个唯一回放覆盖 64 个正反序位置。"
        f"第 2 至 4 轮完整方法相对固定增强选择器的平均 ΔAUC@200 为 {in_family['mean']:+.3f}，"
        f"95% 区间 [{in_family['ci_low']:+.3f}, {in_family['ci_high']:+.3f}]。"
        f"Linear MPC 个案中，原朴素更新相对原固定对照的差值为 {worst['auc']:+.3f}。"
        f"同一位置的完整方法相对固定增强选择器为 {corrected['auc']:+.3f}。",
        "",
        "## 同谱系版本回归",
        "",
        f"在 {regression_summary['joint_q90_wins']}／36 个配对任务上，Ours method 的 Q90 同时优于四项基线。"
        "前 100 次新增碰撞按当前版本相对上一版本的同坐标碰撞差集统计。",
        f"正式回放的计算设备分布为 {devices}。",
        "",
        "| 策略 | Ours method | Ours v1 | SDC-Scissor | 相对 v1 差值 [95% 区间] |",
        "|---|---:|---:|---:|---:|",
    ))
    for policy, totals in regression_summary["new_in_first_100"].items():
        gain = regression_summary["early_gain"][policy]
        lines.append(
            f"| {policy} | {totals['lifelong_full']} | {totals['full']} | {totals['sdc_scissor']} | "
            f"{gain['total']:+d} [{gain['ci_low']:+.1f}, {gain['ci_high']:+.1f}] |"
        )
    lines.extend((
        "",
        "新增碰撞数的区间以场景为重采样单位，每次保留该场景三个版本的配对差值。",
        "",
        "## 图注",
        "",
        "- **逐 ADS 发现曲线。** 每条曲线为四场景等权平均。各方法共享前 20 条初始化反馈。"
        " [发现曲线](../../figures/lifelong_transfer/discovery_by_ads.png)",
        "- **逐 ADS 失效区构成与跨架构消融。** 按同坐标上八个同范式目标中发生碰撞的数量划分区域，"
        "区间为 8、1–7 和 0。图中数量为各场景前 200 次查询的等权平均。"
        " [失效区构成](../../figures/lifelong_transfer/discovery_region_composition.png) · "
        "[跨架构消融](../../figures/lifelong_transfer/cross_architecture_regions.png)",
        "- **少反馈预测与权重。** 热图以 Target-only GP 为对照，蓝色正值代表 Ours method 改善。"
        "权重图在每个任务的 2,000 个候选上汇总历史来源权重，按 ADS 与场景等权。"
        " [预测热图](../../figures/lifelong_transfer/prediction_gain_heatmaps.png) · "
        "[校准与权重](../../figures/lifelong_transfer/calibration_and_weights.png)",
        "- **冷启动积累。** 图中给出四项对照的后续三轮轨迹和 ADS 整块 95% 区间，每轮含三个目标 ADS。"
        " [成长与消融](../../figures/lifelong_transfer/cold_start_growth.png)",
        "- **局部可信度。** 使用共同前 20 条反馈计算概率差和局部历史权重，"
        "同二维坐标的其他参数与类别取实际候选均值，星号表示归档案例所在二维坐标。"
        " [局部可信度空间分布](../../figures/lifelong_transfer/local_reliability_surface.png)",
        "",
        "[配对统计、分层结果、区域分解和计算资源](reports/summary.json)",
        "[执行计划](../../docs/experiment_plan.md)",
        "",
    ))
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    generate()
