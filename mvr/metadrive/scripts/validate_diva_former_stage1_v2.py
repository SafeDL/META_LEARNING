"""Execute the source-only Stage 1-v2 headroom and teacher gates."""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np
import yaml

from ..diva.source_bank import SourceBank, observation_from_dict
from ..diva_ai.counterfactual_teacher_v2 import (
    V2_BASELINE_POLICIES,
    CounterfactualTeacherV2,
)
from ..diva_ai.formal_calibrator import formal_class_indexes
from ..provenance import content_hash


FORBIDDEN_SOURCE_REFS = {"idm_fast_small_gap", "idm_late_response"}


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_bank(path: Path, source_refs: tuple[str, ...]) -> SourceBank:
    rows = [
        observation_from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    bank = SourceBank.from_observations(rows, source_refs)
    bank.require_common_anchors()
    if set(bank.source_refs) & FORBIDDEN_SOURCE_REFS:
        raise ValueError("sealed validation/test SUT appears in the Stage 1-v2 source bank")
    return bank


def _read_inputs(config_path: str | Path) -> tuple[Path, Path, dict, dict, SourceBank]:
    config_file = Path(config_path)
    root = config_file.resolve().parents[3]
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("schema") != "diva_former_stage1_v2":
        raise ValueError("Stage 1-v2 validator requires a v2 config")
    if not str(config["results_dir"]).replace("\\", "/").endswith("/stage1_v2"):
        raise ValueError("Stage 1-v2 results must use an isolated stage1_v2 directory")
    base_config = yaml.safe_load((root / config["base_config"]).read_text(encoding="utf-8"))
    source_refs = tuple(base_config["study"]["source_sut_refs"])
    bank = _load_bank(root / config["source_observations"], source_refs)
    return root, config_file, config, base_config, bank


def build_manifest(config_path: str | Path) -> dict[str, Any]:
    root, config_file, config, _, bank = _read_inputs(config_path)
    protocol_file = root / config["protocol_document"]
    source_file = root / config["source_observations"]
    loso_file = root / config["source_loso_reference"]
    v1_gate = root / "results/metadrive/diva_former/cutin_g01/stage1/stage1_gate.json"
    manifest = {
        "schema": "diva_former_stage1_manifest_v2",
        "git_commit": _git_commit(root),
        "config_sha256": _file_hash(config_file),
        "protocol_sha256": _file_hash(protocol_file),
        "base_config_sha256": _file_hash(root / config["base_config"]),
        "source_observations_sha256": _file_hash(source_file),
        "source_loso_reference_sha256": _file_hash(loso_file),
        "stage1_v1_gate_sha256": _file_hash(v1_gate),
        "implementation_sha256": {
            "formal_calibrator": _file_hash(root / "mvr/metadrive/diva_ai/formal_calibrator.py"),
            "counterfactual_teacher_v2": _file_hash(
                root / "mvr/metadrive/diva_ai/counterfactual_teacher_v2.py"
            ),
            "validator": _file_hash(
                root / "mvr/metadrive/scripts/validate_diva_former_stage1_v2.py"
            ),
        },
        "config_hash": content_hash(config),
        "source_refs": list(bank.source_refs),
        "source_only": True,
        "new_simulator_calls": 0,
        "sealed_sut_refs": sorted(FORBIDDEN_SOURCE_REFS),
        "budgets": {
            "primary": int(config["protocol"]["primary_budget"]),
            "final": int(config["protocol"]["final_budget"]),
        },
        "rebuild_command": (
            "conda run -n metadrive python -m "
            "mvr.metadrive.scripts.validate_diva_former_stage1_v2 "
            "--config mvr/metadrive/configs/diva_former_stage1_v2.yaml"
        ),
    }
    output = root / config["results_dir"] / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _teachers(
    bank: SourceBank, config: dict, base_config: dict
) -> dict[int, CounterfactualTeacherV2]:
    settings = config["formal_calibration"]
    result = {}
    for held in range(len(bank.source_refs)):
        training = tuple(index for index in range(len(bank.source_refs)) if index != held)
        result[held] = CounterfactualTeacherV2.from_source_bank(
            bank,
            training,
            rank=int(config["teacher"]["rank"]),
            proxy_event_threshold=float(
                base_config["vulnerability_response"]["proxy_event_threshold"]
            ),
            calibration_context_sizes=tuple(int(value) for value in settings["context_sizes"]),
            calibration_l2=float(settings["l2"]),
            calibration_max_iterations=int(settings["max_iterations"]),
        )
    return result


def _curve_metrics(curve: Iterable[float], budgets: tuple[int, ...]) -> dict[str, Any]:
    values = tuple(float(value) for value in curve)
    first = next((index + 1 for index, value in enumerate(values) if value > 0.0), None)
    result: dict[str, Any] = {"first_critical_step": first, "formal_curve": list(values)}
    for budget in budgets:
        result[f"score_at_{budget}"] = values[budget - 1]
        result[f"auc_at_{budget}"] = float(sum(values[:budget]))
    return result


def _replay_rows(
    bank: SourceBank,
    teachers: dict[int, CounterfactualTeacherV2],
    budgets: tuple[int, ...],
) -> list[dict[str, Any]]:
    final_budget = max(budgets)
    policies = ("teacher", *V2_BASELINE_POLICIES)
    rows: list[dict[str, Any]] = []
    for held, teacher in teachers.items():
        training = tuple(index for index in range(len(bank.source_refs)) if index != held)
        for source_index in training:
            for policy in policies:
                evaluation = teacher.run(
                    source_index,
                    final_budget,
                    policy,
                )
                row = {
                    "fold": held,
                    "held_out_source": bank.source_refs[held],
                    "source_sut": bank.source_refs[source_index],
                    "policy": policy,
                    "selected_design_ids": [
                        bank.design_ids[index] for index in evaluation.selected_indexes
                    ],
                    "cumulative_vulnerability_at_20": float(
                        evaluation.vulnerability_curve[-1]
                    ),
                }
                row.update(_curve_metrics(evaluation.formal_curve, budgets))
                rows.append(row)
    return rows


def _mean(rows: list[dict[str, Any]], policy: str, metric: str, sources: set[str]) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row["policy"] == policy and row["source_sut"] in sources
    ]
    if not values:
        raise ValueError("aggregate has no eligible replay rows")
    return float(np.mean(values))


def _best_baseline(
    rows: list[dict[str, Any]], metric: str, sources: set[str]
) -> tuple[str, float]:
    values = {
        policy: _mean(rows, policy, metric, sources) for policy in V2_BASELINE_POLICIES
    }
    return max(values.items(), key=lambda item: (item[1], item[0]))


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _headroom_audit(
    bank: SourceBank,
    rows: list[dict[str, Any]],
    config: dict,
) -> dict[str, Any]:
    budgets = tuple(int(value) for value in config["protocol"]["reporting_budgets"])
    event_sources = set(config["protocol"]["event_bearing_sources"])
    source_counts = {}
    positive_sets = {}
    for index, source in enumerate(bank.source_refs):
        scores = bank.formal_scores[index]
        source_counts[source] = {
            "collision": int(np.sum(scores == 1.0)),
            "near_miss": int(np.sum(scores == 0.5)),
            "formal_positive": int(np.sum(scores > 0.0)),
            "formal_total": float(np.sum(scores)),
        }
        positive_sets[source] = set(np.flatnonzero(scores > 0.0))
    overlaps = [
        {
            "left": left,
            "right": right,
            "intersection": len(positive_sets[left] & positive_sets[right]),
            "union": len(positive_sets[left] | positive_sets[right]),
            "jaccard": _jaccard(positive_sets[left], positive_sets[right]),
        }
        for left, right in combinations(bank.source_refs, 2)
    ]

    fold_rows = []
    replay_groups: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        replay_groups.setdefault((int(row["fold"]), row["source_sut"]), {})[
            row["policy"]
        ] = row
    for (fold, source), policies in sorted(replay_groups.items()):
        source_index = bank.source_refs.index(source)
        oracle_rewards = np.sort(bank.formal_scores[source_index])[::-1]
        oracle_curve = np.cumsum(oracle_rewards[: max(budgets)])
        entry: dict[str, Any] = {"fold": fold, "source_sut": source, "metrics": {}}
        for budget in budgets:
            policy_metrics = {
                policy: {
                    "score": float(row[f"score_at_{budget}"]),
                    "auc": float(row[f"auc_at_{budget}"]),
                }
                for policy, row in policies.items()
            }
            entry["metrics"][str(budget)] = {
                "oracle_score": float(oracle_curve[budget - 1]),
                "oracle_auc": float(np.sum(oracle_curve[:budget])),
                "policies": policy_metrics,
            }
        overlap_policies = (
            "teacher",
            "mine_now",
            "formal_mine_now",
            "highest_shared_risk",
        )
        entry["top_b_overlap"] = {}
        for budget in budgets:
            selections = {
                policy: set(policies[policy]["selected_design_ids"][:budget])
                for policy in overlap_policies
            }
            entry["top_b_overlap"][str(budget)] = {
                f"{left}__{right}": _jaccard(selections[left], selections[right])
                for left, right in combinations(overlap_policies, 2)
            }
        fold_rows.append(entry)

    aggregate = {}
    for budget in budgets:
        oracle_scores = []
        oracle_aucs = []
        for row in fold_rows:
            if row["source_sut"] in event_sources:
                oracle_scores.append(row["metrics"][str(budget)]["oracle_score"])
                oracle_aucs.append(row["metrics"][str(budget)]["oracle_auc"])
        score_policy, score_baseline = _best_baseline(
            rows, f"score_at_{budget}", event_sources
        )
        auc_policy, auc_baseline = _best_baseline(rows, f"auc_at_{budget}", event_sources)
        aggregate[str(budget)] = {
            "oracle_score": float(np.mean(oracle_scores)),
            "best_baseline_score_policy": score_policy,
            "best_baseline_score": score_baseline,
            "terminal_headroom": float(np.mean(oracle_scores) - score_baseline),
            "oracle_auc": float(np.mean(oracle_aucs)),
            "best_baseline_auc_policy": auc_policy,
            "best_baseline_auc": auc_baseline,
            "auc_headroom": float(np.mean(oracle_aucs) - auc_baseline),
        }

    source_auc_headroom = {}
    for source in sorted(event_sources):
        subset = set((source,))
        policy, baseline = _best_baseline(rows, "auc_at_20", subset)
        source_index = bank.source_refs.index(source)
        curve = np.cumsum(np.sort(bank.formal_scores[source_index])[::-1][:20])
        source_auc_headroom[source] = {
            "oracle": float(np.sum(curve)),
            "baseline_policy": policy,
            "baseline": baseline,
            "headroom": float(np.sum(curve) - baseline),
        }
    threshold = config["gates"]["g1_a"]
    criteria = {
        "score_headroom_at_10": (
            aggregate["10"]["terminal_headroom"]
            >= float(threshold["min_score_headroom_at_10"])
        ),
        "auc_headroom_at_10": aggregate["10"]["auc_headroom"] > 0.0,
        "auc_headroom_at_20": aggregate["20"]["auc_headroom"] > 0.0,
        "event_source_auc_headroom": sum(
            value["headroom"] > 0.0 for value in source_auc_headroom.values()
        )
        >= int(threshold["min_event_sources_with_auc_headroom"]),
    }
    return {
        "schema": "diva_former_stage1_headroom_audit_v2",
        "source_counts": source_counts,
        "formal_positive_jaccard": overlaps,
        "fold_rows": fold_rows,
        "event_bearing_aggregate": aggregate,
        "event_source_auc_headroom": source_auc_headroom,
        "criteria": criteria,
        "pass": bool(all(criteria.values())),
    }


def _ndcg(actual: np.ndarray, predicted: np.ndarray, count: int) -> float:
    order = np.argsort(-np.asarray(predicted), kind="mergesort")[:count]
    ideal_order = np.argsort(-np.asarray(actual), kind="mergesort")[:count]
    discount = 1.0 / np.log2(np.arange(2, len(order) + 2))
    gains = np.power(2.0, actual[order]) - 1.0
    ideal = np.power(2.0, actual[ideal_order]) - 1.0
    denominator = float(np.dot(ideal, discount))
    return float(np.dot(gains, discount) / denominator) if denominator > 0.0 else 0.0


def _bootstrap_ci(deltas: np.ndarray, seed: int = 20260912) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(5000)]
    )
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _formal_metrics(
    teacher: CounterfactualTeacherV2,
    bank: SourceBank,
    held: int,
    method: str,
    shots: int,
    seed: int,
    ndcg_k: int,
) -> dict[str, Any]:
    state = teacher.initial_state()
    rng = np.random.default_rng(seed)
    for _ in range(shots):
        available = np.flatnonzero(state.available_mask())
        if method == "diagnostic_support":
            selected = state.select_index("diagnostic")
        elif method == "random_support":
            selected = int(rng.choice(available))
        else:
            raise ValueError("only diagnostic or random support can reveal observations")
        teacher._observe_oracle(state, held, selected)
    query = np.flatnonzero(state.available_mask())
    probabilities = teacher.calibrator.predict_proba(state)[query]
    actual = bank.formal_scores[held, query]
    expected = probabilities @ np.asarray((0.0, 0.5, 1.0))
    labels = formal_class_indexes(actual)
    one_hot = np.eye(3, dtype=np.float64)[labels]
    valid_probabilities = bool(
        np.isfinite(probabilities).all()
        and np.all(probabilities >= 0.0)
        and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    )
    return {
        "held_out_source": bank.source_refs[held],
        "method": method,
        "k": shots,
        "seed": seed,
        "query_count": int(len(query)),
        "formal_ndcg_at_10": _ndcg(actual, expected, ndcg_k),
        "multiclass_brier": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "valid_probabilities": valid_probabilities,
    }


def _formal_alignment_gate(
    bank: SourceBank,
    teachers: dict[int, CounterfactualTeacherV2],
    config: dict,
) -> dict[str, Any]:
    settings = config["formal_alignment"]
    shots = int(settings["support_shots"])
    seeds = tuple(int(value) for value in settings["seeds"])
    ndcg_k = int(settings["ndcg_k"])
    event_sources = set(config["protocol"]["event_bearing_sources"])
    records = []
    for held, teacher in teachers.items():
        for seed in seeds:
            records.append(_formal_metrics(teacher, bank, held, "diagnostic_support", 0, seed, ndcg_k))
            records.append(_formal_metrics(teacher, bank, held, "diagnostic_support", shots, seed, ndcg_k))
            records.append(_formal_metrics(teacher, bank, held, "random_support", shots, seed, ndcg_k))

    def selected(method: str, k: int, *, event_only: bool) -> list[dict[str, Any]]:
        return [
            row
            for row in records
            if row["method"] == method
            and row["k"] == k
            and (not event_only or row["held_out_source"] in event_sources)
        ]

    k0_event = selected("diagnostic_support", 0, event_only=True)
    diagnostic_event = selected("diagnostic_support", shots, event_only=True)
    random_event = selected("random_support", shots, event_only=True)
    k0_all = selected("diagnostic_support", 0, event_only=False)
    diagnostic_all = selected("diagnostic_support", shots, event_only=False)
    k0_ndcg = float(np.mean([row["formal_ndcg_at_10"] for row in k0_event]))
    diagnostic_ndcg = float(
        np.mean([row["formal_ndcg_at_10"] for row in diagnostic_event])
    )
    paired = np.asarray(
        [
            diagnostic["formal_ndcg_at_10"] - random["formal_ndcg_at_10"]
            for diagnostic, random in zip(diagnostic_event, random_event)
        ],
        dtype=np.float64,
    )
    lower, upper = _bootstrap_ci(paired)
    k0_brier = float(np.mean([row["multiclass_brier"] for row in k0_all]))
    diagnostic_brier = float(
        np.mean([row["multiclass_brier"] for row in diagnostic_all])
    )
    thresholds = config["gates"]["g1_c"]
    criteria = {
        "formal_ndcg_gain_vs_k0": (
            diagnostic_ndcg - k0_ndcg
            >= float(thresholds["min_formal_ndcg_gain_vs_k0"])
        ),
        "positive_paired_ndcg_ci_vs_random": lower > 0.0,
        "brier_non_degradation": diagnostic_brier
        <= (1.0 + float(thresholds["max_brier_relative_degradation_vs_k0"]))
        * k0_brier,
        "valid_probabilities": all(row["valid_probabilities"] for row in records),
        "held_out_not_in_calibrator_fit": all(
            held not in teachers[held].calibrator.training_source_indexes for held in teachers
        ),
    }
    return {
        "schema": "diva_former_stage1_formal_alignment_gate_v2",
        "aggregates": {
            "k0_event_formal_ndcg_at_10": k0_ndcg,
            "diagnostic_k4_event_formal_ndcg_at_10": diagnostic_ndcg,
            "diagnostic_gain_vs_k0": diagnostic_ndcg - k0_ndcg,
            "diagnostic_vs_random_paired_mean": float(np.mean(paired)),
            "diagnostic_vs_random_paired_ci95": [lower, upper],
            "k0_all_multiclass_brier": k0_brier,
            "diagnostic_k4_all_multiclass_brier": diagnostic_brier,
        },
        "calibrators": {
            str(held): {
                "held_out_source": bank.source_refs[held],
                "training_source_indexes": list(teacher.calibrator.training_source_indexes),
                "class_counts": list(teacher.calibrator.class_counts),
                "weights_sha256": sha256(
                    teacher.calibrator.weights.astype("<f8").tobytes()
                ).hexdigest(),
            }
            for held, teacher in teachers.items()
        },
        "criteria": criteria,
        "pass": bool(all(criteria.values())),
        "records": records,
    }


def _diagnosis_gate(root: Path, config: dict) -> dict[str, Any]:
    reference = json.loads(
        (root / config["source_loso_reference"]).read_text(encoding="utf-8")
    )
    aggregates = reference["aggregates"]
    diagnostic = aggregates["diagnostic_support_k4"]
    k0 = aggregates["shared_prior_k0"]
    random = aggregates["random_support_k4"]
    ci = reference["paired_diagnostic_vs_random_k4"]["ci95"]
    thresholds = config["gates"]["g1_b"]
    criteria = {
        "ndcg_gain_vs_k0": diagnostic["ndcg_at_8"] - k0["ndcg_at_8"]
        >= float(thresholds["min_ndcg_gain_vs_k0"]),
        "positive_paired_ndcg_ci": float(ci[0]) > 0.0,
        "rmse_non_degradation": diagnostic["rmse"]
        <= (1.0 + float(thresholds["max_rmse_relative_degradation_vs_random"]))
        * random["rmse"],
    }
    return {
        "schema": "diva_former_stage1_diagnosis_gate_v2",
        "source_loso_schema": reference["schema"],
        "source_loso_sha256": _file_hash(root / config["source_loso_reference"]),
        "metrics": {
            "k0_ndcg_at_8": k0["ndcg_at_8"],
            "diagnostic_k4_ndcg_at_8": diagnostic["ndcg_at_8"],
            "diagnostic_k4_rmse": diagnostic["rmse"],
            "random_k4_rmse": random["rmse"],
            "diagnostic_vs_random_ndcg_ci95": ci,
        },
        "criteria": criteria,
        "pass": bool(all(criteria.values())),
    }


def _signal_rows(
    bank: SourceBank,
    teachers: dict[int, CounterfactualTeacherV2],
    config: dict,
) -> list[dict[str, Any]]:
    result = []
    for held, teacher in teachers.items():
        training = tuple(index for index in range(len(bank.source_refs)) if index != held)
        for total_budget in (10, 20):
            contexts = tuple(
                int(value)
                for value in config["teacher"][f"signal_context_sizes_b{total_budget}"]
            )
            for source_index in training:
                state = teacher.initial_state()
                for context_size in contexts:
                    while len(state.history_indexes) < context_size:
                        selected = state.select_index("diagnostic")
                        teacher._observe_oracle(state, source_index, selected)
                    remaining = total_budget - context_size
                    available = state.available_mask()
                    if remaining < 4 or not np.any(
                        bank.formal_scores[source_index, available] > 0.0
                    ):
                        continue
                    values = teacher.action_values(state, source_index, remaining)
                    formal = np.asarray(
                        [value.normalized_q_auc_formal for value in values]
                    )
                    vulnerability = np.asarray(
                        [value.normalized_q_auc_vulnerability for value in values]
                    )
                    result.append(
                        {
                            "fold": held,
                            "source_sut": bank.source_refs[source_index],
                            "total_budget": total_budget,
                            "context_size": context_size,
                            "remaining_budget": remaining,
                            "formal_p90_p10": float(
                                np.percentile(formal, 90) - np.percentile(formal, 10)
                            ),
                            "vulnerability_p90_p10": float(
                                np.percentile(vulnerability, 90)
                                - np.percentile(vulnerability, 10)
                            ),
                        }
                    )
    return result


def _teacher_gate(
    rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    config: dict,
) -> dict[str, Any]:
    event_sources = set(config["protocol"]["event_bearing_sources"])
    metrics = ("score_at_10", "auc_at_10", "score_at_20", "auc_at_20")
    aggregates = {}
    for metric in metrics:
        baseline_policy, baseline = _best_baseline(rows, metric, event_sources)
        teacher = _mean(rows, "teacher", metric, event_sources)
        aggregates[metric] = {
            "teacher": teacher,
            "best_baseline_policy": baseline_policy,
            "best_baseline": baseline,
            "absolute_gain": teacher - baseline,
            "relative_gain": (teacher - baseline) / baseline if baseline > 0.0 else None,
        }
    first_teacher = [
        row["first_critical_step"] or 21
        for row in rows
        if row["policy"] == "teacher" and row["source_sut"] in event_sources
    ]
    first_baselines = {}
    for policy in V2_BASELINE_POLICIES:
        first_baselines[policy] = float(
            np.median(
                [
                    row["first_critical_step"] or 21
                    for row in rows
                    if row["policy"] == policy and row["source_sut"] in event_sources
                ]
            )
        )
    best_first_policy, best_first = min(
        first_baselines.items(), key=lambda item: (item[1], item[0])
    )
    teacher_first = float(np.median(first_teacher))
    auc20_baseline_policy = aggregates["auc_at_20"]["best_baseline_policy"]
    fold_gains = {}
    for fold in sorted({int(row["fold"]) for row in rows}):
        teacher_values = [
            row["auc_at_20"]
            for row in rows
            if row["fold"] == fold
            and row["policy"] == "teacher"
            and row["source_sut"] in event_sources
        ]
        baseline_values = [
            row["auc_at_20"]
            for row in rows
            if row["fold"] == fold
            and row["policy"] == auc20_baseline_policy
            and row["source_sut"] in event_sources
        ]
        fold_gains[str(fold)] = float(np.mean(teacher_values) - np.mean(baseline_values))

    thresholds = config["gates"]["g1_d"]
    formal_signal = float(
        np.mean(
            [
                row["formal_p90_p10"]
                >= float(thresholds["min_formal_normalized_p90_p10"])
                for row in signal_rows
            ]
        )
    ) if signal_rows else 0.0
    vulnerability_signal = float(
        np.mean(
            [
                row["vulnerability_p90_p10"]
                >= float(thresholds["min_vulnerability_normalized_p90_p10"])
                for row in signal_rows
            ]
        )
    ) if signal_rows else 0.0
    score10 = aggregates["score_at_10"]
    relative_threshold = float(thresholds["min_relative_gain"])
    criteria = {
        "score_at_10_noninferior": score10["absolute_gain"] >= 0.0,
        "score_at_10_gain": score10["absolute_gain"]
        >= float(thresholds["min_score_absolute_gain_at_10"])
        or (score10["relative_gain"] is not None and score10["relative_gain"] >= relative_threshold),
        "auc_at_10_gain": aggregates["auc_at_10"]["relative_gain"] is not None
        and aggregates["auc_at_10"]["relative_gain"] >= relative_threshold,
        "auc_at_20_gain": aggregates["auc_at_20"]["relative_gain"] is not None
        and aggregates["auc_at_20"]["relative_gain"] >= relative_threshold,
        "score_at_20_noninferior": aggregates["score_at_20"]["absolute_gain"] >= 0.0,
        "first_critical_noninferior": teacher_first <= best_first,
        "positive_fold_consistency": sum(value > 0.0 for value in fold_gains.values())
        >= int(thresholds["min_positive_fold_count"]),
        "nondegenerate_formal_q": formal_signal
        >= float(thresholds["min_formal_signal_rate"]),
        "nondegenerate_vulnerability_q": vulnerability_signal
        >= float(thresholds["min_vulnerability_signal_rate"]),
        "fixed_budget_and_no_duplicates": all(
            len(row["selected_design_ids"]) == 20
            and len(set(row["selected_design_ids"])) == 20
            for row in rows
        ),
    }
    return {
        "schema": "diva_former_stage1_teacher_gate_v2",
        "aggregates": aggregates,
        "first_critical": {
            "teacher_median": teacher_first,
            "best_baseline_policy": best_first_policy,
            "best_baseline_median": best_first,
        },
        "fold_auc_at_20_gains": fold_gains,
        "signal_rates": {
            "formal": formal_signal,
            "vulnerability": vulnerability_signal,
        },
        "criteria": criteria,
        "pass": bool(all(criteria.values())),
        "replay_rows": rows,
        "signal_rows": signal_rows,
    }


def run(config_path: str | Path) -> dict[str, Any]:
    root, _, config, base_config, bank = _read_inputs(config_path)
    manifest = build_manifest(config_path)
    results_dir = root / config["results_dir"]
    teachers = _teachers(bank, config, base_config)
    calibrator_artifact = {
        "schema": "diva_former_stage1_formal_calibrators_v2",
        "folds": {
            str(held): {
                "held_out_source": bank.source_refs[held],
                "training_source_indexes": list(teacher.calibrator.training_source_indexes),
                "class_counts": list(teacher.calibrator.class_counts),
                "l2": teacher.calibrator.l2,
                "feature_mean": teacher.calibrator.feature_mean.tolist(),
                "feature_scale": teacher.calibrator.feature_scale.tolist(),
                "weights": teacher.calibrator.weights.tolist(),
            }
            for held, teacher in teachers.items()
        },
    }
    calibrator_path = results_dir / "formal_calibrators.json"
    calibrator_path.write_text(
        json.dumps(calibrator_artifact, indent=2) + "\n", encoding="utf-8"
    )
    budgets = tuple(int(value) for value in config["protocol"]["reporting_budgets"])
    rows = _replay_rows(bank, teachers, budgets)
    audit = _headroom_audit(bank, rows, config)
    diagnosis = _diagnosis_gate(root, config)
    formal_alignment = _formal_alignment_gate(bank, teachers, config)
    signals = _signal_rows(bank, teachers, config)
    teacher = _teacher_gate(rows, signals, config)

    artifacts = {
        "headroom_audit.json": audit,
        "diagnosis_gate.json": diagnosis,
        "formal_alignment_gate.json": formal_alignment,
        "teacher/teacher_gate.json": teacher,
    }
    for relative, payload in artifacts.items():
        output = results_dir / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest["calibrator_sha256"] = _file_hash(calibrator_path)
    manifest["artifact_sha256"] = {
        relative: _file_hash(results_dir / relative) for relative in artifacts
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    unlocked = all(
        report["pass"] for report in (audit, diagnosis, formal_alignment, teacher)
    )
    gate = {
        "schema": "diva_former_stage1_gate_v2",
        "source_only": True,
        "new_simulator_calls": 0,
        "manifest_sha256": _file_hash(results_dir / "manifest.json"),
        "g1_a_headroom": {"pass": audit["pass"], "report": "headroom_audit.json"},
        "g1_b_diagnosis": {"pass": diagnosis["pass"], "report": "diagnosis_gate.json"},
        "g1_c_formal_alignment": {
            "pass": formal_alignment["pass"],
            "report": "formal_alignment_gate.json",
        },
        "g1_d_mining_utility": {
            "pass": teacher["pass"],
            "report": "teacher/teacher_gate.json",
        },
        "transformer": {
            "trained": False,
            "status": "unlocked_not_implemented" if unlocked else "prohibited_by_v2_gate",
        },
        "decision": "ready_for_transformer_implementation" if unlocked else "stop_before_transformer",
        "failure_reasons": {
            name: [criterion for criterion, passed in report["criteria"].items() if not passed]
            for name, report in (
                ("g1_a", audit),
                ("g1_b", diagnosis),
                ("g1_c", formal_alignment),
                ("g1_d", teacher),
            )
            if not report["pass"]
        },
        "sealed_sut_refs": manifest["sealed_sut_refs"],
    }
    (results_dir / "stage1_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/metadrive/configs/diva_former_stage1_v2.yaml")
    parser.add_argument("--phase", choices=("manifest", "all"), default="all")
    args = parser.parse_args()
    if args.phase == "manifest":
        print(json.dumps(build_manifest(args.config), indent=2))
        return
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
