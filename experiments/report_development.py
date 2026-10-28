"""Lineage-weighted domain results, matched mechanism effects and the registered gate."""

from collections import defaultdict

import numpy as np

from .collision_metrics import equal_count_coverage, lineage_survival, paired_lineage_bootstrap
from .construction import load_archives
from .development import METHODS, OUTPUT, REFERENCES, validate_protocol
from .discovery_config import development_tasks, read_json, write_json
from .executions import ExecutionBank
from .protocol import ROOT, TEST_RECORDS_ROOT
from .source_registration import maintained_path
from .statistics import method_summary, summary

METRICS = ("unique_collisions", "collision_recall", "discovery_auc")


def validate_rows(rows):
    expected = {
        (t["target_id"], t["scenario"], t["algorithm_seed"], m)
        for t in development_tasks()
        for m in METHODS
    }
    actual = [(r["target_id"], r["scenario"], r["algorithm_seed"], r["method"]) for r in rows]
    if (
        len(actual) != len(expected)
        or set(actual) != expected
        or any(r["queries"] != 128 for r in rows)
    ):
        raise ValueError("Domain statistics require every exact 400-run paid-budget counterpart")
    totals = {}
    for row in rows:
        key = (row["target_id"], row["scenario"], row["algorithm_seed"])
        if totals.setdefault(key, row["pool_unique_collisions"]) != row["pool_unique_collisions"]:
            raise ValueError("Paired domain methods have different complete-pool denominators")


def mechanism_rows(rows):
    return [
        {**r, "method": "H1F1" if r["method"] == "gcrt" else r["method"]}
        for r in rows
        if r["method"] in ("gcrt", "H0F0", "H1F0", "H0F1")
    ]


def gate(rows):
    validate_rows(rows)
    stats = method_summary(rows)
    reference = max(
        REFERENCES,
        key=lambda m: (
            stats[m]["collision_recall"]["mean"]
            if stats[m]["collision_recall"]["mean"] is not None
            else -1,
            stats[m]["unique_collisions"]["mean"],
            stats[m]["discovery_auc"]["mean"],
            m,
        ),
    )
    main, other = stats["gcrt"], stats[reference]
    recall, baseline = main["collision_recall"]["mean"], other["collision_recall"]["mean"]
    reduction = (
        (recall - baseline) / (1 - baseline)
        if baseline is not None and baseline < 1 - 1e-12
        else None
    )
    gains = {
        k: value - other["collision_recall"]["by_lineage"][k]
        for k, value in main["collision_recall"]["by_lineage"].items()
    }
    headroom = [
        k for k, value in other["collision_recall"]["by_lineage"].items() if value < 1 - 1e-12
    ]
    conditions = {
        "remaining_misses_reduced_by_half": reduction is not None and reduction >= 0.5 - 1e-12,
        "collision_count_not_lower": main["unique_collisions"]["mean"]
        >= other["unique_collisions"]["mean"] - 1e-12,
        "at_least_two_positive_lineages": sum(v > 1e-12 for v in gains.values()) >= 2,
    }
    return {
        "status": "passed" if all(conditions.values()) else "not_passed",
        "main": "gcrt",
        "strongest_reference": reference,
        "reference_recall": baseline,
        "main_recall": recall,
        "remaining_miss_reduction": reduction,
        "required_recall": 0.5 * (1 + baseline) if baseline is not None else None,
        "lineage_recall_gains": gains,
        "conditions": conditions,
        "feasibility": {
            "reference_has_remaining_misses": baseline is not None and baseline < 1 - 1e-12,
            "lineages_with_headroom": headroom,
            "at_least_two_lineages_have_headroom": len(headroom) >= 2,
        },
        "reference_selection_includes_indago": True,
        "old_failed_gates_changed": False,
        "budget_or_targets_changed": False,
    }


def interaction(rows, metric):
    groups = defaultdict(dict)
    for row in mechanism_rows(rows):
        groups[tuple(row[k] for k in development_tasks()[0])][row["method"]] = row
    effects = []
    for group in groups.values():
        if any(group[cell][metric] is None for cell in ("H1F1", "H1F0", "H0F1", "H0F0")):
            continue
        effects.append(
            {
                **group["H1F1"],
                "interaction": group["H1F1"][metric]
                - group["H1F0"][metric]
                - group["H0F1"][metric]
                + group["H0F0"][metric],
            }
        )
    return summary(effects, "interaction")


def aggregate_curves(rows):
    output = {}
    for method in METHODS:
        chosen = [r for r in rows if r["method"] == method]
        result = {}
        for metric in ("discovery_curve", "recall_curve", "behavior_curve"):
            groups = defaultdict(list)
            for row in chosen:
                if row[metric][0] is not None:
                    groups[row["lineage"]].append(row[metric])
            result[metric] = (
                np.mean([np.mean(v, axis=0) for v in groups.values()], axis=0).tolist()
                if groups
                else [None] * 128
            )
        result["checkpoints"] = {
            str(b): {
                "unique_collisions": result["discovery_curve"][b - 1],
                "collision_recall": result["recall_curve"][b - 1],
            }
            for b in (16, 32, 64, 128)
        }
        output[method] = result
    return output


def construction_costs():
    banks = []
    for kind in ("cut_in", "cut_out"):
        history, development = load_archives(kind)
        for archive in (*history, *development):
            bank = ExecutionBank(
                TEST_RECORDS_ROOT / "physical" / archive.role / kind / archive.source_id
            )
            banks.append(
                {
                    "role": archive.role,
                    "scenario": kind,
                    "source_id": archive.source_id,
                    "candidates": len(bank.candidates),
                    "successful_calls": bank.manifest["successful_calls"],
                    "attempted_calls": bank.manifest["attempted_calls"],
                    "failed_or_interrupted_calls": bank.manifest["failed_or_interrupted_calls"],
                    "failed_call_wall_s": bank.manifest["failed_call_wall_s"],
                    "interrupted_call_duration_unknown": bank.manifest[
                        "interrupted_call_duration_unknown"
                    ],
                    "physical_execution_s": bank.manifest["physical_execution_s"],
                }
            )
    main = read_json(
        ROOT
        / "experiment_outputs/gcrt/collision/discovery/domain_comparison/models/construction.json"
    )
    indago = read_json(
        ROOT / "experiment_outputs/indago/collision/discovery/domain_comparison/stage_a/audit.json"
    )
    graphs = read_json(
        ROOT
        / "experiment_outputs/scenario_fuzz/collision/discovery/domain_comparison/stage_a/training.json"
    )
    return {
        "reused_physical_banks": banks,
        "new_physical_calls": 0,
        "model_training": {
            "gcrt": [
                {
                    "scenario": r["scenario"],
                    "target_kernel_s": r["target_kernel"]["elapsed_s"],
                    "historical_response_encoder_s": r["historical_response_encoder"]["wall_s"],
                    "historical_posterior_construction_s": r["source_priors"]["construction_s"],
                }
                for r in main["tasks"]
            ],
            "indago": [
                {"scenario": r["scenario"], "training_s": r["training_s"]}
                for r in indago["training"]
            ],
            "scenario_fuzz": [
                {"scenario": r["scenario"], "training_s": r["training_s"]} for r in graphs["tasks"]
            ],
        },
        "paired_proxy": "historical kernel predictions, constructed separately per run; no new proxy executions",
        "timing_scope": "actual measured wall times under the recorded worker count; replay retains original simulation cost",
    }


def report():
    specification = read_json(OUTPUT / "protocol.json")
    validate_protocol(specification)
    audit = read_json(OUTPUT / "audit.json")
    index = read_json(OUTPUT / "run_index.json")
    if audit["status"] != "passed" or audit["runs"] != 400 or index["status"] != "complete":
        raise ValueError("Complete and independently audit all domain runs before reporting")
    rows = index["runs"]
    decision = gate(rows)
    stats = method_summary(rows)
    paired = []
    mechanisms = mechanism_rows(rows)
    for compared in REFERENCES:
        for metric in METRICS:
            available = [
                r for r in rows if r["method"] in ("gcrt", compared) and r[metric] is not None
            ]
            if available:
                paired.append(paired_lineage_bootstrap(available, "gcrt", compared, metric))
    complementarity = []
    for control in ("H0F1", "H1F0", "H0F0"):
        for metric in METRICS:
            available = [
                r for r in mechanisms if r["method"] in ("H1F1", control) and r[metric] is not None
            ]
            if available:
                complementarity.append(paired_lineage_bootstrap(available, "H1F1", control, metric))
    coverage = []
    for task in development_tasks():
        chosen = [
            r
            for r in rows
            if all(r[k] == v for k, v in task.items()) and r["method"] in ("gcrt", *REFERENCES)
        ]
        coverage.append(
            {"task": task, "methods": [r["method"] for r in chosen], **equal_count_coverage(chosen)}
        )
    result = {
        "status": "complete_audited_development_comparison",
        "gate": decision,
        "summary": stats,
        "comparisons": paired,
        "mechanism_comparisons": complementarity,
        "interaction": {metric: interaction(rows, metric) for metric in METRICS},
        "mechanism_overlap": {
            "gcrt_is_H1F1": True,
            "identical_underlying_run_files": 40,
            "unique_runs": 400,
            "equivalent_queries": 51200,
        },
        "equal_collision_count_behavior_coverage": coverage,
        "first_collision_survival": {
            m: lineage_survival([r for r in rows if r["method"] == m], restriction=128)
            for m in METHODS
        },
        "paid_safety_measurements": safety_summaries(rows),
        "cost": construction_costs(),
        "new_independent_targets_evaluated": 0,
        "next_stage": "C method finalization"
        if decision["status"] == "passed"
        else "retain B/C evidence; D/E not entered",
    }
    write_json(OUTPUT / "gate.json", decision)
    write_json(OUTPUT / "report.json", result)
    curves = aggregate_curves(rows)
    write_json(OUTPUT / "curves.json", curves)
    plot(curves)
    reference = decision["strongest_reference"]
    lines = [
        "# 领域方法开发比较与历史/反馈机制对照",
        "",
        "400项独立运行、51,200次等效查询；全部执行账本、完整库分母和主方法独立选例核查通过。新增真实仿真调用0。",
        "",
        "| 方法 | 去重碰撞数 | 有限库召回 | 发现AUC |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        values = [stats[method][metric]["mean"] for metric in METRICS]
        formatted = ["不适用" if v is None else f"{v:.6f}" for v in values]
        lines.append(f"| {method} | " + " | ".join(formatted) + " |")
    reduction = decision["remaining_miss_reduction"]
    reduction_text = "不适用" if reduction is None else f"{100 * reduction:.2f}%"
    lines += [
        "",
        f"最强参考为{reference}；主方法剩余漏检减少{reduction_text}，研发扩大门槛{decision['status']}。",
        f"谱系召回变化：{decision['lineage_recall_gains']}。",
        "",
        "H1F1直接复用主方法的40份相同账本。H0为同信息高斯汇总教师；F0不使用初始化或后续标签更新。"
        "主要机制对比、2,000次配对谱系自助区间及交互量见report.json。交互量不作为必须超加性的条件。",
        "",
        "发现曲线见development_curves.png/svg/pdf；逐次曲线和检查点见curves.json。"
        "成本单列历史/开发物理库、历史训练、后验构建、在线适配、选例与等效执行时间。",
        "",
        result["next_stage"],
        "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(decision, flush=True)
    return result


def safety_summaries(rows):
    output = {}
    for method in METHODS:
        measurements = defaultdict(list)
        for row in rows:
            if row["method"] == method:
                run = read_json(ROOT / maintained_path(row["path"]))
                for query in run["result"]["queries"]:
                    for name in (
                        "impact_delta_v_mps",
                        "min_clearance_m",
                        "min_ttc_s",
                        "driving_quality_score",
                    ):
                        value = query["feedback"]["safety"][name]
                        if value is not None:
                            measurements[name].append(value)
        output[method] = {
            name: {
                "available_paid_values": len(values),
                "mean": float(np.mean(values)),
                "quantiles_10_50_90": np.quantile(values, [0.1, 0.5, 0.9]).tolist(),
            }
            for name, values in measurements.items()
        }
    return output


def plot(curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method in ("gcrt", *REFERENCES):
        for axis, metric in zip(axes, ("discovery_curve", "recall_curve")):
            values = curves[method][metric]
            if values[0] is not None:
                axis.plot(np.arange(1, 129), values, label=method)
    axes[0].set(xlabel="Paid queries", ylabel="Unique collision groups")
    axes[1].set(xlabel="Paid queries", ylabel="Recall over complete finite pool", ylim=(0, 1.01))
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    for extension in ("png", "svg", "pdf"):
        figure.savefig(OUTPUT / f"development_curves.{extension}", dpi=180)
    plt.close(figure)
