"""Count/recall comparisons and evaluator-only mechanism diagnostics."""

import numpy as np

from methods.common.catalog import NAMES
from methods.baselines.dart.evaluation.config import SUT_IDS
from methods.baselines.hgt.io import read_json, write_json

from ..config import ROOT, SEEDS, TRAIN_MODES
from .evaluate import bank_path, cases, case_variants, trace_path

METHOD_LABELS = {
    "cg_brt": NAMES["cg_brt"],
    "previous": NAMES["dart"],
    "full": NAMES["lrt"],
    "global_sources": "LRT：全局来源状态",
    "no_archive_covariance": "LRT：无档案协方差",
    "no_target_component": "LRT：无目标独立分量",
    "no_value_learning": "LRT：一步前瞻",
    "nominal_guard": "LRT：单模型损失约束",
    "unprotected": "LRT：无损失预算",
}


def measured(pool, sut, seed, condition, variant):
    trace = read_json(trace_path(pool, sut, seed, variant, condition))
    with np.load(bank_path(pool, sut)) as bank:
        risk = bank["risk"]
        family = bank["x"][:, 4].astype(int)
    selected = trace["selected_indices"]
    if len(selected) != 200 or len(set(selected)) != 200:
        raise ValueError("Every trajectory must contain 200 unique paid observations")
    for index, query in zip(selected, trace["queries"]):
        if index != query["index"] or float(risk[index]) != query["risk"]:
            raise ValueError("Recorded query differs from evaluator response")
    high = risk > 0.5
    found = int(np.count_nonzero(high[selected]))
    replacements = (
        []
        if variant == "cg_brt"
        else [
            d for d in trace["decisions"] if d["selected_index"] != d["reference_index"]
        ]
    )
    return {
        "pool": pool,
        "sut": sut,
        "seed": seed,
        "condition": condition,
        "variant": variant,
        "H200": found,
        "high_risk_recall_pct": 100 * found / np.count_nonzero(high),
        "replacements": len(replacements),
        "actual_same_history_loss": sum(
            int(high[d["reference_index"]] and not high[d["selected_index"]])
            for d in replacements
        ),
        "predicted_loss": sum(
            d.get("posterior_replacement_loss", 0) for d in replacements
        ),
        "charged_loss": sum(d.get("loss_charged", 0) for d in trace["decisions"]),
        "rejected_candidates": (
            sum(d.get("rejected_candidates", 0) for d in trace["decisions"])
            if variant != "cg_brt"
            else 0
        ),
        "rounds_with_rejection": (
            sum(d.get("rejected_candidates", 0) > 0 for d in trace["decisions"])
            if variant != "cg_brt"
            else 0
        ),
        "runtime_s": trace.get("selector_elapsed_s", 0),
        "by_family": {
            str(c): {
                "queries": int(np.count_nonzero(family[selected] == c)),
                "high_risk_found": int(
                    np.count_nonzero(high[selected] & (family[selected] == c))
                ),
                "high_risk_recall_pct": float(
                    100
                    * np.count_nonzero(high[selected] & (family[selected] == c))
                    / max(1, np.count_nonzero(high & (family == c)))
                ),
            }
            for c in (0, 1)
        },
    }


def summary(rows):
    metrics = (
        "H200",
        "high_risk_recall_pct",
        "replacements",
        "actual_same_history_loss",
        "predicted_loss",
        "charged_loss",
        "rejected_candidates",
        "rounds_with_rejection",
        "runtime_s",
    )
    names = tuple(dict.fromkeys(r["variant"] for r in rows))
    aggregate = {
        v: {
            k: float(np.mean([r[k] for r in rows if r["variant"] == v]))
            for k in metrics
        }
        for v in names
    }
    by_sut = {
        s: {
            v: {
                k: float(
                    np.mean([r[k] for r in rows if r["variant"] == v and r["sut"] == s])
                )
                for k in metrics[:2]
            }
            for v in names
        }
        for s in SUT_IDS
    }
    retention = {k: aggregate["full"][k] / aggregate["cg_brt"][k] for k in metrics[:2]}
    delta = np.array(
        [
            [by_sut[s]["full"][k] - by_sut[s]["cg_brt"][k] for k in metrics[:2]]
            for s in SUT_IDS
        ]
    )
    rng = np.random.default_rng(980031)
    bootstrap = delta[rng.integers(0, len(SUT_IDS), (10000, len(SUT_IDS)))].mean(1)
    return {
        "aggregate": aggregate,
        "by_sut": by_sut,
        "retention": retention,
        "effect_accepted": all(v >= 0.9 for v in retention.values()),
        "activity_accepted": aggregate["full"]["replacements"] >= 10,
        "paired_delta_95pct_system_block_bootstrap": {
            k: np.quantile(bootstrap[:, j], [0.025, 0.975]).tolist()
            for j, k in enumerate(metrics[:2])
        },
    }


def query_changes(rows):
    """Compare realized trajectories; this does not isolate causal mediation."""
    comparisons = []
    for row in rows:
        if row["variant"] == "full":
            continue
        pool, sut, seed, condition = (
            row[k] for k in ("pool", "sut", "seed", "condition")
        )
        full = read_json(trace_path(pool, sut, seed, "full", condition))[
            "selected_indices"
        ]
        other = read_json(trace_path(pool, sut, seed, row["variant"], condition))[
            "selected_indices"
        ]
        comparisons.append(
            {
                "pool": pool,
                "sut": sut,
                "seed": seed,
                "condition": condition,
                "comparison": row["variant"],
                "different_query_positions": sum(a != b for a, b in zip(full, other)),
                "new_unique_queries": len(set(full) - set(other)),
                "query_set_jaccard": len(set(full) & set(other))
                / len(set(full) | set(other)),
            }
        )
    return comparisons


def checkpoint_summary(mode, seed):
    report = read_json(ROOT / "models" / f"training_{mode}_{seed}.json")
    selected = [row for row in report["curve"] if row["selected"]][-1]
    return {
        "training_status": report["status"],
        "executed_steps": report["executed_steps"],
        "selected_step": selected["step"],
        "validation": selected["validation"],
        "eligible_checkpoint_found": report["eligible_checkpoint_found"],
    }


def main():
    rows = [
        measured(pool, sut, seed, condition, variant)
        for pool, seed, condition in cases()
        for sut in SUT_IDS
        for variant in case_variants(pool, condition)
    ]
    sections = {
        "original_pools": [
            r for r in rows if r["condition"] == "complete" and r["pool"] != 2
        ],
        "third_existing_pool": [
            r for r in rows if r["condition"] == "complete" and r["pool"] == 2
        ],
    }
    for condition in dict.fromkeys(
        r["condition"] for r in rows if r["condition"] != "complete"
    ):
        sections[condition] = [r for r in rows if r["condition"] == condition]
    result = {
        "method_names": METHOD_LABELS,
        "sections": {name: summary(data) for name, data in sections.items()},
        "rows": rows,
        "metrics": ["H200", "high_risk_recall_pct"],
        "physical_calls": 0,
        "all_pools_previously_observed": True,
        "target_data_used_for_weight_selection": False,
        "query_set_changes": query_changes(rows),
        "historical_checkpoint_selection": {
            mode: {str(seed): checkpoint_summary(mode, seed) for seed in SEEDS}
            for mode in TRAIN_MODES
        },
        "continuation_audit": read_json(ROOT / "diagnostics.json")["summary"],
    }
    main_replacements = float(
        np.mean(
            [
                row["replacements"]
                for row in rows
                if row["condition"] == "complete" and row["variant"] == "full"
            ]
        )
    )
    result["acceptance"] = {
        "effect_accepted": all(
            result["sections"][name]["effect_accepted"]
            for name in ("original_pools", "third_existing_pool")
        ),
        "mean_main_trajectory_replacements": main_replacements,
        "activity_accepted": main_replacements >= 10,
        "mechanism_contribution_requires_joint_evidence": True,
        "stress_results_are_diagnostic": True,
    }
    write_json(ROOT / "comparison.json", result)
    lines = [
        "# 局部风险迁移方法的实验比较",
        "",
        "主评价仅包含 H200 与危险场景 Recall。所有目标池均为已有池。",
        "正式效果门槛分别适用于原有两池合并结果及第三池结果；档案失配条件用于机制诊断。",
        f"主实验效果门槛通过：{result['acceptance']['effect_accepted']}；"
        f"完整方法平均每条主轨迹偏离 {main_replacements:.2f} 次。",
        "",
        "![发现数量与覆盖率](figures/discovery_outcomes.png)",
        "",
        "虚线表示同池 CG-BRT 均值的 90%，不是置信区间。配对区间见下表后的统计结果。",
        "",
    ]
    for name, section in result["sections"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                "| 方法 | H200 | Recall % | 替换数 | 实际同历史损失 | 预测损失 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for variant, score in section["aggregate"].items():
            lines.append(
                f"| {METHOD_LABELS[variant]} | {score['H200']:.3f} | {score['high_risk_recall_pct']:.3f} | "
                f"{score['replacements']:.2f} | {score['actual_same_history_loss']:.2f} | {score['predicted_loss']:.3f} |"
            )
        lines.extend(
            [
                "",
                (
                    f"效果门槛通过：{section['effect_accepted']}。"
                    if name in ("original_pools", "third_existing_pool")
                    else f"相对同条件 CG-BRT 的保留比例：{section['retention']}（机制诊断）。"
                ),
                "",
            ]
        )
        if name in ("original_pools", "third_existing_pool"):
            lines.extend(
                [
                    "| 系统 | CG-BRT H200 / Recall % | DART H200 / Recall % | LRT H200 / Recall % |",
                    "|---|---:|---:|---:|",
                ]
            )
            for sut, scores in section["by_sut"].items():
                cells = [
                    f"{scores[v]['H200']:.2f} / {scores[v]['high_risk_recall_pct']:.2f}"
                    for v in ("cg_brt", "previous", "full")
                ]
                lines.append(f"| {sut} | " + " | ".join(cells) + " |")
            intervals = section["paired_delta_95pct_system_block_bootstrap"]
            lines.extend(
                [
                    "",
                    f"H200 保留比例 {section['retention']['H200']:.4f}；Recall 保留比例 "
                    f"{section['retention']['high_risk_recall_pct']:.4f}。",
                    f"按系统成块重采样的配对差值 95% 区间：H200 {intervals['H200']}，"
                    f"Recall 百分点 {intervals['high_risk_recall_pct']}。",
                    "",
                ]
            )
    lines.extend(
        [
            "## 查询集合变化",
            "",
            "原有两池中，完整方法相对各对照平均增加的不同场景数。",
            "",
            "| 对照 | 不同查询位置数 | 新增唯一场景数 | 集合 Jaccard |",
            "|---|---:|---:|---:|",
        ]
    )
    changed = [
        row
        for row in result["query_set_changes"]
        if row["condition"] == "complete" and row["pool"] != 2
    ]
    for variant in dict.fromkeys(row["comparison"] for row in changed):
        selected = [row for row in changed if row["comparison"] == variant]
        values = [
            np.mean([row[k] for row in selected])
            for k in (
                "different_query_positions",
                "new_unique_queries",
                "query_set_jaccard",
            )
        ]
        lines.append(
            f"| {METHOD_LABELS[variant]} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 后续累计收益审计",
            "",
            "评价端通过参照策略分支回放，比较决策相关风险迁移的一步前瞻收益差和局部风险迁移的动作价值预测。",
            "",
            "![后续收益预测与评价端回放](figures/continuation_audit.png)",
            "",
            "| 方法 | 审计到的偏离数 | 收益差 MSE | 实际平均收益差 | 收益差相关系数 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant, audit in result["continuation_audit"].items():
        lines.append(
            f"| {METHOD_LABELS[variant]} | {audit['audited_deviations']} | {audit['gain_mse']} | "
            f"{audit['mean_realized_gain']} | {audit['gain_pearson_correlation']} |"
        )
    lines.extend(
        [
            "",
            "累计损失预算控制模型内的同历史替换事件；完整轨迹收益通过动作价值误差分析联系。",
            "动作价值审计、少样本诊断和匹配消融共同解释各组件的作用，理论关系见[方法说明](../../docs/method.md)。",
            "",
        ]
    )
    (ROOT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    from .figures import main as figures

    figures(result)
    return result


if __name__ == "__main__":
    main()
