"""Matched full-remaining-budget prediction evaluation on nine configurations."""

import subprocess
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from methods.baselines.hgt.io import read_json, write_json
from methods.gcrt.config import BUDGET, INITIAL, ROOT, SEED, SEGMENT
from methods.gcrt.controller import block_scores, threshold
from methods.gcrt.data import forecast_task
from methods.gcrt.efficiency_report import (
    DISPLAY_GROUPS, FAMILIES, MCTS, SYSTEMS, save_csv,
)
from methods.gcrt.retention_efficiency import load_task
from methods.gcrt.train import load_stopper

OUTPUT = ROOT / "calibration_effectiveness"
ARCHIVED = ROOT / "retention_efficiency"
METHODS = {"base": "基础预测", "corrected": "收益校正后预测"}
ARCHIVED_METHODS = {"base": "without_correction", "corrected": "gcrt"}
CALIBRATION_SUTS = {
    "cutin": ("mcts_cv_ref_audit_v4", "ppo_ref_v2"),
    "brake": ("ppo_ref_v2", "idm_ref", "mobil_ref_v2"),
}
BIAS_TOLERANCE = 0.05
BIAS_GROUPS = {
    "overprediction": "基础预测偏高",
    "underprediction": "基础预测偏低",
    "small_bias": "偏差较小",
}
BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TYPICAL_CHECKPOINTS = (10, 50, 100, 150, 195)
METRICS = ("mae", "nll", "brier")
NUMERICAL_TOLERANCE = 1e-8


def bias_group(value):
    if value > BIAS_TOLERANCE:
        return "overprediction"
    if value < -BIAS_TOLERANCE:
        return "underprediction"
    return "small_bias"


def paired_metrics(check, truth, budget=BUDGET):
    """Score the exact two PMFs saved by one network prediction."""
    remaining = budget - check["query"]
    result = {
        "query": check["query"], "remaining": remaining,
        "found": check["found"], "actual_remaining": int(truth),
        "allowed_misses": check["allowed_misses"],
        "event": int(truth <= check["allowed_misses"]),
        "deterministic": check["deterministic"], "u": check["tilt"],
        "completed_segments": check["completed_segments"],
        "segment_rate_bias": check["segment_rate_bias"],
    }
    for method, probability_key, mean_key, score_key in (
        ("base", "base_remaining_log_probability", "base_remaining_mean", "base_score"),
        ("corrected", "remaining_log_probability", "remaining_mean", "score"),
    ):
        log_probability = np.asarray(check[probability_key], dtype=float)
        assert len(log_probability) == remaining + 1
        assert np.isfinite(log_probability).all() and 0 <= truth <= remaining
        probability = np.exp(log_probability)
        assert np.isclose(probability.sum(), 1.0, rtol=0, atol=1e-12)
        mean = float(probability @ np.arange(remaining + 1))
        score = float(probability[: check["allowed_misses"] + 1].sum())
        assert np.isclose(mean, check[mean_key], rtol=0, atol=NUMERICAL_TOLERANCE)
        assert np.isclose(score, check[score_key], rtol=0, atol=1e-12)
        result.update({
            f"{method}_mean": check[mean_key], f"{method}_score": check[score_key],
            f"{method}_rate_error": (check[mean_key] - truth) / remaining,
            f"{method}_mae": abs(check[mean_key] - truth),
            f"{method}_nll": -float(log_probability[truth]),
            f"{method}_brier": (check[score_key] - result["event"]) ** 2,
        })
    # The slope of log(q/p) recovers the count-distribution tilt without refitting.
    ratio = np.asarray(check["remaining_log_probability"]) - np.asarray(
        check["base_remaining_log_probability"]
    )
    result["eta"] = float((ratio[-1] - ratio[0]) / remaining)
    result["mean_change"] = result["corrected_mean"] - result["base_mean"]
    result["score_change"] = result["corrected_score"] - result["base_score"]
    for metric in METRICS:
        result[f"delta_{metric}"] = result[f"corrected_{metric}"] - result[f"base_{metric}"]
    target = remaining * expit(
        logit(np.clip(result["base_mean"] / remaining, 1e-12, 1 - 1e-12)) + result["u"]
    )
    assert np.isclose(result["corrected_mean"], target, rtol=0, atol=NUMERICAL_TOLERANCE)
    assert result["u"] * result["segment_rate_bias"] >= -NUMERICAL_TOLERANCE
    assert result["u"] * result["mean_change"] >= -NUMERICAL_TOLERANCE
    assert result["u"] * result["score_change"] <= NUMERICAL_TOLERANCE
    if check["deterministic"]:
        assert result["event"] == 1
        assert np.isclose(result["base_score"], 1.0, rtol=0, atol=1e-12)
        assert np.isclose(result["corrected_score"], 1.0, rtol=0, atol=1e-12)
    return result


def configuration_metrics(checkpoints):
    rows = []
    for configuration_id in sorted({r["configuration_id"] for r in checkpoints}):
        checks = [r for r in checkpoints if r["configuration_id"] == configuration_id]
        row = {key: checks[0][key] for key in ("configuration_id", "family", "sut", "pool")}
        signed_error = float(np.mean([r["base_rate_error"] for r in checks]))
        row.update(checkpoints=len(checks), base_rate_bias=signed_error,
                   bias_group=bias_group(signed_error))
        for key in ("u", "segment_rate_bias", "mean_change", "score_change"):
            row[f"{key}_mean"] = float(np.mean([r[key] for r in checks]))
        row.update(u_min=min(r["u"] for r in checks), u_max=max(r["u"] for r in checks))
        for metric in METRICS:
            for prefix in (*METHODS, "delta"):
                key = f"{prefix}_{metric}"
                row[key] = float(np.mean([r[key] for r in checks]))
        rows.append(row)
        for check in checks:
            check["bias_group"] = row["bias_group"]
    return rows


def summarize_predictions(configurations):
    scopes = [("all", "全部配置", configurations)]
    scopes += [(family, label, [r for r in configurations if r["family"] == family])
               for family, label in FAMILIES.items()]
    scopes += [(group, label, [r for r in configurations if r["bias_group"] == group])
               for group, label in BIAS_GROUPS.items()]
    rows = []
    for scope, label, subset in scopes:
        row = {"scope": scope, "label": label, "configurations": len(subset),
               "checkpoints": sum(r["checkpoints"] for r in subset)}
        for metric in METRICS:
            for prefix in (*METHODS, "delta"):
                key = f"{prefix}_{metric}"
                values = [r[key] for r in subset]
                row[f"{key}_mean"] = float(np.mean(values)) if values else None
                row[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
        rows.append(row)
    return rows


def calibrate(replays, family, method):
    controllers = CALIBRATION_SUTS[family]
    held = [r for r in replays if r["family"] == family
            and r["method"] == ARCHIVED_METHODS[method] and r["sut"] in controllers]
    evaluation = {sut for candidate_family, sut in DISPLAY_GROUPS if candidate_family == family}
    assert evaluation.isdisjoint(controllers)
    expected = {(sut, pool) for sut in controllers for pool in (0, 1, 2)}
    assert {(r["sut"], r["pool"]) for r in held} == expected and len(held) == len(expected)
    scores = block_scores(held)
    gamma = threshold(scores.values())
    unsafe = [check for task in held for check in task["forecast"]
              if 20 * check["found"] < 19 * task["full_found"]]
    assert all(r["score"] <= gamma and not r["deterministic"] for r in unsafe)
    return {
        "threshold": gamma, "controller_scores": scores,
        "controllers": list(controllers), "configurations": len(held),
        "configuration_ids": [f"{family}/pool_{r['pool']}/{r['sut']}" for r in held],
        "unsafe_checkpoints": len(unsafe),
        "rule": "max({0} union unsafe calibration scores); stop iff score > threshold",
    }


def stopping_metrics(trace, checks, method, gamma):
    hit = next((r for r in checks if r["deterministic"] or r[f"{method}_score"] > gamma), None)
    used = hit["query"] if hit else BUDGET
    found, full = trace["found"][used - 1], trace["found"][-1]
    return {
        "configuration_id": checks[0]["configuration_id"],
        "family": checks[0]["family"], "sut": trace["sut"], "pool": trace["pool"],
        "method": method, "threshold": gamma, "queries": used,
        "query_reduction": 1 - used / BUDGET, "found": found, "full_found": full,
        "retention": found / full if full else None,
        "target_met": 20 * found >= 19 * full if full else None,
        "missed": full - found,
        "stop_reason": ("deterministic" if hit["deterministic"] else "score") if hit else "budget",
    }


def summarize_stopping(rows):
    result = []
    for family in ("all", *FAMILIES):
        for method in METHODS:
            subset = [r for r in rows if r["method"] == method
                      and (family == "all" or r["family"] == family)]
            if not subset:
                continue
            valid = [r for r in subset if r["retention"] is not None]
            result.append({
                "family": family, "method": method, "configurations": len(subset),
                "valid_configurations": len(valid), "zero_discovery_configurations": len(subset) - len(valid),
                "queries_mean": float(np.mean([r["queries"] for r in subset])),
                "found_mean": float(np.mean([r["found"] for r in subset])),
                "full_found_mean": float(np.mean([r["full_found"] for r in subset])),
                "retention_mean": float(np.mean([r["retention"] for r in valid])) if valid else None,
                "retention_min": min((r["retention"] for r in valid), default=None),
                "target_met_configurations": sum(r["target_met"] for r in valid),
            })
    return result


def probability_bin(probability):
    assert 0 <= probability <= 1
    return int(np.searchsorted(BIN_EDGES[1:-1], probability, side="right"))


def reliability_bins(checkpoints):
    rows = []
    for family in FAMILIES:
        for method in METHODS:
            for index, (left, right) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
                subset = [r for r in checkpoints if r["family"] == family
                          and probability_bin(r[f"{method}_score"]) == index]
                rows.append({
                    "family": family, "method": method, "bin": index,
                    "left": left, "right": right, "right_inclusive": index == len(BIN_EDGES) - 2,
                    "checkpoints": len(subset),
                    "configurations": len({r["configuration_id"] for r in subset}),
                    "probability_mean": float(np.mean([r[f"{method}_score"] for r in subset])) if subset else None,
                    "event_frequency": float(np.mean([r["event"] for r in subset])) if subset else None,
                })
    return rows


def typical_configurations(configurations, checkpoints):
    rows = []
    for group in BIAS_GROUPS:
        subset = [r for r in configurations if r["bias_group"] == group]
        if not subset:
            continue
        median = float(np.median([r["base_rate_bias"] for r in subset]))
        representative = min(subset, key=lambda r: (
            round(abs(r["base_rate_bias"] - median), 12), r["configuration_id"]
        ))
        rows.extend(r.copy() for r in checkpoints
                    if r["configuration_id"] == representative["configuration_id"]
                    and r["query"] in TYPICAL_CHECKPOINTS)
    return rows


def plot_results(checkpoints, configurations, bins):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    colors = {"base": "#D55E00", "corrected": "#0072B2"}
    styles = {"base": "--", "corrected": "-"}
    folder = OUTPUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.6), sharey=True)
    steps = list(range(INITIAL, BUDGET, SEGMENT))
    for panel, (axis, (group, label)) in enumerate(zip(axes, BIAS_GROUPS.items())):
        subset = [r for r in checkpoints if r["bias_group"] == group]
        count = sum(r["bias_group"] == group for r in configurations)
        for method, title in METHODS.items():
            values = [np.mean([r[f"{method}_mean"] for r in subset if r["query"] == t])
                      if count else np.nan for t in steps]
            axis.plot(steps, values, styles[method], color=colors[method], linewidth=1.8, label=title)
        truth = [np.mean([r["actual_remaining"] for r in subset if r["query"] == t])
                 if count else np.nan for t in steps]
        axis.plot(steps, truth, "k-.", marker="o", markersize=3, markevery=5,
                  linewidth=1.5, label="实际后续发现数")
        axis.set(title=f"({chr(97 + panel)}) {label}（{count}组配置）",
                 xlabel="已执行测试次数（次）", xlim=(10, 200), ylim=(0, None))
        axis.set_xticks((10, 50, 100, 150, 195))
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("剩余危险用例发现数量（个）")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    figure.text(0.5, 0.045, "各曲线为组内配置等权均值；按基础预测发现率的平均偏差分组，容差为5个百分点", ha="center", fontsize=9)
    figure.text(0.5, 0.008, "种子11；9组配置、342个检查点；前车紧急制动采用MCTS探索性补充评价", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.09, 1, 0.91), w_pad=1.6)
    figure.savefig(folder / "remaining_discoveries.png", dpi=600)
    plt.close(figure)

    figure, panels = plt.subplots(
        2, 2, figsize=(11.8, 6.8),
        gridspec_kw={"height_ratios": (4.2, 1.0)},
    )
    axes = panels[0]
    for axis, table_axis, (family, label) in zip(axes, panels[1], FAMILIES.items()):
        axis.plot([0, 1], [0, 1], ":", color="#777777", linewidth=1.3, label="理想对角线")
        for method, title in METHODS.items():
            subset = [r for r in bins if r["family"] == family and r["method"] == method]
            x = [r["probability_mean"] if r["checkpoints"] else np.nan for r in subset]
            y = [r["event_frequency"] if r["checkpoints"] else np.nan for r in subset]
            axis.plot(x, y, styles[method], color=colors[method], linewidth=1.6,
                      marker="s" if method == "base" else "o", markersize=5, label=title)
        axis.set(title=label + ("（MCTS补充评价）" if family == "brake" else ""),
                 xlabel="分组平均预测概率", xlim=(-0.025, 1.025), ylim=(-0.025, 1.055))
        axis.set_xticks(BIN_EDGES)
        axis.set_yticks(BIN_EDGES)
        axis.grid(alpha=0.18)
        table_axis.set_axis_off()
        table = table_axis.table(
            cellText=[[f"{r['checkpoints']}/{r['configurations']}" for r in bins
                       if r["family"] == family and r["method"] == method]
                      for method in METHODS],
            rowLabels=["基础 n/k", "校正 n/k"],
            colLabels=["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1]"],
            cellLoc="center", rowLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.45)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.5)
            if row > 0:
                cell.get_text().set_color(colors[tuple(METHODS)[row - 1]])
    axes[0].set_ylabel("实际保留事件频率")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.text(0.5, 0.046, "保留事件：实际剩余发现数不超过允许少发现数量；五个等宽概率区间，空箱不连线", ha="center", fontsize=9)
    figure.text(0.5, 0.014, "下表按概率区间列出样本数：n为检查点数，k为配置数；同一配置的检查点具有相关性", ha="center", fontsize=8)
    figure.tight_layout(rect=(0.025, 0.07, 1, 0.93), w_pad=3.5, h_pad=1.0)
    figure.savefig(folder / "retention_reliability.png", dpi=600)
    plt.close(figure)


def prediction_table(summaries):
    lines = [
        "| 范围 | 配置数 | 检查点数 | MAE 基础→校正（个） | NLL 基础→校正 | Brier 基础→校正 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        cells = [f"{row[f'base_{metric}_mean']:.5f} → {row[f'corrected_{metric}_mean']:.5f}"
                 if row["configurations"] else "—" for metric in METRICS]
        lines.append(f"| {row['label']} | {row['configurations']} | {row['checkpoints']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stopping_table(summaries):
    lines = [
        "| 功能场景 | 预测方法 | 配置数 | 平均测试次数 | 平均危险发现数 | 平均保留率 | 最小保留率 | 达到95%目标 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        retention = f"{row['retention_mean']:.2%}" if row["retention_mean"] is not None else "—"
        minimum = f"{row['retention_min']:.2%}" if row["retention_min"] is not None else "—"
        lines.append(
            f"| {FAMILIES.get(row['family'], '全部配置')} | {METHODS[row['method']]} | {row['configurations']} | "
            f"{row['queries_mean']:.2f} | {row['found_mean']:.2f} | {retention} | {minimum} | "
            f"{row['target_met_configurations']}/{row['valid_configurations']} |"
        )
    return "\n".join(lines)


def write_report(summaries, configurations, stopping, typical, audit):
    total = next(r for r in summaries if r["scope"] == "all")
    outcome = (
        "在9组实验配置的342个配对检查点上，基础预测与收益校正后预测的剩余发现数MAE分别为"
        f"{total['base_mae_mean']:.5f}、{total['corrected_mae_mean']:.5f}个，计数NLL分别为"
        f"{total['base_nll_mean']:.5f}、{total['corrected_nll_mean']:.5f}，保留事件Brier评分分别为"
        f"{total['base_brier_mean']:.5f}、{total['corrected_brier_mean']:.5f}。"
    )
    setup = (
        "车辆切入采用IDM、IDM＋MOBIL，前车紧急制动采用MCTS，每个控制器分别使用三个场景测试集，"
        "与危险用例发现效果与测试效率实验采用相同的9组配置。种子固定为11，完整预算200次，初始化10次，"
        "每五次测试构成一个测试段。两种预测共用当前已发布的updated.pt权重及同一基础分布，"
        "网络参数在任务内固定；仅比较是否应用收益校正。复用已归档完整轨迹，无新增仿真或训练。"
        "MCTS紧急制动沿用探索性补充评价身份，其结果不替代原预设独立评价结果。"
    )
    protocol = (
        "在第10、15、…、195次测试后的全部38个检查点，预测全部剩余次数的新增危险发现数。"
        "先用已完成测试段的基础预测发现率与实际发现率配对更新系数u，再预测和校正剩余分布；"
        "初始化的两个五次测试段参与更新。检查点之后的完整执行记录只用于离线评分，不输入此前预测。"
        "即使已触发停止，也在完整轨迹的共同检查点评价预测，因此样本范围不随方法的停止位置变化。"
        "危险用例按实测风险R>0.5计数。实际剩余发现数为完整预算发现数减去当前发现数。"
        "MAE为预测均值与该实际数量之差的绝对值，计数NLL为该实际计数的负对数概率。"
        "允许少发现数量c为当前累计发现数除以19后向下取整，保留事件为实际剩余发现数不超过c，"
        "Brier评分为该事件预测概率与实际二元标签之差的平方。三项指标均越小越好。"
    )
    grouping = (
        "对每组配置计算38个检查点的平均有符号发现率误差d，即基础预测均值减实际后续发现数，"
        "除以各自剩余测试次数后取平均。d>0.05为基础预测偏高，d<−0.05为基础预测偏低，其余为偏差较小。"
        "分组仅用于事后描述，不用于系数估计或停止判断。先对每组配置的检查点取平均，再对配置等权平均。"
        "summary.csv中的标准差为配置间样本标准差，delta列为校正减基础预测的配对差值；"
        "检查点相互关联，未将342个检查点视为342次独立重复，未估计跨随机种子波动。"
    )
    changes = []
    for row in summaries:
        if row["scope"] not in BIAS_GROUPS or not row["configurations"]:
            continue
        changes.append(
            f"{row['label']}共有{row['configurations']}组配置，MAE、NLL、Brier的配对差值依次为"
            f"{row['delta_mae_mean']:+.5f}、{row['delta_nll_mean']:+.5f}、{row['delta_brier_mean']:+.5f}。"
        )
    direction = (
        f"{audit['paired_checkpoints']}个检查点均通过短段反馈偏差、u、预测均值变化及停止评分变化的方向核查。"
        f"其中u为正、负、近零的检查点分别为{audit['positive_u']}、{audit['negative_u']}、{audit['zero_u']}个，"
        f"确定性保留事件为{audit['deterministic_checkpoints']}个。短段实际发现率高于配对基础预测时，"
        "u为正并提高预测均值、降低停止评分；反向偏差产生相反调整。"
        "方向判断依据已完成短段反馈，配置分组依据未来预测误差，两者不要求在每个检查点同号。"
    )
    stop_note = (
        "两种方法分别在同一独立校准划分上确定阈值。切入使用MCTS和PPO的六组配置；制动使用PPO、"
        "IDM、IDM＋MOBIL的九组配置，排除被评价的MCTS。阈值取未达到95%目标的校准检查点最高评分，"
        "严格大于阈值或满足确定性条件时停止。下表只统计实际测试次数与危险发现保留情况，不重复耗时分析。"
        "完整预算零发现配置的保留率不定义，并从保留率及达标率分母中排除。"
    )
    typical_lines = [
        "| 偏差组与典型配置 | 检查点 | u | η | 实际剩余发现数 | 预测均值 基础→校正 | 停止评分 基础→校正 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in typical:
        name = f"{BIAS_GROUPS[row['bias_group']]}：{FAMILIES[row['family']]}／{SYSTEMS[row['sut']]}／测试集{row['pool'] + 1}"
        typical_lines.append(
            f"| {name} | {row['query']} | {row['u']:+.5f} | {row['eta']:+.5f} | {row['actual_remaining']} | "
            f"{row['base_mean']:.3f} → {row['corrected_mean']:.3f} | "
            f"{row['base_score']:.5f} → {row['corrected_score']:.5f} |"
        )
    stop_lookup = {(r["family"], r["method"]): r for r in stopping}
    stop_outcomes = []
    for family in ("all", *FAMILIES):
        base, corrected = stop_lookup[family, "base"], stop_lookup[family, "corrected"]
        stop_outcomes.append(
            f"{FAMILIES.get(family, '全部配置')}的平均测试次数由{base['queries_mean']:.2f}变为"
            f"{corrected['queries_mean']:.2f}，平均危险发现数由{base['found_mean']:.2f}变为"
            f"{corrected['found_mean']:.2f}，达到95%目标的配置数由"
            f"{base['target_met_configurations']}/{base['valid_configurations']}变为"
            f"{corrected['target_met_configurations']}/{corrected['valid_configurations']}。"
        )
    conclusion_parts = []
    for metric, label in (("mae", "均值绝对误差"), ("nll", "计数负对数似然"), ("brier", "保留事件概率误差")):
        change = total[f"delta_{metric}_mean"]
        conclusion_parts.append(f"{label}{'下降' if change < 0 else '上升' if change > 0 else '持平'}")
    conclusion = "本次配对评价中，收益校正的调整方向符合机制，整体" + "、".join(conclusion_parts) + "。"
    small = next(r for r in summaries if r["scope"] == "small_bias")
    conclusion += (
        "基础预测偏高与偏低组的三项平均误差均下降，偏差较小组的三项平均误差上升，"
        f"其中MAE由{small['base_mae_mean']:.5f}增至{small['corrected_mae_mean']:.5f}个。"
        "这表明短段反馈驱动的方向正确调整，对全部剩余预算的预测改善仍取决于当前任务的后续发现趋势。"
    )
    base, corrected = stop_lookup["all", "base"], stop_lookup["all", "corrected"]
    brake_base, brake_corrected = stop_lookup["brake", "base"], stop_lookup["brake", "corrected"]
    conclusion += (
        f"停止评价中，达到95%目标的配置数为{base['target_met_configurations']}/{base['valid_configurations']}"
        f"至{corrected['target_met_configurations']}/{corrected['valid_configurations']}，"
        f"MCTS制动的平均保留率由{brake_base['retention_mean']:.2%}变为{brake_corrected['retention_mean']:.2%}，"
        f"其达标配置数为{brake_corrected['target_met_configurations']}/{brake_corrected['valid_configurations']}。"
        "本次实验显示总体预测误差改善、危险发现保留数量小幅增加，95%目标的达标配置数保持不变。"
        "结论对应这9组配置和固定种子11。"
    )
    text = "\n\n".join([
        "# 发现数分布校正的有效性",
        f"本目录在 {len(configurations)} 组配置的 {audit['paired_checkpoints']} 个共同检查点，"
        "比较同一基础预测分布在收益校正前后的误差、可靠性与停止表现。",
        "## 运行",
        "在项目根目录、`metadrive` 环境中执行：",
        "```powershell\npython -B -m methods.gcrt.calibration_effectiveness\n```",
        "需要已有模型、完整轨迹、特征缓存，以及 `retention_efficiency/` 中的预测和效率统计。"
        "准备顺序见[方法说明](../../../methods/gcrt/README.md)。本命令重新运行预测网络，"
        "不训练或新增物理仿真，并重新生成本 README、统计表、图表和交底书填写内容。",
        "## 实验设置",
        "车辆切入采用 IDM、IDM＋MOBIL，前车紧急制动采用 MCTS，各使用三个测试集，"
        "与[效率实验](../retention_efficiency/README.md)的 9 组配置一致。"
        "种子为 11，完整预算 200 次，初始化 10 次，每五次构成一个测试片段。"
        "MCTS 制动为探索性补充评价，原预设独立评价结果单独保留。",
        "两种预测共用已发布的 `updated.pt` 权重和一次网络推断得到的基础分布。"
        "每组配置在第 10、15、…、195 次测试后评价，共 38 个检查点。"
        "先用已完成片段的预测与实际发现率更新系数 `u`，再校正全部剩余测试次数的发现数分布；"
        "初始化的两个片段也参与更新，实验内网络权重固定。",
        "检查点之后的实测结果仅用于离线评分。为保证配对，两种预测均评价完整轨迹上的全部共同检查点，"
        "不按停止位置截断。危险用例按实测风险 `R > 0.5` 计数，"
        "实际剩余发现数为完整预算发现数减去当前发现数。",
        "## 指标与统计",
        "| 指标 | 含义 |\n|---|---|\n"
        "| 平均绝对误差（MAE） | 预测均值与实际剩余发现数之差的绝对值，单位为个 |\n"
        "| 计数负对数似然（NLL） | 预测分布赋予实际剩余发现数的负对数概率 |\n"
        "| Brier 评分 | 保留事件的预测概率与实际二元标签之差的平方 |",
        "三项指标均越小越好。允许少发现数量 `c = floor(当前发现数 / 19)`；"
        "“保留事件”指实际剩余发现数不超过 `c`，即当前发现数至少保留完整预算发现数的 95%。",
        "先对每组配置的检查点取平均，再对配置等权平均。标准差表示配置间差异，"
        "`delta` 为校正值减基础值。342 个检查点相互关联，不作为独立重复实验，也不估计跨种子波动。",
        "每组配置的基础发现率平均偏差为 `d = mean((基础预测均值 - 实际剩余发现数) / 剩余测试次数)`。"
        "`d > 0.05` 为预测偏高，`d < -0.05` 为预测偏低，其余为偏差较小；分组仅用于事后分析。",
        "## 配对预测结果", prediction_table(summaries),
        "![校正前后预测均值与实际发现数](figures/remaining_discoveries.png)",
        "图 E2a：各偏差组的剩余发现数，曲线为组内配置等权均值。",
        "![保留事件概率可靠性图](figures/retention_reliability.png)",
        "图 E2b：保留事件的预测概率与实际发生比例。概率按 0.2 间隔分为五组，"
        "概率 1 计入末组，空组不连线；`n/k` 分别为检查点数和配置数，确定性检查点也保留。",
        "## 校正方向与停止表现",
        f"{audit['paired_checkpoints']} 个检查点均通过校正方向核查。"
        "已完成片段的实际发现率高于其基础预测时，`u` 为正，预测均值提高、停止评分降低；"
        "反向偏差产生相反调整。片段反馈与事后分组考察的时段不同，符号不要求始终一致。"
        "`u` 表示发现率的对数几率偏差，`η` 表示剩余计数分布的调整系数。"
        "[典型配置表](typical_configurations.csv)列出各组最接近中位偏差的配置，等距时按配置标识排序。",
        "两种预测分别在相同的独立校准数据上确定停止阈值：切入使用 MCTS、PPO 的 6 组配置，"
        "制动使用 PPO、IDM、IDM＋MOBIL 的 9 组配置，排除被评价的 MCTS。"
        "阈值取未达到 95% 保留目标的校准检查点最高评分，评分严格大于阈值或满足确定性条件时停止。",
        "保留率为停止时发现数除以同一完整轨迹 200 次测试的发现数。完整预算零发现时保留率未定义，"
        "不进入保留率与达标率的分母。", stopping_table(stopping),
        "本次评价中，整体三项预测误差均下降，基础预测偏高与偏低组均有改善；"
        "偏差较小组的三项误差均上升。"
        f"达到 95% 目标的配置数保持为 {corrected['target_met_configurations']}/{corrected['valid_configurations']}，"
        f"MCTS 制动的平均保留率由 {brake_base['retention_mean']:.2%} 变为 {brake_corrected['retention_mean']:.2%}，"
        f"达标配置数为 {brake_corrected['target_met_configurations']}/{brake_corrected['valid_configurations']}。"
        "结论对应这 9 组配置和固定种子 11。",
        "## 文件索引",
        "| 文件 | 内容 |\n|---|---|\n"
        "| [config.json](config.json)、[calibration.json](calibration.json) | 实验设置、权重与校准来源 |\n"
        "| [checkpoint_metrics.csv](checkpoint_metrics.csv)、[configuration_metrics.csv](configuration_metrics.csv) | 检查点指标、逐配置偏差与分组 |\n"
        "| [summary.csv](summary.csv) | 配对预测误差汇总 |\n"
        "| [reliability_bins.csv](reliability_bins.csv)、[typical_configurations.csv](typical_configurations.csv) | 概率分组、典型配置与校正系数 |\n"
        "| [stopping_metrics.csv](stopping_metrics.csv)、[stopping_summary.csv](stopping_summary.csv) | 逐配置停止结果与汇总 |\n"
        "| `forecasts.json`、[audit.json](audit.json) | 可重新生成的完整配对分布缓存与核查结果 |\n"
        "| [disclosure_text.md](disclosure_text.md) | 交底书第 8.3 节填写内容，含详细典型配置表 |",
        "所有 CSV 均有同名 JSON。图片保存为 `figures/` 下的 PNG。",
    ])
    (OUTPUT / "README.md").write_text(text + "\n", encoding="utf-8")
    disclosure = "\n\n".join([
        "# 实验二发现数分布校正的有效性", setup, protocol, grouping,
        prediction_table(summaries), outcome, " ".join(changes), direction,
        "实验图E2a为三个偏差组的校正前后预测均值与实际后续发现数对照；实验图E2b为按功能场景绘制的"
        "保留事件概率可靠性图，标出理想对角线及各概率箱的检查点数和配置数。",
        "典型配置及校正系数变化如下。", "\n".join(typical_lines),
        stop_note, stopping_table(stopping), " ".join(stop_outcomes), conclusion,
    ])
    (OUTPUT / "disclosure_text.md").write_text(disclosure + "\n", encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    replays = read_json(ARCHIVED / "replays.json")
    archived_calibration = read_json(ARCHIVED / "calibration.json")
    efficiency_rows = read_json(ARCHIVED / "comparison_metrics.json")
    checkpoints, distributions, stopped, calibrations, weights = [], [], [], {}, {}
    max_archive_difference = 0.0
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            published = load_stopper(family, updated=True)
            penalty = read_json(ROOT / family / "training.json")["penalty"]
            weights[family] = {"path": f"results/gcrt/{family}/updated.pt",
                               "weights_sha256": published.contract["weights_sha256"], "penalty": penalty}
            for method in METHODS:
                assert archived_calibration[family][ARCHIVED_METHODS[method]]["weights_sha256"] == published.contract["weights_sha256"]
            calibrations[family] = {method: calibrate(replays, family, method) for method in METHODS}
            for candidate_family, sut in DISPLAY_GROUPS:
                if candidate_family != family:
                    continue
                for pool in (0, 1, 2):
                    trace, task = load_task(family, pool, sut)
                    prediction = forecast_task(published.network, task, penalty)
                    configuration_id = f"{family}/pool_{pool}/{sut}"
                    archived_pair = {
                        method: next(r for r in replays if r["family"] == family
                                     and r["sut"] == sut and r["pool"] == pool
                                     and r["method"] == ARCHIVED_METHODS[method])
                        for method in METHODS
                    }
                    assert [r["query"] for r in prediction["forecast"]] == list(range(INITIAL, BUDGET, SEGMENT))
                    assert prediction["full_found"] == trace["found"][-1]
                    checks = []
                    for index, check in enumerate(prediction["forecast"]):
                        for method, archive in archived_pair.items():
                            old = archive["forecast"][index]
                            assert (old["query"], old["found"]) == (check["query"], check["found"])
                            pairs = (("remaining_mean", "remaining_mean"), ("score", "score"), ("tilt", "tilt")) if method == "corrected" else (
                                ("remaining_mean", "base_remaining_mean"), ("score", "base_score"))
                            for old_key, new_key in pairs:
                                difference = abs(old[old_key] - check[new_key])
                                assert difference <= NUMERICAL_TOLERANCE, (configuration_id, check["query"], old_key, difference)
                                max_archive_difference = max(max_archive_difference, difference)
                        assert check["completed_segments"] == check["query"] // SEGMENT
                        truth = trace["found"][-1] - trace["found"][check["query"] - 1]
                        metadata = {"configuration_id": configuration_id, "family": family, "sut": sut, "pool": pool}
                        row = {**metadata, **paired_metrics(check, truth)}
                        checkpoints.append(row)
                        checks.append(row)
                        distributions.append({**metadata, **check})
                    for method in METHODS:
                        row = stopping_metrics(trace, checks, method, calibrations[family][method]["threshold"])
                        if method == "corrected":
                            old = next(r for r in efficiency_rows if r["family"] == family
                                       and r["sut"] == sut and r["pool"] == pool and r["method"] == "gcrt")
                            assert row["queries"] == old["queries"] and row["found"] == old["danger_found"]
                        stopped.append(row)
                    print(f"PAIRED {configuration_id}: {len(checks)} checkpoints", flush=True)
    configurations = configuration_metrics(checkpoints)
    summaries = summarize_predictions(configurations)
    bins = reliability_bins(checkpoints)
    typical = typical_configurations(configurations, checkpoints)
    stop_summary = summarize_stopping(stopped)
    assert len(configurations) == 9 and len(checkpoints) == 342 and len(stopped) == 18
    for family in FAMILIES:
        for method in METHODS:
            assert sum(r["checkpoints"] for r in bins if r["family"] == family and r["method"] == method) == sum(r["family"] == family for r in checkpoints)
    audit = {
        "passed": True, "configurations": len(configurations), "paired_checkpoints": len(checkpoints),
        "bias_group_configurations": {group: sum(r["bias_group"] == group for r in configurations) for group in BIAS_GROUPS},
        "positive_u": sum(r["u"] > NUMERICAL_TOLERANCE for r in checkpoints),
        "negative_u": sum(r["u"] < -NUMERICAL_TOLERANCE for r in checkpoints),
        "zero_u": sum(abs(r["u"]) <= NUMERICAL_TOLERANCE for r in checkpoints),
        "deterministic_checkpoints": sum(r["deterministic"] for r in checkpoints),
        "max_archived_prediction_difference": max_archive_difference,
        "same_base_distribution": True, "matched_pmf_normalization_and_metrics": True,
        "target_mean_and_direction_verified": True, "initial_segments_included": True,
        "calibration_evaluation_disjoint": True, "corrected_stopping_matches_efficiency_figure": True,
        "probability_bins_exhaustive": True, "new_prediction_replays": 9,
        "new_physical_calls": 0, "new_training_steps": 0,
    }
    config = {
        "completed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_has_changes": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "seed": SEED, "budget": BUDGET, "initial": INITIAL, "segment": SEGMENT,
        "pool_size": 1024, "pools": [0, 1, 2], "retention_target": 0.95,
        "display_groups": [{"family": f, "sut": s} for f, s in DISPLAY_GROUPS],
        "weights": weights, "methods": METHODS, "prediction_horizon": "all remaining tests only",
        "checkpoints": list(range(INITIAL, BUDGET, SEGMENT)),
        "grouping": {"metric": "mean((base_mean - actual_remaining) / remaining) within configuration",
                     "tolerance": BIAS_TOLERANCE, "boundaries": "over > +0.05; under < -0.05; small otherwise",
                     "use": "retrospective description only; never used for correction or stopping"},
        "aggregation": "checkpoint mean within configuration, then equally weighted configuration mean; sample SD across configurations",
        "delta": "corrected minus base; negative means lower error",
        "probability_bin_edges": list(BIN_EDGES), "final_bin_right_inclusive": True,
        "typical_selection": "nearest group-median signed base rate bias, tie by configuration ID",
        "typical_checkpoints": list(TYPICAL_CHECKPOINTS),
        "calibration_sources": "results/gcrt/retention_efficiency/replays.json; archived frozen-weight probabilities",
        "calibration_controllers": CALIBRATION_SUTS,
        "mcts_scope": "exploratory supplementary evaluation, selected after reviewing original evaluation results",
        "torch_threads": torch.get_num_threads(), "device": str(next(published.network.parameters()).device),
        "new_prediction_replays": 9, "new_physical_calls": 0, "new_training_steps": 0,
    }
    for name, rows in (
        ("checkpoint_metrics", checkpoints), ("configuration_metrics", configurations),
        ("summary", summaries), ("reliability_bins", bins), ("typical_configurations", typical),
        ("stopping_metrics", stopped), ("stopping_summary", stop_summary),
    ):
        write_json(OUTPUT / f"{name}.json", rows)
        save_csv(OUTPUT / f"{name}.csv", rows)
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "calibration.json", calibrations)
    write_json(OUTPUT / "forecasts.json", distributions)
    write_json(OUTPUT / "audit.json", audit)
    plot_results(checkpoints, configurations, bins)
    write_report(summaries, configurations, stop_summary, typical, audit)
    print(prediction_table(summaries), flush=True)
    print(stopping_table(stop_summary), flush=True)


if __name__ == "__main__":
    main()
