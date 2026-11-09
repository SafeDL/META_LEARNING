"""Evidence tables and compact publication figures for the completed attribution replay."""

import csv
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .attribution import CHECKPOINTS, OUTPUT, VARIANTS, csv_report, registered_runs
from .joint_figures import STYLE, STYLES
from .storage import read_json, write_json

SCENES = ("car_following", "front_brake", "cut_out", "lane_change_rear")
TITLES = ("Car following", "Lead braking", "Cut-out", "Lane change / rear conflict")


def totals(rows):
    failures = sum(m["unique_collisions"] for m in rows)
    truth = sum(m["pool_unique_collisions"] for m in rows)
    coverage = [m["collision_recall"] for m in rows if m["collision_recall"] is not None]
    return {
        "tasks": len(rows),
        "failures": failures,
        "post_initial_failures": sum(m["post_initial_collisions"] for m in rows),
        "mean_auc": float(np.mean([m["discovery_auc"] for m in rows])),
        "truth_failures": truth,
        "pooled_coverage": failures / truth if truth else None,
        "macro_coverage": float(np.mean(coverage)) if coverage else None,
        "defined_coverage_tasks": len(coverage),
        "found_at": {str(n): sum(m["discovery_curve"][n - 1] for m in rows) for n in CHECKPOINTS},
    }


def feedback_report():
    tasks = sorted({key[:2] for key in registered_runs() if key[2] == "joint"})
    affected = {
        row["scenario"]
        for row in read_json(OUTPUT / "reports/source_views.json")
        if row["affected"]
    }
    records = []
    corrected = {}
    costs = dict.fromkeys(("selection_s", "adaptation_s", "replay_wall_s"), 0.0)
    for sut, kind in tasks:
        variants = {}
        for variant in (*VARIANTS, *(["historical_collision_fixed"] if kind in affected else [])):
            folder = "common_feedback" if variant in VARIANTS else "corrected_control"
            saved = read_json(OUTPUT / folder / sut / kind / f"{variant}.json.gz")
            specification = saved["specification"]
            if (specification["sut_id"], specification["scenario"], specification["variant"]) != (
                sut,
                kind,
                variant,
            ):
                raise ValueError("Attribution report contains another task's replay")
            for name in costs:
                costs[name] += saved["metrics"][name]
            if variant not in VARIANTS:
                corrected[sut, kind] = saved["metrics"]
                continue
            variants[variant] = saved["result"]["prediction_snapshots"]
        for position, n in enumerate(CHECKPOINTS):
            reference = variants["joint"][position]
            for variant in VARIANTS:
                snapshot = variants[variant][position]
                if (
                    snapshot["paid_queries"] != n
                    or snapshot["observed_indices"] != reference["observed_indices"]
                ):
                    raise ValueError("Common-feedback diagnostic used different prefixes")
                remaining = np.ones(len(snapshot["probabilities"]), dtype=bool)
                remaining[snapshot["observed_indices"]] = False
                difference = np.asarray(snapshot["probabilities"]) - reference["probabilities"]
                records.append(
                    {
                        "sut_id": sut,
                        "scenario": kind,
                        "variant": variant,
                        "paid_queries": n,
                        **snapshot["evaluation"],
                        "mean_abs_probability_change_from_joint": float(
                            np.abs(difference[remaining]).mean()
                        ),
                        "max_abs_probability_change_from_joint": float(
                            np.abs(difference[remaining]).max()
                        ),
                        "weight_l1_change_from_joint": float(
                            np.abs(
                                np.asarray(snapshot["component_weights"])
                                - reference["component_weights"]
                            ).sum()
                        ),
                    }
                )
    csv_report(OUTPUT / "reports/common_feedback.csv", records)
    return records, corrected, costs


@plt.rc_context(STYLE)
def figures(runs, feedback, corrected):
    root = OUTPUT / "reports/figures"
    root.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    methods = ("joint", "no_history", "collision", "fixed_weights", "scenario_fuzz", "random")
    handles = []
    for panel, (axis, kind, title) in enumerate(zip(axes.flat, SCENES, TITLES)):
        for method in methods:
            curves = [
                r["metrics"]["discovery_curve"]
                for (s, k, m), r in runs.items()
                if k == kind and m == method
            ]
            label, color, style = STYLES[method]
            (line,) = axis.plot(
                np.arange(1, 201),
                np.sum(curves, axis=0),
                color=color,
                linestyle=style,
                linewidth=1.3,
                label=label,
            )
            if panel == 0:
                handles.append(line)
        axis.axvline(20, color="#aaaaaa", linestyle=":", linewidth=0.8)
        axis.set(
            title=f"({chr(97 + panel)}) {title}",
            xlabel="Paid queries per task",
            ylabel="Failures across eight ADSs",
            xlim=(1, 200),
            ylim=(0, None),
        )
        axis.grid(alpha=0.15)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(root / "discovery_attribution.png", dpi=600)
    plt.close(fig)

    systems = sorted({key[0] for key in runs})
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 3.6), sharey=True)
    controls = ("no_history", "collision", "fixed_weights", "historical_collision")
    matrices = [
        np.array(
            [
                [
                    runs[s, k, "joint"]["metrics"]["unique_collisions"]
                    - (
                        corrected.get((s, k), runs[s, k, m]["metrics"])
                        if m == "historical_collision"
                        else runs[s, k, m]["metrics"]
                    )["unique_collisions"]
                    for k in SCENES
                ]
                for s in systems
            ]
        )
        for m in controls
    ]
    scale = max(abs(matrix).max() for matrix in matrices)
    for panel, (axis, method, matrix) in enumerate(zip(axes, controls, matrices)):
        picture = axis.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
        for (i, j), number in np.ndenumerate(matrix):
            axis.text(
                j,
                i,
                f"{number:+d}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(number) > scale * 0.65 else "black",
            )
        label = (
            "Historical collision only\n(same record IDs)"
            if method == "historical_collision"
            else STYLES[method][0]
        )
        axis.set(
            title=f"({chr(97 + panel)}) Full −\n{label}",
            xticks=range(4),
            xticklabels=("CF", "FB", "CO", "LC"),
            yticks=range(len(systems)),
            yticklabels=[s.removeprefix("independent_") for s in systems],
        )
    fig.subplots_adjust(left=0.20, right=0.84, top=0.82, bottom=0.12, wspace=0.18)
    color_axis = fig.add_axes((0.87, 0.16, 0.015, 0.64))
    fig.colorbar(picture, cax=color_axis, label="Failure-count difference")
    fig.savefig(root / "task_contributions.png", dpi=600)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for axis, metric, title in zip(
        axes,
        ("nll", "brier", "top_200_failures"),
        ("Unqueried collision NLL", "Unqueried Brier", "Failures in top 200"),
    ):
        for variant in VARIANTS:
            means = [
                np.mean(
                    [
                        r[metric]
                        for r in feedback
                        if r["variant"] == variant and r["paid_queries"] == n
                    ]
                )
                for n in CHECKPOINTS
            ]
            label, color, style = STYLES[variant]
            axis.plot(
                CHECKPOINTS,
                means,
                marker="o",
                markersize=3,
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.2,
            )
        axis.set(xlabel="Paid feedback count", ylabel=title, xticks=CHECKPOINTS)
        axis.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(root / "common_feedback.png", dpi=600)
    plt.close(fig)
    return [p.relative_to(OUTPUT).as_posix() for p in sorted(root.glob("*.png"))]


def main():
    runs = registered_runs()
    feedback, corrected, costs = feedback_report()
    methods = (
        "joint",
        "no_history",
        "collision",
        "fixed_weights",
        "random",
        "av_bo_rf_ei",
        "samota",
        "scenario_fuzz",
        "paired_system_active_evaluation",
        "behavexplor",
    )
    aggregate = {
        m: totals([r["metrics"] for (s, k, name), r in runs.items() if name == m]) for m in methods
    }
    corrected_metrics = [
        corrected.get((s, k), runs[s, k, "historical_collision"]["metrics"])
        for s, k, method in runs
        if method == "joint"
    ]
    aggregate["historical_collision_fixed"] = totals(corrected_metrics)
    corrected_rows = []
    for (sut, kind, method), row in runs.items():
        if method != "joint":
            continue
        control = corrected.get((sut, kind), runs[sut, kind, "historical_collision"]["metrics"])
        corrected_rows.append(
            {
                "sut_id": sut,
                "scenario": kind,
                "replayed": (sut, kind) in corrected,
                "joint_failures": row["metrics"]["unique_collisions"],
                "control_failures": control["unique_collisions"],
                "delta_failures": row["metrics"]["unique_collisions"]
                - control["unique_collisions"],
                "joint_auc": row["metrics"]["discovery_auc"],
                "control_auc": control["discovery_auc"],
                "delta_auc": row["metrics"]["discovery_auc"] - control["discovery_auc"],
                "control_coverage": control["collision_recall"],
            }
        )
    csv_report(OUTPUT / "reports/corrected_history_comparison.csv", corrected_rows)
    contrasts = {}
    for method in (*methods[1:], "historical_collision_fixed"):
        values = [
            (
                r["metrics"]["unique_collisions"]
                - (
                    corrected.get((s, k), runs[s, k, "historical_collision"]["metrics"])
                    if method == "historical_collision_fixed"
                    else runs[s, k, method]["metrics"]
                )["unique_collisions"]
            )
            for (s, k, name), r in runs.items()
            if name == "joint"
        ]
        contrasts[method] = {
            "delta_failures": sum(values),
            "wins": sum(v > 0 for v in values),
            "losses": sum(v < 0 for v in values),
            "ties": sum(v == 0 for v in values),
        }
    with (OUTPUT / "reports/collision_diagnostics.csv").open(encoding="utf-8-sig") as stream:
        diagnostics = list(csv.DictReader(stream))
    following = [r for r in diagnostics if r["scenario"] == "car_following"]
    margin_count = sum(float(r["constant_lead_max_brake_margin_m"]) < 0 for r in following)
    partners = {
        k: dict(Counter(r["partner"] for r in diagnostics if r["scenario"] == k)) for k in SCENES
    }
    common_means = [
        {
            "variant": v,
            "paid_queries": n,
            **{
                metric: float(
                    np.mean(
                        [
                            r[metric]
                            for r in feedback
                            if r["variant"] == v and r["paid_queries"] == n
                        ]
                    )
                )
                for metric in (
                    "nll",
                    "brier",
                    "top_200_failures",
                    "mean_abs_probability_change_from_joint",
                )
            },
        }
        for n in CHECKPOINTS
        for v in VARIANTS
    ]
    with (OUTPUT / "reports/paired_differences.csv").open(encoding="utf-8-sig") as stream:
        pairs = list(csv.DictReader(stream))
    diversity = []
    for kind, title in zip(SCENES, TITLES):
        valid = [
            row
            for row in pairs
            if row["scenario"] == kind
            and row["control"] == "scenario_fuzz"
            and int(row["matched_discovery_count"]) >= 2
        ]
        diversity.append(
            {
                "scenario": kind,
                "title": title,
                "defined_tasks": len(valid),
                **{
                    f"{which}_{metric}": float(
                        np.mean([float(row[f"{which}_{metric}"]) for row in valid])
                    )
                    if valid
                    else None
                    for which in ("joint", "control")
                    for metric in ("pair_distance", "nearest_neighbor_distance")
                },
            }
        )
    pictures = figures(runs, feedback, corrected)
    summary = {
        "status": "complete",
        "platform": "HighwayEnv",
        "new_physical_calls": 0,
        "common_feedback_runs": 96,
        "corrected_control_runs": len(corrected),
        "logical_queries": (96 + len(corrected)) * 200,
        "aggregate": aggregate,
        "contrasts": contrasts,
        "common_feedback": common_means,
        "diagnostic_collision_records": len(diagnostics),
        "collision_partners": partners,
        "following_longitudinal_margin_negative": margin_count,
        "figures": pictures,
        "recorded_costs_s": costs,
        "matched_count_diversity_vs_scenario_fuzz": diversity,
    }
    write_json(OUTPUT / "reports/summary.json", summary)
    lines = [
        "# 现有收益归因报告",
        "",
        "本报告使用已完成的 HighwayEnv 八目标、四场景档案。",
        "",
        "## 复核范围",
        "",
        "32 个任务，共 56,000 个候选执行真值；碰撞分母 5,041。"
        "固定 20 次共同初始化、200 次预算、种子 11。重新复核已保存方法的失效 ID、发现曲线、AUC、覆盖率和距离。"
        "新增 96 次共同反馈诊断及 16 次历史对照回放，共 22,400 次逻辑查询；新增物理执行为 0。",
        "",
        f"新增运行记录中的适配耗时合计 {costs['adaptation_s']:.1f} s、选例耗时 {costs['selection_s']:.1f} s、"
        f"付费档案回放耗时 {costs['replay_wall_s']:.1f} s。这是各运行已记录分项的求和，"
        "不包含来源构建、额外诊断快照和报告绘图，也不等同于并行任务的端到端墙钟时间。",
        "",
        "## 总体效果",
        "",
        "AUC 为每任务 200 次累计发现数的均值，再在 32 个任务上取平均；不是归一化概率。"
        "宏覆盖率只平均 26 个非零失效任务；汇总覆盖率用全部库真实失效数作为分母。",
        "",
        "| 方法 | 发现数 | 初始化后新增 | 平均 AUC | 汇总覆盖率 | 宏覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, values in aggregate.items():
        lines.append(
            f"| {method} | {values['failures']} | {values['post_initial_failures']} | "
            f"{values['mean_auc']:.4f} | {values['pooled_coverage']:.2%} | {values['macro_coverage']:.2%} |"
        )
    lines += [
        "",
        "## 贡献与归因边界",
        "",
        "| 实现行为 | 比较证据 | 收益方向与适用范围 | 尚不能归因的部分 |",
        "| --- | --- | --- | --- |",
        "| 历史响应先验与表征 | joint 对 no_history | 见上表和逐任务差异，主要检查早期发现 | 同时移除了先验及表征，不能独立证明元训练 |",
        "| 目标碰撞—RSS 适配 | joint 对 collision；共同 Random 前缀 | 自适应发现数净差 −4，未支持总体发现收益；预测变化另列 | 预测质量变化不直接等于失效发现收益 |",
        "| 查询前似然更新来源权重 | joint 对 fixed_weights；共同前缀 | 自适应发现数净增 78；具体 ADS/场景见表 | 权重与目标适配共同作用，不证明每种来源必不可少 |",
        "| 历史 RSS 处理流程 | 固定完整方法记录 ID 后的 historical_collision_fixed | 以本次纠正结果为准 | 原离线训练差异保留，不是纯 RSS 观测似然消融 |",
        "| 来源分离、元训练 | 既有机制比较不能完全独立隔离 | 未单独证实 | 本轮不增加机制实验填补空白 |",
        "| 任务完成后积累 | 原顺序实验独立保留 | 不属于本轮新实验 | 本轮没有新平台顺序积累证据 |",
        "",
        "## 早期与最终发现",
        "",
        "| 方法 | 20 | 50 | 100 | 200 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in ("joint", "no_history", "collision", "fixed_weights", "scenario_fuzz", "random"):
        lines.append(
            "| "
            + method
            + " | "
            + " | ".join(str(aggregate[method]["found_at"][str(n)]) for n in CHECKPOINTS)
            + " |"
        )
    lines += [
        "",
        "## 收益集中在哪些任务",
        "",
        "| 场景 | 真值碰撞数 | 完整方法 | 无历史 | 固定权重 | ScenarioFuzz |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, title in zip(SCENES, TITLES):
        counts = {
            m: sum(
                r["metrics"]["unique_collisions"]
                for (s, k, name), r in runs.items()
                if k == kind and name == m
            )
            for m in ("joint", "no_history", "fixed_weights", "scenario_fuzz")
        }
        truth = sum(
            r["metrics"]["pool_unique_collisions"]
            for (s, k, name), r in runs.items()
            if k == kind and name == "joint"
        )
        lines.append(
            f"| {title} | {truth} | {counts['joint']} | {counts['no_history']} | "
            f"{counts['fixed_weights']} | {counts['scenario_fuzz']} |"
        )
    lines += [
        "",
        "完整方法相对 ScenarioFuzz 的净增 185 个失效中，换道场景贡献 164 个（88.65%）。"
        "因此当前优势集中于换道冲突，不应表述为所有场景均有同等增益。"
        "前车制动库的 99 个碰撞中，98 个来自自训 DQN、1 个来自 UCT，另外六个 ADS 无碰撞；"
        "这一分布支持修正被测系统和参数依据的必要性，不能作为调节碰撞率的目标。",
        "",
        "目标库总失效率为 5,041/56,000=9.00%；完整方法选中样本中的失效率为 "
        "2,152/6,400=33.63%。后者是失效导向选择后的比例，两者均不是自然驾驶失效率。",
    ]
    lines += [
        "",
        "| ADS | 跟车真值失效率 | 制动 | 切出 | 换道 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for sut in sorted({key[0] for key in runs}):
        values = [runs[sut, kind, "joint"]["metrics"]["pool_failure_rate"] for kind in SCENES]
        lines.append(
            "| "
            + sut.removeprefix("independent_")
            + " | "
            + " | ".join(f"{value:.2%}" for value in values)
            + " |"
        )
    lines += [
        "",
        "## 同反馈前缀的预测诊断",
        "",
        "每项均使用同一 Random 查询顺序并逐条更新，完整保留初始化期间的反馈。"
        "下表仅评估未查询候选；top-200 是评估端排序诊断，不是新增执行或发现数量。",
        "",
        "| 已反馈数 | 变体 | NLL | Brier | top-200 真实失效数（任务均值） |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in common_means:
        lines.append(
            f"| {row['paid_queries']} | {row['variant']} | {row['nll']:.5f} | "
            f"{row['brier']:.5f} | {row['top_200_failures']:.3f} |"
        )
    lines += [
        "",
        "## 物理诊断与多样性",
        "",
        f"跟车库的 {len(following)} 条真实碰撞中，{margin_count} 条的初始净间距小于 "
        "max(自车速度−前车速度, 0)²/(2×8)。这表示前车匀速、立即以 8 m/s² 纵向制动的参考模型空间不足，"
        "不是全策略不可避免碰撞证明；横向避让及实际反应时间另由轨迹记录核验。",
        "",
        "[碰撞诊断表](collision_diagnostics.csv)逐条保存碰撞对象、碰撞时刻、初始间距、实际制动、"
        "横向位移、事件发生及障碍可见至碰撞的时间，引用原始轨迹。零失效任务保留于任务表。",
        "",
        "[逐任务指标](task_metrics.csv)提供同预算距离；[配对差异](paired_differences.csv)"
        "使用每对方法共同发现数 k 的前 k 个失效比较距离。k<2 时未定义。"
        "多样性不保证随发现数同步提高，不能仅凭失效数量宣称多样性优势。",
        "",
        "下表在每个任务的共同发现数 k 下比较完整方法与 ScenarioFuzz，再对有定义的任务取均值；"
        "参数使用该功能场景的归一化坐标。换道的两项距离均较低，当前结果不支持一致的多样性优势。",
        "",
        "| 场景 | 有效任务 | 两两距离：完整 / ScenarioFuzz | 最近邻距离：完整 / ScenarioFuzz |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {row['title']} | {row['defined_tasks']} | "
            f"{row['joint_pair_distance']:.4f} / {row['control_pair_distance']:.4f} | "
            f"{row['joint_nearest_neighbor_distance']:.4f} / {row['control_nearest_neighbor_distance']:.4f} |"
            for row in diversity
            if row["defined_tasks"]
        ],
        "",
        "## 纠正的历史 RSS 对照",
        "",
        "固定全部来源记录 ID 后，前车制动的 IDM 来源替换 162 条，切出的 IDM 来源替换 29 条；"
        "两场景各八目标共回放 16 个任务，另外 16 个任务的视图和模型一致，复用原结果。"
        "原历史 RSS 关闭的离线模型保持不变；旧正式结果未覆盖。记录清单见 [source_views.json](source_views.json)。",
        "[纠正后的 32 任务比较](corrected_history_comparison.csv)同时标注实际回放与视图一致而复用的任务。",
        "",
        f"纠正后对照发现 {aggregate['historical_collision_fixed']['failures']} 个失效，"
        f"完整方法发现 {aggregate['joint']['failures']} 个，净差 "
        f"{contrasts['historical_collision_fixed']['delta_failures']:+d}；"
        f"对应平均 AUC 为 {aggregate['historical_collision_fixed']['mean_auc']:.4f} 与 {aggregate['joint']['mean_auc']:.4f}。"
        "固定记录 ID 后，总体收益方向仍为负，当前历史 RSS 处理流程的增益未获支持。",
        "",
        "## 产物与停止位置",
        "",
        "图像为 600 dpi PNG："
        + "、".join(f"[{p.rsplit('/', 1)[-1]}]({p.removeprefix('reports/')})" for p in pictures)
        + "。"
        "本轮没有重复生成案例 GIF；已有实际轨迹 GIF 仍保留。",
        "",
        "本批次的收益归因已经完成，设置、共同反馈及固定记录对照均按保留协议复核。",
        "",
        "单种子结果不作显著性结论。保留全部中性、负向和零失效任务；本轮结束不以获得正结果为条件。",
        "",
    ]
    (OUTPUT / "reports/results.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
