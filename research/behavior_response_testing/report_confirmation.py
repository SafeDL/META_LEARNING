"""Report the audited prospective experiment without changing its acceptance gate."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods.history_guided_testing.io import read_json

from .confirmation import CONFIRMATION, verify_lock
from .train import OUTPUT, ROOT

LABELS = {
    "candidate": "行为后验候选",
    "previous_best": "此前六源冻结最佳",
    "frozen_original": "原始冻结方法",
    "ras_frt_uq": "RAS-FRT-UQ",
    "matched_collision_gp": "同信息碰撞得分 GP",
    "fixed_behavior": "固定行为先验排序",
    "discrete_history": "十六历史行为后验"
}
PLOT_LABELS = {
    "candidate": "Behavior posterior",
    "previous_best": "Frozen previous best",
    "frozen_original": "Frozen original",
    "ras_frt_uq": "RAS-FRT-UQ",
    "matched_collision_gp": "Matched collision GP",
    "fixed_behavior": "Fixed behavior prior",
    "discrete_history": "Discrete history posterior"
}
COLORS = {
    "candidate": "#0072B2",
    "previous_best": "#D55E00",
    "frozen_original": "#999999",
    "ras_frt_uq": "#009E73",
    "matched_collision_gp": "#CC79A7",
    "fixed_behavior": "#E69F00",
    "discrete_history": "#56B4E9"
}


def format_profiles(rows, key, percent=False):
    values = np.asarray([row[key] for row in rows]) * (100 if percent else 1)
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def normalized_curves(summary):
    protocol, records = summary["protocol"], summary["records"]
    curves = {}
    expected = len(protocol["seeds"]) * protocol["replicates_per_profile"]
    for method in protocol["methods"]:
        profile_curves = []
        for profile in protocol["profiles"]:
            rows = [
                row for row in records if row["method"] == method
                and row["profile"] == profile["name"]
            ]
            assert len(rows) == expected
            profile_curves.append(
                np.mean([
                    np.asarray(row["curve"]) / row["pool_collisions"]
                    if row["pool_collisions"] else np.zeros(protocol["budget"])
                    for row in rows
                ],
                        axis=0))
        curves[method] = np.asarray(profile_curves)
        saved = np.array(
            [row["normalized_area"] for row in summary["per_profile"][method]])
        np.testing.assert_allclose(curves[method].mean(1),
                                   saved,
                                   atol=1e-12,
                                   rtol=0)
    return curves


def figures(summary):
    protocol = summary["protocol"]
    curves = normalized_curves(summary)
    queries = np.arange(1, protocol["budget"] + 1)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for method in protocol["methods"]:
        axes[0, 0].plot(queries,
                        100 * curves[method].mean(0),
                        color=COLORS[method],
                        label=PLOT_LABELS[method],
                        linewidth=2.2 if method == "candidate" else 1.2)
    candidate_rows = [
        r for r in summary["records"] if r["method"] == "candidate"
    ]
    oracle = np.mean([
        np.minimum(queries, r["pool_collisions"]) / r["pool_collisions"]
        if r["pool_collisions"] else np.zeros(protocol["budget"])
        for r in candidate_rows
    ],
                     axis=0)
    axes[0, 0].plot(queries,
                    100 * oracle,
                    "k--",
                    label="Budget oracle",
                    linewidth=1.2)
    axes[0, 0].set(xlabel="Target queries",
                   ylabel="Mean discovery fraction (%)")
    for control in protocol["primary_controls"]:
        delta = curves["candidate"] - curves[control]
        axes[0, 1].plot(queries,
                        100 * delta.mean(0),
                        color=COLORS[control],
                        label=PLOT_LABELS[control])
    axes[0, 1].axhline(0, color="gray", lw=0.8)
    axes[0, 1].set(xlabel="Target queries",
                   ylabel="Candidate minus control (pp)")
    axes[0, 1].legend(fontsize=8)
    controllers = np.array([p["controller"] for p in protocol["profiles"]])
    for axis, control in zip(axes[1],
                             ("previous_best", "matched_collision_gp")):
        area = 100 * (curves["candidate"].mean(1) - curves[control].mean(1))
        recall = 100 * np.array([
            a["recall"] - b["recall"]
            for a, b in zip(summary["per_profile"]["candidate"],
                            summary["per_profile"][control])
        ])
        for controller, marker, color in (("IDM", "o", "#0072B2"),
                                          ("FVDM", "s", "#D55E00")):
            mask = controllers == controller
            axis.scatter(area[mask],
                         recall[mask],
                         marker=marker,
                         color=color,
                         alpha=0.8,
                         label=controller)
        axis.axhline(0, color="gray", lw=0.8)
        axis.axvline(0, color="gray", lw=0.8)
        axis.set(title=f"Against {PLOT_LABELS[control]}",
                 xlabel="Discovery area difference (pp)",
                 ylabel="Final recall difference (pp)")
        axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    directory = OUTPUT / "figures"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"confirmation.{extension}", dpi=180)
    plt.close(figure)


def main():
    summary = read_json(CONFIRMATION / "summary.json")
    protocol = summary["protocol"]
    verify_lock(protocol)
    physical = read_json(OUTPUT / "confirmation_audit/summary.json")
    statistics = read_json(OUTPUT / "confirmation_audit/statistics.json")
    assert physical["all_physical_banks_audited"] and physical[
        "statistical_audit_passed"]
    assert statistics["success"] == summary["success"]
    methods, profiles = protocol["methods"], protocol["profiles"]
    lines = [
        "# 行为后验方法的完整独立确认", "",
        f"共同主指标验收：**{'通过' if summary['success'] else '未通过'}**。", "",
        f"{len(profiles)} 个新系统参数配置，每配置两个 2048 场景池，五个冻结网络种子。"
        "每方法每次运行 200 次不重复查询。统计单位为系统配置，先在配置内平均十个内部重复，"
        "再报告配置均值 ± 样本标准差。速度和最终召回分别验收。", "", "## 完整方法比较", "",
        "| 方法 | 归一化发现面积 (%) | 最终召回 (%) | F10 | F30 | F50 | F100 | F150 | F200 | 遗漏数 | 找全比例 (%) | 预算上界达到率 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    keys = (("normalized_area", True), ("recall", True), ("F10", False),
            ("F30", False), ("F50", False), ("F100", False), ("F150", False),
            ("F200", False), ("missed_failures", False),
            ("all_failures_found", True), ("budget_upper_bound_attainment",
                                           True))
    for method in methods:
        rows = summary["per_profile"][method]
        lines.append(f"|{LABELS[method]}|" + "|".join(
            format_profiles(rows, key, percent)
            for key, percent in keys) + "|")
    lines += [
        "", "总失效数 N 从完整场景池事后计算，未交给选择器。空池统一记面积 0、召回 1。"
        "预算上界达到率为 F200/min(200,N)，空池记 1；总遗漏数包含超过预算的失效。"
        "曲线中的空池累计发现比例记 0，因此含空池时，曲线终点与表中召回约定有差别。", "", "## 八项预登记主比较", "",
        f"配对差以百分点表示。八项精确双侧符号翻转检验使用 Holm 校正，轮内阈值 {protocol['round_alpha']}；"
        "区间在控制器类型内以系统配置重采样 20000 次。", "",
        "| 对照 | 指标 | 候选减对照 | 配对 95% CI | 精确 p | Holm p |",
        "|---|---|---:|---:|---:|---:|"
    ]
    failed = []
    for control, values in summary["comparisons"].items():
        for metric, result in values.items():
            label = "发现面积" if metric == "normalized_area" else "最终召回"
            low, high = np.asarray(result["bootstrap_95_ci"]) * 100
            lines.append(
                f"|{LABELS[control]}|{label}|{100 * result['mean_difference']:+.5f}|"
                f"[{low:+.5f}, {high:+.5f}]|{result['exact_two_sided_p']:.8g}|{result['holm_p']:.8g}|"
            )
            if result["mean_difference"] <= 0 or result["holm_p"] >= protocol[
                    "round_alpha"]:
                failed.append(f"{LABELS[control]}的{label}未满足正向且显著")
            if control in ("previous_best",
                           "matched_collision_gp") and low <= 0:
                failed.append(f"相对{LABELS[control]}的{label}区间下限未高于零")
    assert (not failed) == summary["success"]
    if failed:
        lines += ["", "未通过的验收条件：", "", *[f"- {value}。" for value in failed]]
    lines += [
        "", "## 预算内可找全的场景池", "", "以下为 N≤200 的池上的描述性结果，空池计为已找全。按池及网络重复汇总，"
        "不替代以系统配置为统计单位的主检验。", "",
        "| 方法 | 内部运行数 | 找全比例 (%) | 平均最终召回 (%) | 平均剩余遗漏 |",
        "|---|---:|---:|---:|---:|"
    ]
    for method in methods:
        rows = [
            row for row in summary["records"] if row["method"] == method
            and row["pool_collisions"] <= protocol["budget"]
        ]
        assert rows
        lines.append(
            f"|{LABELS[method]}|{len(rows)}|"
            f"{100 * np.mean([r['all_failures_found'] for r in rows]):.3f}|"
            f"{100 * np.mean([r['recall'] for r in rows]):.3f}|"
            f"{np.mean([r['missed_failures'] for r in rows]):.3f}|")
    lines += [
        "", "## 控制器类型与目标差异", "", "| 类型 | 方法 | 发现面积 (%) | 最终召回 (%) | F200 |",
        "|---|---|---:|---:|---:|"
    ]
    for controller in ("IDM", "FVDM"):
        for method in methods:
            rows = [
                row for row in summary["per_profile"][method]
                if row["controller"] == controller
            ]
            lines.append(f"|{controller}|{LABELS[method]}|" + "|".join(
                format_profiles(rows, key, percent)
                for key, percent in (("normalized_area", True),
                                     ("recall", True), ("F200", False))) + "|")
    lines += [
        "", "每个目标的单元格为“面积百分比 / 召回百分比”。", "",
        "| 目标 |" + "|".join(LABELS[m] for m in methods) + "|",
        "|---|" + "---:|" * len(methods)
    ]
    for index, profile in enumerate(profiles):
        rows = [summary["per_profile"][method][index] for method in methods]
        lines.append(f"|{profile['name']}|" + "|".join(
            f"{100 * row['normalized_area']:.3f} / {100 * row['recall']:.3f}"
            for row in rows) + "|")
    original = (ROOT.parent / "history_response_testing/README.md").read_text(
        encoding="utf-8")
    table = original.split("## 原始池的完整现有对照", 1)[1].split("##", 1)[0].strip()
    lines += [
        "", "## 原始池完整现有 baseline", "", "以下为原冻结场景池的完整结果；与新系统确认队列分别报告。", "",
        table, "", "## 训练信息、核验与范围", "",
        protocol["candidate"]["history_scope"] + "。同信息 GP、固定行为排序和离散行为后验"
        "使用同一条件预测器，原六源方法与 RAS 保留冻结训练输入。", "",
        f"完整池测量 {protocol['total_physical_measurements']} 次，独立物理复跑 {physical['additional_physical_calls']} 次；"
        f"逐项核验 {statistics['disclosures']} 条查询反馈。"
        f"原始 {physical['frozen_original_files']} 个冻结文件、第一轮 {physical['first_confirmation_files']} 个封存输入保持不变，"
        f"本轮 {physical['locked_files']} 个输入与封存字节一致。精确检验独立复算，bootstrap 用计数实现复算，均通过核验。",
        "", "范围为声明区间的 IDM/FVDM 参数族、当前 highway-env 仿真器及两个有限候选场景族。"
        "精确符号翻转检验依赖零假设下配对差符号可交换。"
        "性能优势与方法创新分别判断；有限池找全不等价于连续场景空间覆盖。", "",
        "![独立确认曲线与目标差值](figures/confirmation.png)", ""
    ]
    figures(summary)
    (OUTPUT / "confirmation_report.md").write_text("\n".join(lines),
                                                   encoding="utf-8")
    print("BEHAVIOR CONFIRMATION REPORT", summary["success"], flush=True)


if __name__ == "__main__":
    main()
