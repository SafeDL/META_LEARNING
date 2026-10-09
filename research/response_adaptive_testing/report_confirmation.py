"""Report the complete locked experiment and separately label fixed-prior controls."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods.history_guided_testing.io import read_json

from .config import OUTPUT, ROOT
from .confirmation import CONFIRMATION, verify_lock

LABELS = {
    "candidate": "联合风险—失效 GP",
    "previous_best": "此前最佳",
    "frozen_original": "冻结原始方法",
    "ras_frt_uq": "RAS-FRT-UQ",
    "class_rank": "历史碰撞固定排序",
    "locked_prior_static": "联合 GP 初始后验固定排序",
    "learned_mean_static": "学习失效均值后固定排序"
}
PLOT_LABELS = ("Joint response GP", "Previous best", "Frozen original",
               "RAS-FRT-UQ", "Historical collision rank", "Fixed joint prior",
               "Learned fixed prior")


def profile_rows(records, profiles, method):
    rows = []
    for profile in profiles:
        selected = [
            row for row in records
            if row["method"] == method and row["profile"] == profile["name"]
        ]
        assert len(selected) == 10
        rows.append({
            key: float(np.mean([row[key] for row in selected]))
            for key in ("normalized_area", "recall", "F50", "F100", "F200",
                        "missed_failures", "all_failures_found")
        })
    return rows


def number(rows, key, percent=False):
    values = np.array([row[key] for row in rows]) * (100 if percent else 1)
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def figures(records, methods, profiles):
    curves = {}
    for method in methods:
        curves[method] = np.array([
            np.mean([
                np.asarray(row["curve"]) / row["pool_collisions"]
                if row["pool_collisions"] else np.zeros(200)
                for row in records if row["method"] == method
                and row["profile"] == profile["name"]
            ],
                    axis=0) for profile in profiles
        ])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method, label in zip(methods, PLOT_LABELS):
        axes[0].plot(np.arange(1, 201),
                     100 * curves[method].mean(0),
                     label=label)
    oracle = np.mean([
        np.minimum(np.arange(1, 201), row["pool_collisions"]) /
        row["pool_collisions"] if row["pool_collisions"] else np.zeros(200)
        for row in records if row["method"] == "candidate"
    ],
                     axis=0)
    axes[0].plot(np.arange(1, 201), 100 * oracle, "k--", label="Budget oracle")
    axes[0].set(xlabel="Target queries", ylabel="Mean pool recall (%)")
    axes[0].grid(alpha=0.2)
    delta_area = 100 * (curves["candidate"].mean(1) -
                        curves["previous_best"].mean(1))
    delta_recall = 100 * (curves["candidate"][:, -1] -
                          curves["previous_best"][:, -1])
    for controller, marker in (("IDM", "o"), ("FVDM", "s")):
        mask = np.array(
            [profile["controller"] == controller for profile in profiles])
        axes[1].scatter(delta_area[mask],
                        delta_recall[mask],
                        marker=marker,
                        label=controller)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].set(xlabel="Area difference vs previous best (pp)",
                ylabel="Final recall difference (pp)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    figure.tight_layout(rect=(0, 0.14, 1, 1))
    folder = OUTPUT / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(folder / f"confirmation.{extension}", dpi=180)
    plt.close(figure)


def main():
    summary = read_json(CONFIRMATION / "summary.json")
    protocol = summary["protocol"]
    verify_lock(protocol)
    statistics = read_json(OUTPUT / "confirmation_audit/statistics.json")
    physical = read_json(OUTPUT / "confirmation_audit/summary.json")
    static = read_json(OUTPUT / "confirmation_static_controls/summary.json")
    assert physical["all_physical_banks_audited"] and static[
        "full_suite_complete"]
    assert statistics["success"] == summary["success"]
    methods = [
        *protocol["methods"], "locked_prior_static", "learned_mean_static"
    ]
    records = [*summary["records"], *static["records"]]
    profiles = protocol["profiles"]
    rows = {
        method: profile_rows(records, profiles, method)
        for method in methods
    }
    lines = [
        "# 完整独立确认结果", "",
        f"本轮预登记双主要指标验收：**{'通过' if summary['success'] else '未通过'}**。", "",
        "24 个 IDM/FVDM 参数组合，每组合两个独立 2048 候选池，五个冻结历史网络种子，"
        "每方法 200 次不重复查询。下表先对每个参数组合的十个内部重复取均值，"
        "再报告 24 个组合的均值 ± 样本标准差。两个百分比为完整池归一化发现面积和最终召回率。", "",
        "|方法|发现面积 (%)|最终召回 (%)|F50|F100|F200|遗漏数|找全比例 (%)|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for method in methods:
        values = [
            number(rows[method], key, percent)
            for key, percent in (("normalized_area", True), ("recall", True),
                                 ("F50", False), ("F100", False),
                                 ("F200", False), ("missed_failures", False),
                                 ("all_failures_found", True))
        ]
        lines.append(f"|{LABELS[method]}|" + "|".join(values) + "|")
    lines += [
        "", "最后两个固定排序是附加诊断；学习均值在线方案没有参加本轮主要确认。"
        "未测目标响应仅用于事后完整评价，选择器只获得自己的已查询反馈。", "", "## 预登记的六项主要比较", "",
        "差值单位为百分点；配对 bootstrap 在 IDM/FVDM 类型内重采样。"
        f"六项精确双侧符号翻转检验使用 Holm 校正，轮内阈值为 {protocol['round_alpha']}。", "",
        "|对照|主要指标|候选减对照|95% CI|精确 p|Holm p|", "|---|---|---:|---:|---:|---:|"
    ]
    for control, metrics in summary["comparisons"].items():
        for metric, values in metrics.items():
            low, high = np.array(values["bootstrap_95_ci"]) * 100
            label = "发现面积" if metric == "normalized_area" else "最终召回"
            lines.append(
                f"|{LABELS[control]}|{label}|"
                f"{100 * values['mean_difference']:+.4f}|[{low:+.4f}, {high:+.4f}]|"
                f"{values['exact_two_sided_p']:.8g}|{values['holm_p']:.8g}|")
    lines += [
        "", "## 各目标参数组合", "", "单元格为“发现面积百分比 / 最终召回百分比”。", "",
        "|目标|" + "|".join(LABELS[method] for method in methods) + "|",
        "|---|" + "---:|" * len(methods)
    ]
    for index, profile in enumerate(profiles):
        values = [
            f"{100 * rows[method][index]['normalized_area']:.3f} / "
            f"{100 * rows[method][index]['recall']:.3f}" for method in methods
        ]
        lines.append(f"|{profile['name']}|" + "|".join(values) + "|")
    original = (ROOT.parent / "history_response_testing/README.md").read_text(
        encoding="utf-8")
    table = original.split("## 原始池的完整现有对照", 1)[1].split("##", 1)[0].strip()
    lines += [
        "", "## 原始池完整现有 baseline", "", "以下保持冻结目录的原始结果，不能与新目标队列混为同一次比较。", "",
        table, "", "## 核验与范围", "",
        f"完整物理测量 {protocol['total_physical_measurements']} 条，额外物理复跑 192 条；"
        f"主要选择器的 {statistics['disclosures']} 条反馈逐项核验。"
        "半集合求和复算精确检验，多项计数复算 bootstrap，"
        "均与主要评价一致。原始 176 个冻结文件保持不变。", "",
        "结果范围为声明参数区间内的 IDM/FVDM、当前仿真器及两个有限候选场景族。"
        "符号翻转检验依赖零假设下配对差符号可交换。"
        "性能结果不能单独证明方法新颖性或连续场景空间中的全部失效覆盖。", "",
        "![完整确认发现曲线与目标差值](figures/confirmation.png)", ""
    ]
    figures(records, methods, profiles)
    (OUTPUT / "confirmation_report.md").write_text("\n".join(lines),
                                                   encoding="utf-8")
    print("CONFIRMATION REPORT", summary["success"], flush=True)


if __name__ == "__main__":
    main()
