"""Write a concise research report and a co-primary comparison figure."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from methods.history_guided_testing.io import read_json

from .confirmation import CONFIRMATION


OUTPUT = CONFIRMATION.parent / "confirmation_report.md"
FIGURE = CONFIRMATION.parent / "figures" / "confirmation.png"
METHOD_NAMES = {
    "risk_conditioned": "Risk-conditioned candidate",
    "collision_only_ablation": "Collision-only decoder ablation",
    "behavior_posterior": "Behavior-posterior candidate",
    "previous_best": "Previous best",
    "frozen_original": "Frozen original",
    "ras_frt_uq": "RAS-FRT-UQ",
    "matched_collision_gp": "Matched collision GP",
}


def format_pp(value):
    return f"{100 * value:+.3f} pp"


def make_figure(summary):
    controls = summary["protocol"]["primary_controls"]
    labels = [METHOD_NAMES[control] for control in controls]
    area = [summary["comparisons"][control]["normalized_area"]
            for control in controls]
    recall = [summary["comparisons"][control]["recall"]
              for control in controls]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=False)
    for axis, metric, title in zip(
            axes, (area, recall),
            ("Discovery area", "Recall after 200 queries")):
        means = np.asarray([entry["mean_difference"] for entry in metric]) * 100
        intervals = np.asarray(
            [entry["bootstrap_95_ci"] for entry in metric]) * 100
        yerr = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
        colors = ["#1b8a5a" if mean > 0 else "#c85845" for mean in means]
        axis.barh(np.arange(len(means)), means, color=colors, alpha=0.85)
        axis.errorbar(means, np.arange(len(means)), xerr=yerr,
                      fmt="none", ecolor="#262626", capsize=3, linewidth=1)
        axis.axvline(0, color="#444444", linewidth=0.9)
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Candidate minus control (percentage points)")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Independent round-three co-primary comparisons")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = read_json(CONFIRMATION / "summary.json")
    audit_path = CONFIRMATION.parent / "confirmation_audit" / "summary.json"
    audit = read_json(audit_path)
    make_figure(summary)
    aggregate = summary["aggregate"]
    rows = [
        "# 风险残差条件化方法：第三轮独立确认",
        "",
        f"**共同主指标验收：{'通过' if summary['success'] else '未通过'}**。",
        "",
        (f"本轮包含 {summary['statistical_units']} 个新 SUT 配置、每配置两个独立 "
         f"2048 场景池、五个冻结预测器种子和每策略 200 次不重复查询。"
         f"完整物理测量 {summary['protocol']['total_physical_measurements']:,} 次；"
         f"选择器披露 {summary['selector_disclosures_audited']:,} 次。"),
        "",
        "## 全部方法对比",
        "",
        "| 方法 | 发现面积 (%) | 200 次召回 (%) | F50 | F100 | F150 | F200 | 平均漏失 | 全发现率 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in summary["protocol"]["methods"]:
        value = aggregate[method]
        rows.append(
            f"| {METHOD_NAMES[method]} | "
            f"{100*value['normalized_area']['mean']:.3f} ± "
            f"{100*value['normalized_area']['profile_std']:.3f} | "
            f"{100*value['recall']['mean']:.3f} ± "
            f"{100*value['recall']['profile_std']:.3f} | "
            f"{value['F50']['mean']:.2f} | {value['F100']['mean']:.2f} | "
            f"{value['F150']['mean']:.2f} | {value['F200']['mean']:.2f} | "
            f"{value['missed_failures']['mean']:.2f} | "
            f"{100*value['all_failures_found']['mean']:.2f} |")
    rows += [
        "",
        "发现面积刻画整个查询过程；终点召回衡量 200 次预算内找到的失效比例。零失效池按预注册约定计面积 0、召回 1。",
        "",
        "## 共同主指标检验",
        "",
        "差值均为新候选减对照，单位为百分点；每项对四个主对照进行八项 Holm 校正。",
        "",
        "| 主对照 | 发现面积差 (95% CI) | Holm p | 终点召回差 (95% CI) | Holm p |",
        "|---|---:|---:|---:|---:|",
    ]
    for control in summary["protocol"]["primary_controls"]:
        area = summary["comparisons"][control]["normalized_area"]
        recall = summary["comparisons"][control]["recall"]
        rows.append(
            f"| {METHOD_NAMES[control]} | "
            f"{format_pp(area['mean_difference'])} "
            f"[{format_pp(area['bootstrap_95_ci'][0])}, "
            f"{format_pp(area['bootstrap_95_ci'][1])}] | "
            f"{area['holm_p']:.6g} | "
            f"{format_pp(recall['mean_difference'])} "
            f"[{format_pp(recall['bootstrap_95_ci'][0])}, "
            f"{format_pp(recall['bootstrap_95_ci'][1])}] | "
            f"{recall['holm_p']:.6g} |")
    rows += [
        "",
        "## 机制消融",
        "",
        "消融比较为描述性配置级 bootstrap 区间，不属于成功门槛或校正后的主检验。",
        "",
        "| 对照 | 面积差 (95% 描述区间) | 召回差 (95% 描述区间) |",
        "|---|---:|---:|",
    ]
    for control, values in summary["mechanism_comparisons_descriptive"].items():
        area, recall = values["normalized_area"], values["recall"]
        rows.append(
            f"| {METHOD_NAMES[control]} | {format_pp(area['mean_difference'])} "
            f"[{format_pp(area['bootstrap_95_ci_descriptive_only'][0])}, "
            f"{format_pp(area['bootstrap_95_ci_descriptive_only'][1])}] | "
            f"{format_pp(recall['mean_difference'])} "
            f"[{format_pp(recall['bootstrap_95_ci_descriptive_only'][0])}, "
            f"{format_pp(recall['bootstrap_95_ci_descriptive_only'][1])}] |")
    rows += [
        "",
        "## 复现与边界",
        "",
        (f"独立物理审计覆盖全部 {len(audit['physical_banks'])} 个完整池，"
         f"另对 {audit['physical_replays']} 个场景逐项重跑并匹配风险值和碰撞标签。"
         "查询序列、逐条反馈、两项主指标、精确检验和 bootstrap 均由独立审计重建。"),
        "",
        ("训练出的风险残差解码器只使用此前解封的 48 个开发配置，"
         "并在本轮新目标测量前冻结。线上选择器仅获得本次查询的连续风险；"
         "RAS-FRT-UQ 只获得其规定的碰撞反馈。结论范围限于本仿真器、"
         "声明的 IDM/FVDM 参数范围和两类有限场景池。"),
        "",
        f"![新候选与主对照的共同主指标差值](figures/confirmation.png)",
        "",
        "详细逐配置和逐运行结果见 `confirmation/summary.json`；锁定输入和物理审计见 `confirmation_audit/summary.json`。",
        "",
    ]
    OUTPUT.write_text("\n".join(rows), encoding="utf-8")
    print("ROUND-THREE REPORT", OUTPUT, "SUCCESS", summary["success"],
          flush=True)


if __name__ == "__main__":
    main()
