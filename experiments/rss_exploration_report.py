"""Verify completed shared/private replays and export their paired evidence."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import load_model

from .attribution import CHECKPOINTS, csv_report, registered_runs, validate_saved_metrics
from .attribution_report import SCENES, TITLES, totals
from .joint_figures import STYLE
from .protocol import ROOT
from .storage import read_json, write_json

OUTPUT = ROOT / "artifacts/runs/joint_response/method_exploration"
METHODS = ("original", "collision", "shared_private")
LABELS = ("Original joint", "Original target collision-only", "Shared/private")


def kernel_parameters(model):
    values = {
        "collision_shared_variance_fraction": model.shared_fraction,
        "rss_shared_variance_fraction": model.shared_loading.square(),
        "shared_loading": model.shared_loading,
        "point_correlation": model.correlation,
        "shared_lengthscale": model.log_lengthscale.exp(),
        "collision_private_lengthscale": model.log_collision_lengthscale.exp(),
        "rss_private_lengthscale": model.log_rss_lengthscale.exp(),
        "collision_sd": model.log_scale.exp(),
        "rss_sd": model.log_margin_scale.exp(),
        "collision_mean": model.mean,
        "rss_mean": model.margin_mean,
    }
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def paired_metrics(candidate, control):
    count = min(candidate["unique_collisions"], control["unique_collisions"])
    return {
        "delta_failures": candidate["unique_collisions"] - control["unique_collisions"],
        "delta_auc": candidate["discovery_auc"] - control["discovery_auc"],
        "matched_discovery_count": count,
        **{
            f"{method}_{name}": metrics["distances_by_discovery_count"][name][count - 1]
            if count >= 2
            else None
            for method, metrics in (("candidate", candidate), ("control", control))
            for name in ("pair_distance", "nearest_neighbor_distance")
        },
    }


def common_evidence(task, candidate, original):
    rows = []
    for left, right in zip(
        candidate["result"]["prediction_snapshots"],
        original["result"]["prediction_snapshots"],
        strict=True,
    ):
        if left["observed_indices"] != right["observed_indices"]:
            raise ValueError("E2 common-feedback prefixes differ from their original control")
        remaining = np.ones(len(left["probabilities"]), dtype=bool)
        remaining[left["observed_indices"]] = False
        joint = np.asarray(left["probabilities"])[remaining]
        collision = np.asarray(left["collision_conditioned_probabilities"])[remaining]
        k = min(200, len(joint))
        overlap = len(
            set(np.argsort(-joint, kind="stable")[:k])
            & set(np.argsort(-collision, kind="stable")[:k])
        )
        rows.append(
            {
                "sut_id": task["sut_id"],
                "scenario": task["scenario"],
                "paid_queries": left["paid_queries"],
                **{
                    f"{name}_{metric}": snapshot[metric]
                    for name, snapshot in (
                        ("original", right["evaluation"]),
                        ("joint", left["evaluation"]),
                        ("collision_conditioned", left["collision_conditioned_evaluation"]),
                    )
                    for metric in ("nll", "brier", "top_200_failures")
                },
                "mean_abs_conditional_probability_change": float(np.abs(joint - collision).mean()),
                "max_abs_conditional_probability_change": float(np.abs(joint - collision).max()),
                "conditional_top_200_overlap": overlap,
                "target_shared_loading": float(
                    np.tanh(left["target_parameters"]["raw_correlation"])
                ),
                "raw_shared_loading": float(
                    np.tanh(left["physical_parameters"]["raw_correlation"])
                ),
            }
        )
    if [r["paid_queries"] for r in rows] != list(CHECKPOINTS):
        raise ValueError("E2 common diagnostic checkpoint list changed")
    return rows


def component_evidence(records, pairs, common_rows):
    diversity = {}
    for kind in (*SCENES, "all"):
        tasks = records if kind == "all" else [r for r in records if r["scenario"] == kind]
        paired = [
            r
            for r in pairs
            if r["control"] == "original" and (kind == "all" or r["scenario"] == kind)
        ]
        diversity[kind] = {}
        for metric in ("pair_distance", "nearest_neighbor_distance"):
            matched = [
                r[f"candidate_{metric}"] - r[f"control_{metric}"]
                for r in paired
                if r[f"candidate_{metric}"] is not None and r[f"control_{metric}"] is not None
            ]
            budget = [
                r["shared_private"][metric] - r["original"][metric]
                for r in tasks
                if r["shared_private"][metric] is not None and r["original"][metric] is not None
            ]
            diversity[kind][metric] = {
                "matched_count_tasks": len(matched),
                "matched_count_mean_delta": float(np.mean(matched)) if matched else None,
                "equal_budget_tasks": len(budget),
                "equal_budget_mean_delta": float(np.mean(budget)) if budget else None,
            }
    groups = []
    for kind in SCENES:
        for count in CHECKPOINTS:
            rows = [r for r in common_rows if r["scenario"] == kind and r["paid_queries"] == count]
            groups.append(
                {
                    "scenario": kind,
                    "paid_queries": count,
                    **{
                        f"delta_{metric}": float(
                            np.mean(
                                [
                                    r[f"joint_{metric}"] - r[f"collision_conditioned_{metric}"]
                                    for r in rows
                                ]
                            )
                        )
                        for metric in ("nll", "brier")
                    },
                }
            )
    return {
        "diversity": diversity,
        "rss_conditioning": {
            "scene_prefix_groups": groups,
            "mean_delta_nll": float(np.mean([r["delta_nll"] for r in groups])),
            "mean_delta_brier": float(np.mean([r["delta_brier"] for r in groups])),
            "nll_worse_groups": sum(r["delta_nll"] > 0 for r in groups),
            "brier_worse_groups": sum(r["delta_brier"] > 0 for r in groups),
        },
    }


@plt.rc_context(STYLE)
def figure(records, common, path):
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6))
    for number, (axis, kind, title) in enumerate(zip(axes.flat, SCENES, TITLES)):
        for method, label, color, style in zip(
            METHODS, LABELS, ("#777777", "#D55E00", "#0072B2"), ("--", ":", "-")
        ):
            curves = [r[method]["discovery_curve"] for r in records if r["scenario"] == kind]
            axis.plot(
                range(1, 201), np.sum(curves, axis=0), label=label, color=color, linestyle=style
            )
        axis.axvline(20, color="#aaaaaa", linestyle=":", linewidth=0.8)
        axis.set(
            title=f"({chr(97 + number)}) {title}",
            xlabel="Paid queries per task",
            ylabel="Failures across eight ADSs",
        )
        axis.grid(alpha=0.15)
    for number, (axis, metric) in enumerate(zip(axes.flat[4:], ("nll", "brier")), start=4):
        for kind, title, color, marker in zip(
            SCENES,
            ("CF", "Brake", "Cut-out", "LC"),
            ("#0072B2", "#E69F00", "#009E73", "#CC79A7"),
            ("o", "s", "^", "D"),
        ):
            delta = [
                np.mean(
                    [
                        r[f"joint_{metric}"] - r[f"collision_conditioned_{metric}"]
                        for r in common
                        if r["scenario"] == kind and r["paid_queries"] == n
                    ]
                )
                for n in CHECKPOINTS
            ]
            axis.plot(CHECKPOINTS, delta, marker=marker, markersize=3, label=title, color=color)
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set(
            title=f"({chr(97 + number)}) RSS conditional {'NLL' if metric == 'nll' else 'Brier'} change",
            xlabel="Paid prefix",
            ylabel="Joint - collision-conditioned",
            xticks=CHECKPOINTS,
        )
        axis.grid(alpha=0.15)
    axes.flat[0].legend(frameon=False, fontsize=6)
    axes.flat[4].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        fontsize=6,
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )
    fig.tight_layout(w_pad=1.8, h_pad=1.4)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def report_e2():
    root = OUTPUT / "e2"
    tasks = read_json(OUTPUT / "protocol.json")["tasks"]
    existing = registered_runs()
    identities = [(task["sut_id"], task["scenario"]) for task in tasks]
    expected = {key[:2] for key in existing if key[2] == "joint"}
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("E2 reporting requires every registered task exactly once")
    records, pairs, common_rows = [], [], []
    for task in tasks:
        identity = {"sut_id": task["sut_id"], "scenario": task["scenario"]}
        original = read_json(ROOT / task["joint"])
        collision = read_json(existing[task["sut_id"], task["scenario"], "collision"]["path"])
        common = read_json(ROOT / task["common_feedback"])
        candidate = read_json(root / "adaptive" / task["sut_id"] / f"{task['scenario']}.json.gz")
        common_candidate = read_json(
            root / "common" / task["sut_id"] / f"{task['scenario']}.json.gz"
        )
        for saved in (candidate, common_candidate):
            validate_saved_metrics(
                {**original, "result": saved["result"], "metrics": saved["metrics"]}
            )
            indices = saved["result"]["selected_indices"]
            if (
                len(set(indices)) != 200
                or indices[:20] != original["result"]["selected_indices"][:20]
            ):
                raise ValueError("E2 unique budget or common initialization changed")
            if (
                saved["specification"]["source_snapshot"]["view_record_ids"]
                != original["specification"]["source_snapshot"]["view_record_ids"]
            ):
                raise ValueError("E2 historical calculation records changed")
        if common_candidate["result"]["selected_indices"] != common["result"]["selected_indices"]:
            raise ValueError("E2 common-feedback query order changed")
        common_rows.extend(common_evidence(task, common_candidate, common))
        row = {
            **identity,
            "original": original["metrics"],
            "collision": collision["metrics"],
            "shared_private": candidate["metrics"],
            "resources": candidate["result"]["resources"],
            "common_resources": common_candidate["result"]["resources"],
        }
        records.append(row)
        for control in ("original", "collision"):
            pairs.append(
                {
                    **identity,
                    "control": control,
                    **paired_metrics(row["shared_private"], row[control]),
                }
            )
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    learned = []
    for kind in SCENES:
        for stage in ("target_kernel", "historical_encoder"):
            model, _ = load_model(
                root / "models" / kind / f"{stage}.pt", kernel_type=SharedPrivateKernel
            )
            learned.append({"scenario": kind, "stage": stage, **kernel_parameters(model)})
    csv_report(reports / "kernel_parameters.csv", learned)
    construction = {}
    for kind in SCENES:
        saved = read_json(root / "models" / kind / "resources.json")
        construction[kind] = {
            **{name: saved[name] for name in ("wall_s", "sampled_peak_cpu_bytes", "device")},
            "target_training_s": saved["training"]["target"]["training_s"],
            "encoder_training_s": saved["training"]["encoder"]["training_s"],
        }
    csv_report(reports / "paired_differences.csv", pairs)
    csv_report(reports / "common_feedback.csv", common_rows)
    csv_report(
        reports / "task_metrics.csv",
        [
            {
                "sut_id": row["sut_id"],
                "scenario": row["scenario"],
                "method": method,
                **{
                    name: row[method][name]
                    for name in (
                        "unique_collisions",
                        "post_initial_collisions",
                        "pool_unique_collisions",
                        "collision_recall",
                        "discovery_auc",
                        "pair_distance",
                        "nearest_neighbor_distance",
                    )
                },
                **{f"found_{n}": row[method]["discovery_curve"][n - 1] for n in CHECKPOINTS},
            }
            for row in records
            for method in METHODS
        ],
    )
    combined = {method: totals([r[method] for r in records]) for method in METHODS}
    by_scenario = {
        kind: {
            method: totals([r[method] for r in records if r["scenario"] == kind])
            for method in METHODS
        }
        for kind in SCENES
    }
    difference_f = combined["shared_private"]["failures"] - combined["original"]["failures"]
    difference_auc = combined["shared_private"]["mean_auc"] - combined["original"]["mean_auc"]
    evidence = component_evidence(records, pairs, common_rows)
    summary = {
        "status": "complete",
        "tasks": len(records),
        "common_runs": 32,
        "adaptive_runs": 32,
        "new_logical_queries": 12800,
        "new_physical_executions": 0,
        "totals": combined,
        "by_scenario": by_scenario,
        **evidence,
        "delta_failures": difference_f,
        "delta_mean_auc": difference_auc,
        "task_wins_losses_ties": [
            int(
                sum(
                    np.sign(r["delta_failures"]) == sign
                    for r in pairs
                    if r["control"] == "original"
                )
            )
            for sign in (1, -1, 0)
        ],
        "construction_costs": {
            "per_scenario": construction,
            "summed_scene_wall_s": sum(v["wall_s"] for v in construction.values()),
            "largest_cpu_peak_gib": max(
                v["sampled_peak_cpu_bytes"] / 2**30 for v in construction.values()
            ),
        },
        "common_feedback_costs": {
            "summed_method_wall_s": sum(r["common_resources"]["wall_s"] for r in records),
            "largest_cpu_peak_gib": max(
                r["common_resources"]["sampled_peak_cpu_bytes"] / 2**30 for r in records
            ),
        },
        "adaptive_costs": {
            "summed_method_wall_s": sum(r["resources"]["wall_s"] for r in records),
            "largest_cpu_peak_gib": max(
                r["resources"]["sampled_peak_cpu_bytes"] / 2**30 for r in records
            ),
        },
        "decision": "candidate_for_final_freeze"
        if difference_f >= 0 and difference_auc >= 0 and (difference_f > 0 or difference_auc > 0)
        else "tradeoff"
        if difference_f > 0 or difference_auc > 0
        else "ended_no_discovery_benefit",
        "validation_scope": "old-platform development replay; new-platform validation pending",
    }
    write_json(reports / "summary.json", summary)
    figure(records, common_rows, reports / "shared_private.png")
    lines = [
        "# 可关闭 RSS 共享探索结果",
        "",
        "八个旧目标、四场景均完成共同 Random 前缀和自适应回放，初始化 20、总预算 200、种子 11。"
        "逻辑查询 12,800，新增物理执行 0；旧库用于方法开发，尚未构成新平台验证。",
        "",
        "| 设置 | 总失效 | 初始化后新增 | 平均发现 AUC | 真值加权覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method, title in zip(METHODS, ("原联合模型", "原模型目标仅碰撞", "共享与独立潜过程")):
        m = combined[method]
        lines.append(
            f"| {title} | {m['failures']} | {m['post_initial_failures']} | "
            f"{m['mean_auc']:.5f} | {m['pooled_coverage']:.2%} |"
        )
    lines += [
        "",
        f"相对原方法，失效 {difference_f:+d}、平均 AUC {difference_auc:+.5f}；"
        f"任务胜／负／平为 {summary['task_wins_losses_ties']}。",
        "",
        "| 场景 | 原方法失效 | 原模型目标仅碰撞 | 新模型失效 | 新模型 AUC 变化 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind, title in zip(SCENES, TITLES):
        values = by_scenario[kind]
        lines.append(
            f"| {title} | {values['original']['failures']} | {values['collision']['failures']} | "
            f"{values['shared_private']['failures']} | "
            f"{values['shared_private']['mean_auc'] - values['original']['mean_auc']:+.5f} |"
        )
    lines += [
        "",
        "`paired_differences.csv` 包含全部逐任务差异及相同发现数下的参数距离。"
        "零失效库覆盖率、不足两个发现的距离保留未定义；不删除不利任务。",
        "",
        "`common_feedback.csv` 比较固定同一新模型参数、来源和权重时，联合反馈与仅碰撞条件化的"
        "剩余候选预测。该比较只反映当前后验中的 RSS 条件增量，不能隔离 RSS 对训练和适配参数的影响。"
        "新旧模型使用相同历史计算记录 ID，但训练结构、RSS 尺度及梯度裁剪不同，主比较按流程对照解释。",
        "",
        f"固定新模型参数后，{evidence['rss_conditioning']['nll_worse_groups']}/16 个场景—前缀组的平均 NLL "
        f"及 {evidence['rss_conditioning']['brier_worse_groups']}/16 组的平均 Brier 在加入 RSS 条件化后变差。"
        "因此，新增发现支持新联合建模流程的整体价值，尚不能归因于 RSS 观测本身；"
        "也不能单独区分独立潜过程、RSS 尺度和梯度分组各自的贡献。",
        "",
        "共享与独立潜过程增加三个离线标量，在线仍适配原六个标量；原共享表征冻结，"
        "来源权重仍仅使用查询前碰撞似然，未加入局部权重。先验共享为场景级标量，不能称为输入相关门控。",
        "",
        "`kernel_parameters.csv` 保存各场景基础核和历史编码器学到的共享方差比例、加载系数、"
        "点相关系数及三个核长度尺度。它们描述模型实际的共享程度，发现收益仍由完整回放判断。",
        "",
        f"四场景构建墙钟合计 {summary['construction_costs']['summed_scene_wall_s']:.2f} 秒，"
        f"单进程 CPU 采样峰值最高 {summary['construction_costs']['largest_cpu_peak_gib']:.3f} GiB。"
        "构建包括读档、两阶段训练及来源导出，训练分项见汇总中的 `construction_costs`；"
        "其余构建开销不再单独解释为纯来源推断耗时。",
        "",
        f"自适应运行方法墙钟合计 {summary['adaptive_costs']['summed_method_wall_s']:.2f} 秒，"
        f"单进程 CPU 采样峰值最高 {summary['adaptive_costs']['largest_cpu_peak_gib']:.3f} GiB；"
        "训练及来源构建成本另见各场景 `models/<scenario>/resources.json`。"
        "合计方法时间不等于并行实验墙钟，新旧耗时来自不同时间运行。",
        f"共同反馈诊断墙钟合计另为 {summary['common_feedback_costs']['summed_method_wall_s']:.2f} 秒，"
        "包含固定参数下的 RSS 条件增量预测，单独列示。",
        "",
        f"本方向决策：`{summary['decision']}`；最终采用方案在 E3 结束后冻结，稿件保持不变。",
        "",
    ]
    lines += [
        "| 场景 | 相同发现数的平均两两距离变化 | 相同发现数的最近邻距离变化 | 有效任务数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for kind, title in zip(SCENES, TITLES):
        pair = evidence["diversity"][kind]["pair_distance"]
        nearest = evidence["diversity"][kind]["nearest_neighbor_distance"]
        p = pair["matched_count_mean_delta"]
        n = nearest["matched_count_mean_delta"]
        displayed_p = "未定义" if p is None else f"{(0 if abs(p) < 1e-12 else p):+.5f}"
        displayed_n = "未定义" if n is None else f"{(0 if abs(n) < 1e-12 else n):+.5f}"
        lines.append(f"| {title} | {displayed_p} | {displayed_n} | {pair['matched_count_tasks']} |")
    lines += [
        "",
        "距离变化为新模型减原方法。相同发现数下，切出的两两距离提高而最近邻距离下降，"
        "换道则相反；总体未显示一致的多样性提升。相同预算及相同发现数的全部距离统计见 "
        "`summary.json` 的 `diversity`、`task_metrics.csv` 和 `paired_differences.csv`。",
        "",
    ]
    (reports / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = report_e2()
    print(result["decision"], result["delta_failures"], result["delta_mean_auc"])
