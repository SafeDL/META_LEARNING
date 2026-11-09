"""Export paired discovery, calibration and cost summaries for joint experiments."""

import csv
from pathlib import Path

import numpy as np

from .failure_metrics import METRIC_CONTRACT, aggregate, equal_count_distances
from .joint_figures import comparison_sets, plot_curves, plot_retention, plot_sequence
from .joint_validation import OUTPUT
from .risk_figures import risk_figures
from .storage import read_json, write_json

JOINT_METHOD_NAME = "历史响应迁移与联合反馈测试"

METRICS = (
    "candidate_count",
    "queries",
    "budget_fraction",
    "pool_unique_collisions",
    "pool_failure_rate",
    "initial_count",
    "initial_collisions",
    "unique_collisions",
    "post_initial_collisions",
    "discovery_auc",
    "first_collision_query",
    "restricted_first_collision_queries",
    "no_collision_within_budget",
    "collision_recall",
    "recall_auc",
    "pair_distance",
    "nearest_neighbor_distance",
    "collision_nll",
    "brier",
    "selection_s",
    "adaptation_s",
    "rss_compute_s",
    "equivalent_execution_s",
    "replay_wall_s",
    "negative_rss_executions",
)


def experiment_group(row):
    if "regime" in row:
        return row["regime"]
    scope = "leave_family_out" if row.get("held_family") is not None else "all_sources"
    return f"{scope}/{row['method']}"


def task_mean(rows, metric):
    values = [r["metrics"][metric] for r in rows if r["metrics"][metric] is not None]
    if not values:
        return None
    return float(np.mean(values))


def task_key(row):
    return row["sut_id"], row["scenario"], row.get("held_family"), row["seed"]


def paired_difference(rows, first, second, metric):
    left = {task_key(r): r for r in rows if experiment_group(r) == first}
    right = {task_key(r): r for r in rows if experiment_group(r) == second}
    if left.keys() != right.keys():
        raise ValueError("Paired comparison requires the same completed tasks")
    differences = []
    for task in sorted(left, key=str):
        a, b = left[task]["metrics"][metric], right[task]["metrics"][metric]
        differences.append(
            {"task": list(task), "difference": None if a is None or b is None else a - b}
        )
    values = [d["difference"] for d in differences if d["difference"] is not None]
    return {
        "mean": float(np.mean(values)) if values else None,
        "defined_tasks": len(values),
        "tasks": len(left),
        "per_task": differences,
    }


def report(root):
    root = Path(root) / "reports"
    data = read_json(root / "summary.json")
    rows = data["runs"]
    if not rows:
        raise ValueError("Report needs completed runs")
    if any(r["metrics"].get("metric_contract") != METRIC_CONTRACT for r in rows):
        raise ValueError("Report requires candidate-ID metrics, not recorded coarse-bin results")
    identities = [(experiment_group(r), task_key(r)) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("Report contains duplicate method/task runs")
    group_key = "method" if "method" in rows[0] else "regime"
    groups = sorted({experiment_group(r) for r in rows})
    summary = {}
    for group in groups:
        selected = [r for r in rows if experiment_group(r) == group]
        summary[group] = {}
        for metric in METRICS:
            summary[group][metric] = task_mean(selected, metric)
    comparisons = {}
    for group in groups:
        reference = f"{group.split('/')[0]}/joint" if group_key == "method" else "fixed_history"
        if group == reference:
            continue
        comparisons[group] = {
            metric: paired_difference(rows, group, reference, metric)
            for metric in (
                "unique_collisions",
                "discovery_auc",
                "collision_recall",
                "pair_distance",
                "nearest_neighbor_distance",
            )
        }
    equal_counts = {}
    for name, members in comparison_sets(rows).items():
        tasks = {}
        for row in members:
            tasks.setdefault(task_key(row), []).append(row)
        equal_counts[name] = [
            {
                "task": list(task),
                "methods": [experiment_group(r) for r in runs],
                "distances": equal_count_distances([r["metrics"] for r in runs]),
            }
            for task, runs in tasks.items()
        ]
    figures, figure_diagnostics = plot_curves(rows, root, group_key)
    if group_key == "regime":
        figures.extend(plot_sequence(rows, root, data.get("updates", [])))
    spatial_figures, prediction_diagnostics = risk_figures(rows, root)
    retention = []
    for update in data.get("updates", []):
        if update.get("retention_path") is not None:
            saved = read_json(root.parent / update["retention_path"])
            retention.extend(
                {
                    "after_systems": update["after_systems"],
                    "scenario": update["scenario"],
                    "branch_id": update["branch_id"],
                    **task,
                }
                for task in saved["tasks"]
            )
    figures.extend(plot_retention(retention, root))
    result = {
        "source_status": data["status"],
        "metric_contract": METRIC_CONTRACT,
        "means_by_task": summary,
        "task_coverage": {
            g: aggregate([r["metrics"] for r in rows if experiment_group(r) == g]) for g in groups
        },
        "by_scenario": {
            s: {
                g: aggregate(
                    [r["metrics"] for r in rows if r["scenario"] == s and experiment_group(r) == g]
                )
                for g in groups
            }
            for s in sorted({r["scenario"] for r in rows})
        },
        "equal_discovery_counts": equal_counts,
        "paired_differences": comparisons,
        "figures": figures + spatial_figures,
        "figure_diagnostics": figure_diagnostics,
        "prediction_diagnostics": prediction_diagnostics,
        "completed_task_replay": retention,
        "online_timing_scope": "Measured selector call wall time with paid truth replay, including initialization and final diagnostics; excludes process startup, offline training, source construction and physical truth-bank preparation.",
    }
    write_json(root / "report.json", result)
    with (root / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sut_id",
                "lineage",
                "scenario",
                "seed",
                group_key,
                "held_family",
                *METRICS,
                "online_wall_s",
                "sampled_peak_cpu_bytes",
                "peak_cuda_allocated_bytes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{
                        k: row.get(k)
                        for k in ("sut_id", "lineage", "scenario", "seed", group_key, "held_family")
                    },
                    **{k: row["metrics"][k] for k in METRICS},
                    "online_wall_s": row["resources"]["wall_s"],
                    **{
                        k: row["resources"][k]
                        for k in ("sampled_peak_cpu_bytes", "peak_cuda_allocated_bytes")
                    },
                }
            )
    lines = [
        f"# {JOINT_METHOD_NAME}",
        "",
        f"实验状态：{data['status']}。表格按任务等权汇总，曲线按功能场景及 ADS 分面。",
        "",
        "| 方法 | 真实失效数 | 初始化后新增 | 发现 AUC | 真值覆盖率 | 两两距离 | 最近邻距离 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, values in summary.items():
        formatted = [
            "—" if values[k] is None else f"{values[k]:.4f}"
            for k in (
                "unique_collisions",
                "post_initial_collisions",
                "discovery_auc",
                "collision_recall",
                "pair_distance",
                "nearest_neighbor_distance",
            )
        ]
        lines.append(f"| {group} | " + " | ".join(formatted) + " |")
    lines += [
        "",
        "逐任务配对差值、等失效数距离和指标可定义任务数见 report.json；成本及内存见 runs.csv。",
        "online_costs 图片及 online_wall_s 列为选择器调用期间的实际墙钟耗时，包含真值回放、初始化及末次诊断；不含进程启动、离线训练、来源构建或完整真值库采集。内存为选择器进程采样到的 CPU 驻留峰值。elapsed 图片为累计等效测试时间，使用原始物理执行耗时与当次适配、选例和 RSS 派生耗时。两者不混作重复实测仿真。",
        "单种子不生成置信区间。零失效库的覆盖率、少于两个发现的距离记为缺失，全部任务保留。",
    ]
    if retention:
        lines += [
            "",
            "已完成任务回放诊断保存在 report.json 的 completed_task_replay。",
            "按原查询顺序比较更新前后模型的查询前 NLL／Brier；记录是否已用于共享训练。该诊断不增加付费查询，也不参与模型发布。",
        ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    report(OUTPUT / "pilot")


if __name__ == "__main__":
    main()
