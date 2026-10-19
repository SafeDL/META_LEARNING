"""Discovery retention and fully charged serial-equivalent lifecycle costs."""

import math

import numpy as np

from methods.baselines.hgt.io import read_json, write_json
from methods.gcrt.config import (
    ARCHIVE_ROOT,
    BASELINE_ROOT,
    BUDGET,
    REUSE,
    ROOT,
    ROLES,
    SEGMENT,
)

SYSTEMS = {
    "idm_ref": "IDM",
    "fvdm_target": "FVDM",
    "mobil_ref_v2": "IDM＋MOBIL",
    "vi_ttc_ref_audit_v4": "碰撞时间值迭代",
    "mcts_cv_ref_audit_v4": "恒速预测蒙特卡洛树搜索",
    "ppo_ref_v2": "PPO",
}
MODEL_NAMES = {
    "initial": "发现比例校正",
    "updated": "发现比例校正＋周期更新",
    "without_correction": "无任务内校正",
    "direct_tilt": "原计数直接倾斜",
}
MODE_NAMES = {"empirical": "经验校准", "strict": "严格校准"}
STAGE_NAMES = {
    "kernel": "基础核",
    "encoder": "历史响应关联编码器",
    "prior": "统一先验拟合",
    "historical": "同尺度历史选择开发、完整轨迹与因果状态",
    "count_network": "四折分组验证与最终收益网络训练",
}
SPLIT_NAMES = {"first_two": "前两个场景测试集", "third": "第三个场景测试集"}
ACCEPTANCE_NAMES = {
    **{split: f"完整选择，{label}" for split, label in SPLIT_NAMES.items()},
    **{
        f"{model}_{mode}_{split}": f"{model_name}，{mode_name}，{split_name}"
        for model, model_name in MODEL_NAMES.items()
        for mode, mode_name in MODE_NAMES.items()
        for split, split_name in SPLIT_NAMES.items()
    },
}


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def computation(trace):
    timing = trace["timing"]
    return timing["task_initialization_s"] + sum(timing["query_compute_s"])


def construction(family):
    folder = ROOT / family
    ledger = read_json(folder / "build_cost.json")["stages"]
    costs = {
        name: stage["elapsed_s"]
        for name, stage in ledger.items()
        if name != "cost_measurement"
    }
    for kind in ("kernel", "encoder"):
        reused = folder / kind / "reuse.json"
        if reused.exists():
            costs[kind] = read_json(reused)["standalone_construction_s"]
    original = ARCHIVE_ROOT / family / "build_cost.json"
    if original.exists():
        costs["prior"] = read_json(original)["stages"]["prior"]["elapsed_s"]
    components = folder / "historical_cost.json"
    if components.exists():
        costs["historical"] = read_json(components)["standalone_construction_s"]
    return costs


def full_online(trace, forecast):
    return (
        computation(trace)
        + sum(trace["state_recording_s"][::SEGMENT])
        + predictor_seconds(forecast)
    )


def predictor_seconds(forecast):
    return (
        sum(forecast["prequery_s"])
        + sum(forecast["correction_s"])
        + sum(r["compute_s"] for r in forecast["forecast"])
    )


def summarize(rows):
    return {
        "tasks": len(rows),
        "found": mean([r["found"] for r in rows]),
        "recall": mean([r["recall"] for r in rows]),
        "pooled_recall": (
            sum(r["found"] for r in rows) / sum(r["positives"] for r in rows)
            if sum(r["positives"] for r in rows)
            else None
        ),
        "queries": mean([r["queries"] for r in rows]),
        "hit_rate": sum(r["found"] for r in rows) / sum(r["queries"] for r in rows),
        "budget_ceiling_fraction": mean(
            [r["found"] / min(r["positives"], BUDGET) for r in rows if r["positives"]]
        ),
        "failed_tasks": sum(r.get("quality_failed", False) for r in rows),
        "retention": mean([r.get("retention") for r in rows]),
        "H10": mean([r["H10"] for r in rows]),
        "H20": mean([r["H20"] for r in rows]),
        "H50": mean([r["H50"] for r in rows]),
        "H100": mean([r["H100"] for r in rows]),
    }


def paired_block_interval(traces, originals):
    controllers = sorted({t["sut"] for t in traces})
    delta = [
        mean(
            [
                t["H200"] - originals[t["pool"], sut]["H200"]
                for t in traces
                if t["sut"] == sut
            ]
        )
        for sut in controllers
    ]
    rng = np.random.default_rng(11)
    samples = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(1)
    return list(map(float, np.quantile(samples, [0.025, 0.975])))


def discovery_plot(folder, traces, originals, rows):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    selected = [t for t in traces if t["sut"] in ROLES["evaluation"]]
    adaptive = {
        (r["pool"], r["sut"]): r
        for r in rows
        if r["model"] == "updated" and r["mode"] == "empirical"
    }
    fixed_curves = [t["found"] for t in selected]
    old_curves = [originals[t["pool"], t["sut"]]["found"] for t in selected]
    stopped_curves = []
    for trace in selected:
        row = adaptive[trace["pool"], trace["sut"]]
        prefix = trace["found"][: row["queries"]]
        stopped_curves.append(prefix + [prefix[-1]] * (BUDGET - len(prefix)))
    figure, axis = plt.subplots(figsize=(7, 4))
    for curves, label, style in (
        (old_curves, "LRT（动作价值学习） 固定200次", "--"),
        (fixed_curves, "GCRT固定200次", "-"),
        (stopped_curves, "GCRT＋自适应停止", "-"),
    ):
        axis.plot(np.arange(1, BUDGET + 1), np.mean(curves, axis=0), style, label=label)
    axis.set(xlabel="测试次数", ylabel="危险场景发现数量", xlim=(1, BUDGET))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(folder / "discovery.png", dpi=180)
    plt.close(figure)


def full_discovery_plot(folder, traces, originals, current):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, pools, title in zip(
        axes, ((0, 1), (2,)), ("前两个场景测试集", "第三个场景测试集")
    ):
        selected = [t for t in traces if t["pool"] in pools]
        for source, label, color, style in (
            (originals, "动作价值学习", "#777777", "--"),
            (current, "全预算前瞻", "#d08722", "-."),
            ({(t["pool"], t["sut"]): t for t in traces}, "GCRT", "#1675ad", "-"),
        ):
            curves = [source[t["pool"], t["sut"]]["found"] for t in selected]
            ax.plot(
                np.arange(1, 201),
                np.mean(curves, axis=0),
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.8,
            )
        ax.axvline(10, color="#bbbbbb", linewidth=0.8)
        ax.set(title=title, xlabel="测试次数", xlim=(1, 200))
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("平均危险场景发现数量")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(folder / "full_discovery.png", dpi=200)
    plt.close(fig)


def campaign_total(bootstrap, continuing, count, key):
    first = min(count, len(bootstrap))
    cycles, tail = divmod(max(0, count - len(bootstrap)), len(continuing))
    value = lambda row: 0.0 if row[key] is None else float(row[key])
    return (
        sum(map(value, bootstrap[:first]))
        + cycles * sum(map(value, continuing))
        + sum(map(value, continuing[:tail]))
    )


def campaign_quality(bootstrap, continuing, count):
    total = lambda key: campaign_total(bootstrap, continuing, count, key)
    controller_recall, controller_full = [], []
    for sut in sorted({row["sut"] for row in bootstrap}):
        mask = lambda rows: [
            dict(
                recall=r["recall"] if r["sut"] == sut else None,
                full_recall=r["full_recall"] if r["sut"] == sut else None,
                valid_recall=r["valid_recall"] if r["sut"] == sut else False,
            )
            for r in rows
        ]
        selected0, selected1 = mask(bootstrap), mask(continuing)
        valid = campaign_total(selected0, selected1, count, "valid_recall")
        if valid:
            controller_recall.append(
                campaign_total(selected0, selected1, count, "recall") / valid
            )
            controller_full.append(
                campaign_total(selected0, selected1, count, "full_recall") / valid
            )
    h = total("found") / total("full_found") if total("full_found") else 1.0
    recall, base = mean(controller_recall), mean(controller_full)
    return dict(
        discovery_retention=h,
        macro_recall=recall,
        pooled_recall=(
            total("found") / total("positives") if total("positives") else None
        ),
        recall_retention=recall / base if base else 1.0,
        passed=h >= 0.95 and (recall >= 0.95 * base if base else True),
    )


def family_report(family):
    folder = ROOT / family
    traces = [read_json(p) for p in sorted((folder / "trajectories").glob("*/*.json"))]
    rows = read_json(folder / "stopping.json")
    forecasts = read_json(folder / "forecasts.json")
    collection = read_json(folder / "collection.json")
    originals = {
        (t["pool"], t["sut"]): read_json(
            BASELINE_ROOT
            / family
            / "trajectories"
            / f"pool_{t['pool']}"
            / t["sut"]
            / "lrt_11.json"
        )
        for t in traces
    }
    current = {
        (t["pool"], t["sut"]): read_json(
            ARCHIVE_ROOT
            / family
            / "trajectories"
            / f"pool_{t['pool']}"
            / f"{t['sut']}.json"
        )
        for t in traces
    }
    builds = construction(family)
    preparations = {
        pool: sum(read_json(folder / "pool_cache" / f"pool_{pool}.json").values())
        for pool in (0, 1, 2)
    }
    build_fixed = sum(builds[k] for k in ("kernel", "encoder", "prior"))
    build_adaptive = sum(builds.values())
    summaries = {}
    acceptance = {}
    text = [
        f"# {'车辆切入' if family == 'cutin' else '跟驰过程中的前车紧急制动'}：发现质量与完整成本\n",
        "危险判定为实测风险 R>0.5。每个场景测试集有 1024 个本类候选，"
        "种子为 11；初始化 10 次计费。时间单位为秒。\n",
        "## 完整 200 次的场景选择能力\n",
        "| 场景测试集 | 被测驾驶控制器 | 危险总数 | LRT（动作价值学习）发现数 | 全预算前瞻发现数 | GCRT发现数 | LRT Recall | 全预算前瞻 Recall | GCRT Recall | H10 | H20 | H50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for trace in traces:
        old = originals[trace["pool"], trace["sut"]]
        previous = current[trace["pool"], trace["sut"]]

        def pct(value):
            return f"{100*value:.2f}%" if value is not None else "无危险场景"

        text.append(
            f"| {trace['pool']+1} | {SYSTEMS[trace['sut']]} | {trace['positives']} | "
            f"{old['H200']} | {previous['H200']} | {trace['H200']} | {pct(old['recall'])} | {pct(previous['recall'])} | {pct(trace['recall'])} | "
            f"{trace['found'][9]} | {trace['found'][19]} | {trace['found'][49]} |"
        )
    for name, pools in (("first_two", (0, 1)), ("third", (2,))):
        selected = [t for t in traces if t["pool"] in pools]
        old = [originals[t["pool"], t["sut"]] for t in selected]
        h = mean([t["H200"] for t in selected])
        recall = mean([t["recall"] for t in selected])
        base_h = mean([t["H200"] for t in old])
        base_recall = mean([t["recall"] for t in old])
        previous = [current[t["pool"], t["sut"]] for t in selected]
        previous_h = mean([t["H200"] for t in previous])
        previous_r = mean([t["recall"] for t in previous])
        acceptance[name] = {
            "H200": h,
            "recall": recall,
            "original_H200": base_h,
            "current_H200": previous_h,
            "current_recall": previous_r,
            "original_recall": base_recall,
            "H_retention": h / base_h if base_h else 1.0,
            "recall_retention": recall / base_recall if base_recall else 1.0,
            "passed": h >= 0.95 * base_h
            and recall >= 0.95 * base_recall
            and h >= 0.95 * previous_h
            and recall >= 0.95 * previous_r,
            "H_difference_block_interval": paired_block_interval(selected, originals),
        }
    mechanism = {
        "lookahead_non_greedy_queries": mean(
            [
                sum(d.get("differs_from_marginal", False) for d in t["decisions"])
                for t in traces
            ]
        ),
        "changed_queries_from_original": mean(
            [
                len(
                    set(t["selected_indices"])
                    - set(originals[t["pool"], t["sut"]]["selected_indices"])
                )
                for t in traces
            ]
        ),
    }
    text.append(
        f"\n每条完整轨迹平均有 {mechanism['lookahead_non_greedy_queries']:.2f} 次选择"
        "不同于当前最高危险概率候选；"
        f"相对LRT（动作价值学习），平均更换了 {mechanism['changed_queries_from_original']:.2f} 个测试场景。"
    )
    text.extend(
        [
            "\n## 留出控制器上的停止与周期更新\n",
            "校准使用 MCTS、PPO；后续评价使用 IDM＋MOBIL、IDM。"
            "每个控制器的三个场景测试集始终属于同一角色。经验阈值不搜索，"
            "严格阈值因只有两个校准控制器块而为 1。表中是同一收益网络的不同校正和训练设置，正式运行只加载发布的一个网络。\n",
            "| 设置 | 校准方式 | 任务数 | 平均查询数 | 平均发现数 | 宏平均 Recall | 平均发现保留率 | 未保留95%的任务数 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    matched = [t for t in traces if t["sut"] in ROLES["evaluation"]]
    summaries["fixed"] = {
        "tasks": len(matched),
        "queries": 200,
        "found": mean([t["H200"] for t in matched]),
        "recall": mean([t["recall"] for t in matched]),
        "pooled_recall": sum(t["H200"] for t in matched)
        / sum(t["positives"] for t in matched),
    }
    fixed = summaries["fixed"]
    text.append(
        f"| 同组控制器固定200次 | 无停止 | {len(matched)} | 200.00 | "
        f"{fixed['found']:.2f} | {100*fixed['recall']:.2f}% | 100.00% | 0 |"
    )
    for model in ("initial", "updated", "without_correction", "direct_tilt"):
        for mode in ("empirical", "strict"):
            selected = [
                r
                for r in rows
                if r["model"] == model
                and r["mode"] == mode
                and r["role"] == "evaluation"
            ]
            result = summarize(selected)
            result["discovery_count_retention"] = result["found"] / fixed["found"]
            result["macro_recall_retention"] = result["recall"] / fixed["recall"]
            summaries[f"{model}_{mode}"] = result
            text.append(
                f"| {MODEL_NAMES[model]} | {MODE_NAMES[mode]} | {len(selected)} | {result['queries']:.2f} | "
                f"{result['found']:.2f} | {100*result['recall']:.2f}% | "
                f"{100*result['retention']:.2f}% | {result['failed_tasks']} |"
            )
            for name, pools in (("first_two", (0, 1)), ("third", (2,))):
                selected_pool = [r for r in selected if r["pool"] in pools]
                base = [
                    t
                    for t in traces
                    if t["sut"] in ROLES["evaluation"] and t["pool"] in pools
                ]
                old = [originals[t["pool"], t["sut"]] for t in base]
                h, rec = mean([r["found"] for r in selected_pool]), mean(
                    [r["recall"] for r in selected_pool]
                )
                requirements = {
                    "new_fixed_discoveries": h
                    >= 0.95 * mean([t["H200"] for t in base]),
                    "new_fixed_recall": rec >= 0.95 * mean([t["recall"] for t in base]),
                    "original_discoveries": h >= 0.95 * mean([t["H200"] for t in old]),
                    "original_recall": rec >= 0.95 * mean([t["recall"] for t in old]),
                    "previous_discoveries": h
                    >= 0.95
                    * mean([current[t["pool"], t["sut"]]["H200"] for t in base]),
                    "previous_recall": rec
                    >= 0.95
                    * mean([current[t["pool"], t["sut"]]["recall"] for t in base]),
                }
                acceptance[f"{model}_{mode}_{name}"] = {
                    **requirements,
                    "passed": all(requirements.values()),
                }
    final_summary = summaries["updated_empirical"]
    text.append(
        f"\n同一留出组的合并计数覆盖率：固定预算 "
        f"{100*summaries['fixed']['pooled_recall']:.2f}%，停止后 "
        f"{100*final_summary['pooled_recall']:.2f}%；停止后命中率 "
        f"{100*final_summary['hit_rate']:.2f}%，预算内可发现上限平均完成 "
        f"{100*final_summary['budget_ceiling_fraction']:.2f}%。"
        "合并计数按危险场景数量加权，宏平均按控制器等权。"
    )
    text.extend(
        [
            "\n### 逐任务停止结果\n",
            "| 网络 | 方式 | 场景测试集 | 控制器 | 查询数 | 发现数/完整发现数 | Recall | 少发现数 | 多测次数 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["role"] == "evaluation" and row["model"] in ("initial", "updated"):
            recall = (
                f"{100*row['recall']:.2f}%"
                if row["recall"] is not None
                else "无危险场景"
            )
            text.append(
                f"| {MODEL_NAMES[row['model']]} | {MODE_NAMES[row['mode']]} | {row['pool']+1} | {SYSTEMS[row['sut']]} | "
                f"{row['queries']} | {row['found']}/{row['full_found']} | {recall} | "
                f"{row['missed']} | {row['excess_queries']} |"
            )
    cost_rows = []
    update_report = read_json(folder / "update.json")
    initial_lookup = {(p["pool"], p["sut"]): p for p in forecasts["initial"]}
    updated_lookup = {(p["pool"], p["sut"]): p for p in forecasts["updated"]}
    collection_lookup = {(r["pool"], r["sut"]): r for r in collection["tasks"]}
    updated_rows = {
        (r["pool"], r["sut"]): r
        for r in rows
        if r["model"] == "updated" and r["mode"] == "empirical"
    }
    initial_gamma = read_json(folder / "training.json")["collection_threshold"]
    from .evaluation import stopping_row

    for trace in traces:
        key = trace["pool"], trace["sut"]
        fixed = computation(trace) + sum(trace["execution_s"])
        if trace["sut"] in ROLES["training"]:
            collected = collection_lookup[key]
            if collected["random_complete"]:
                adaptive = full_online(trace, initial_lookup[key]) + sum(
                    trace["execution_s"]
                )
            else:
                prefix = stopping_row(trace, initial_lookup[key], initial_gamma)
                adaptive = prefix["online_s"] + prefix["execution_s"]
            used = collected["actual_queries"]
        elif trace["sut"] in ROLES["calibration"]:
            # One physical trajectory, two count-head recalculations.
            adaptive = (
                full_online(trace, initial_lookup[key])
                + sum(trace["execution_s"])
                + predictor_seconds(updated_lookup[key])
            )
            used = BUDGET
        else:
            prefix = updated_rows[key]
            adaptive = prefix["online_s"] + prefix["execution_s"]
            used = prefix["queries"]
        cost_rows.append(
            {
                "pool": trace["pool"],
                "sut": trace["sut"],
                "fixed_s": fixed,
                "adaptive_s": adaptive,
                "queries": used,
                "exhaustive_s": trace["full_execution_s"],
                "found": trace["found"][used - 1],
                "full_found": trace["H200"],
                "recall": (
                    trace["found"][used - 1] / trace["positives"]
                    if trace["positives"]
                    else None
                ),
                "full_recall": trace["recall"],
                "positives": trace["positives"],
                "valid_recall": trace["recall"] is not None,
                "quality_failed": 20 * trace["found"][used - 1] < 19 * trace["H200"],
            }
        )
    collection_quality = {
        "discovery_retention": sum(r["found"] for r in cost_rows)
        / sum(r["full_found"] for r in cost_rows),
        "macro_recall": mean([r["recall"] for r in cost_rows]),
        "full_macro_recall": mean([r["full_recall"] for r in cost_rows]),
        "failed_tasks": [r for r in cost_rows if r["quality_failed"]],
        "pooled_recall": sum(r["found"] for r in cost_rows)
        / sum(t["positives"] for t in traces),
    }
    collection_quality["recall_retention"] = (
        collection_quality["macro_recall"] / collection_quality["full_macro_recall"]
    )
    collection_quality["passed"] = (
        collection_quality["discovery_retention"] >= 0.95
        and collection_quality["recall_retention"] >= 0.95
    )
    order = [
        (pool, sut)
        for role in ("training", "calibration", "evaluation")
        for sut in ROLES[role]
        for pool in (0, 1, 2)
    ]
    boot_lookup = {(r["pool"], r["sut"]): r for r in cost_rows}
    bootstrap = [boot_lookup[key] for key in order]
    steady_lookup = {}
    for trace in traces:
        key = trace["pool"], trace["sut"]
        row = dict(boot_lookup[key])
        forced = trace["sut"] in ROLES["calibration"] or (
            trace["sut"] in ROLES["training"]
            and collection_lookup[key]["random_complete"]
        )
        if forced:
            row["adaptive_s"] = full_online(trace, updated_lookup[key]) + sum(
                trace["execution_s"]
            )
            used = BUDGET
        else:
            prefix = updated_rows[key]
            row["adaptive_s"] = prefix["online_s"] + prefix["execution_s"]
            used = prefix["queries"]
        row.update(
            queries=used,
            found=trace["found"][used - 1],
            recall=(
                trace["found"][used - 1] / trace["positives"]
                if trace["positives"]
                else None
            ),
            quality_failed=20 * trace["found"][used - 1] < 19 * trace["H200"],
        )
        steady_lookup[key] = row
    continuing = [steady_lookup[key] for key in order]
    pool_prep = mean(list(preparations.values()))
    costs = []
    text.extend(["\n## 构建与维护费用\n", "| 阶段 | 独立构建耗时/秒 |", "|---|---:|"])
    for stage, seconds in builds.items():
        text.append(f"| {STAGE_NAMES[stage]} | {seconds:.2f} |")
    text.extend(
        [
            f"\n一次预测头维护与历史验证：{update_report['elapsed_s']:.2f} 秒；"
            f"选中更新步数：{update_report['selected_step']}。"
            "首个18任务批次采用确定性启动，后续批次使用已发布的校准规则。新增训练控制器一半随机完整执行，"
            "两个校准控制器完整执行，评价控制器自适应结束。"
            "任务顺序固定为训练、校准、后续评价，各控制器依次使用三个场景测试集的执行剖面。每18个任务计一次维护，末批保守整计；后续批次是既有工作负载的成本重建，不作为新增软件版本的效果证据。\n",
            "| 复用任务数 | 固定200次平均完整成本/秒 | 自适应含维护平均完整成本/秒 | 全量1024次/秒 | 相对固定节省 | 相对全量节省 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for reuse in REUSE:
        fixed = (
            build_fixed
            + pool_prep
            + campaign_total(bootstrap, continuing, reuse, "fixed_s")
        ) / reuse
        maintenance = math.ceil(reuse / len(traces)) * update_report["elapsed_s"]
        adaptive = (
            build_adaptive
            + pool_prep
            + maintenance
            + campaign_total(bootstrap, continuing, reuse, "adaptive_s")
        ) / reuse
        exhaustive = (
            campaign_total(bootstrap, continuing, reuse, "exhaustive_s") / reuse
        )
        row = {
            "reuse": reuse,
            "quality": campaign_quality(bootstrap, continuing, reuse),
            "fixed_s": fixed,
            "adaptive_s": adaptive,
            "exhaustive_s": exhaustive,
            "maintenance_batches": math.ceil(reuse / len(traces)),
            "saving_fixed": 1 - adaptive / fixed,
            "saving_exhaustive": 1 - adaptive / exhaustive,
        }
        costs.append(row)
        text.append(
            f"| {reuse} | {fixed:.2f} | {adaptive:.2f} | {exhaustive:.2f} | "
            f"{100*row['saving_fixed']:.2f}% | {100*row['saving_exhaustive']:.2f}% |"
        )
    text.append(
        f"\n成本情景中训练、校准、评价全部18个任务合计的发现数保留比例为 "
        f"{100*collection_quality['discovery_retention']:.2f}%，宏平均 Recall 保留比例为 "
        f"{100*collection_quality['recall_retention']:.2f}%。合并计数覆盖率为 {100*collection_quality['pooled_recall']:.2f}%。"
        "这与后续留出控制器上的停止质量分开统计。"
    )
    text.extend(
        [
            "\n### 启动与后续使用的任务费用剖面\n",
            "后续列使用发布模型在相同已有工作负载上的前缀，用于成本重建，不作为新增控制器的独立验证。",
            "| 测试集 | 控制器 | 启动查询数 / 发现数 | 后续查询数 / 发现数 | 完整发现数 | 后续保留95% |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for key in order:
        boot, steady = boot_lookup[key], steady_lookup[key]
        text.append(
            f"| {key[0]+1} | {SYSTEMS[key[1]]} | {boot['queries']} / {boot['found']} | {steady['queries']} / {steady['found']} | {boot['full_found']} | {'是' if not steady['quality_failed'] else '否'} |"
        )
    for failure in collection_quality["failed_tasks"]:
        text.append(
            f"训练资料采集阶段：场景测试集{failure['pool']+1}、"
            f"{SYSTEMS[failure['sut']]} 在{failure['queries']}次结束，"
            f"发现{failure['found']}/{failure['full_found']}；"
            "该历史阈值采集结果未达到95%保留目标，不能将其低费用解释为同质量节约。"
        )
    breakeven = next(
        (
            m
            for m in range(18, 10001, 18)
            if build_adaptive
            - build_fixed
            + math.ceil(m / 18) * update_report["elapsed_s"]
            + campaign_total(bootstrap, continuing, m, "adaptive_s")
            < campaign_total(bootstrap, continuing, m, "fixed_s")
            and campaign_quality(bootstrap, continuing, m)["passed"]
        ),
        None,
    )
    held = [
        r
        for r in rows
        if r["model"] == "updated"
        and r["mode"] == "empirical"
        and r["role"] == "evaluation"
    ]
    nll, prediction_errors, audit = {}, {}, []
    for name in ("initial", "updated", "direct_tilt"):
        values, errors, absolute, briers = [], [], [], []
        for prediction in forecasts[name]:
            if prediction["sut"] not in ROLES["evaluation"]:
                continue
            for row in prediction["forecast"]:
                truth = prediction["full_found"] - row["found"]
                values.append(-row["remaining_log_probability"][truth])
                errors.append(
                    ((row["remaining_mean"] - truth) / (BUDGET - row["query"])) ** 2
                )
                absolute.append(abs(row["remaining_mean"] - truth))
                briers.append(
                    (
                        row["score"]
                        - int(20 * row["found"] >= 19 * prediction["full_found"])
                    )
                    ** 2
                )
                if row["query"] in (10, 50, 100, 150, 190):
                    audit.append(
                        {
                            "model": name,
                            "sut": prediction["sut"],
                            "pool": prediction["pool"],
                            "query": row["query"],
                            "predicted_remaining": row["remaining_mean"],
                            "actual_remaining": truth,
                            "quality_score": row["score"],
                        }
                    )
        nll[name] = mean(values)
        prediction_errors[name] = {
            "normalized_mse": mean(errors),
            "count_mae": mean(absolute),
            "retention_brier": mean(briers),
        }
    write_json(folder / "prediction_audit.json", audit)
    text.extend(
        [
            "\n### 收益校正与更新的预测误差\n",
            "| 设置 | 完整剩余计数 NLL | 归一化均值误差 | 95%保留事件 Brier |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("direct_tilt", "initial", "updated"):
        errors = prediction_errors[name]
        text.append(
            f"| {MODEL_NAMES[name]} | {nll[name]:.4f} | {errors['normalized_mse']:.5f} | {errors['retention_brier']:.5f} |"
        )
    text.extend(
        [
            f"\n相对固定200次的费用交叉点：{breakeven if breakeven is not None else '当前构成下未回本'}个任务。"
            + (
                "全体任务质量未达标，因此不认定为等质量回本。"
                if not collection_quality["passed"]
                else "全体任务质量达到95%门槛。"
            )
            + "全量测试的 Recall 为100%；本方法的真实 Recall 见效果表，二者同时用于判断成本收益。\n",
            "## 验收与预测诊断\n",
            f"后续留出任务的剩余发现负对数似然：原头 {nll['initial']:.4f}，"
            f"更新头 {nll['updated']:.4f}。",
            f"归一化均值误差：原头 {prediction_errors['initial']['normalized_mse']:.5f}，"
            f"更新头 {prediction_errors['updated']['normalized_mse']:.5f}；"
            "固定查询位置的预测与实测续期数量见 prediction_audit.json。",
            "历史构建费用由保存的三种规则选例计时、串行先验准备及因果记录计时重建；用于核算的重复测量单列为研究开销，不重复加入部署成本。",
            "\n| 检查 | 结果 |",
            "|---|---|",
        ]
    )
    for name, result in acceptance.items():
        text.append(
            f"| {ACCEPTANCE_NAMES[name]} | {'达到95%门槛' if result['passed'] else '未达到95%门槛'} |"
        )
    main_cost = next(c for c in costs if c["reuse"] == 100)
    cost_pass = main_cost["saving_fixed"] > 0 and main_cost["quality"]["passed"]
    text.append(f"| 复用100次完整成本 | {'降低' if cost_pass else '未降低'} |")
    text.append(
        f"| 成本情景的全体任务质量 | {'达到95%门槛' if collection_quality['passed'] else '未达到95%门槛'} |"
    )
    text.append(
        "\n所有任务的经验、严格及轻量诊断记录保存在 stopping.json；"
        "训练和校准角色的记录不计入后续独立停止成绩。不计算 Area。"
    )
    text.extend(
        [
            "\n## 场景选择与发现前置\n",
            "| 规则 | H20 | H50 | H100 | H200 | 最早保留95%的测试次数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    frontloading = {}
    for label, source in (
        ("LRT（动作价值学习）", originals),
        ("完整预算前瞻", current),
        ("GCRT", {(t["pool"], t["sut"]): t for t in traces}),
    ):
        values = list(source.values())
        result = {
            f"H{k}": mean([t["found"][k - 1] for t in values])
            for k in (20, 50, 100, 200)
        }
        result["earliest95"] = mean(
            [
                next(
                    i
                    for i in range(10, 201)
                    if 20 * t["found"][i - 1] >= 19 * t["found"][-1]
                )
                for t in values
            ]
        )
        frontloading[label] = result
        text.append(
            f"| {label} | {result['H20']:.2f} | {result['H50']:.2f} | {result['H100']:.2f} | {result['H200']:.2f} | {result['earliest95']:.2f} |"
        )
    text.append(
        "\n场景选择规则在历史任务上确定，见 policy.json；后续预测网络只决定停止，不改变查询顺序。"
    )
    full_discovery_plot(folder, traces, originals, current)
    text.append("\n![六种控制器的完整发现曲线](full_discovery.png)\n")
    discovery_plot(folder, traces, originals, rows)
    text.append(
        "\n![留出控制器的危险发现曲线](discovery.png)\n\n停止后的发现数量保持不变。"
    )
    summary = {
        "family": family,
        "frontloading": frontloading,
        "selected_policy": read_json(folder / "policy.json")["selected"],
        "acceptance": acceptance,
        "summaries": summaries,
        "construction": builds,
        "historical_cost_components": read_json(folder / "historical_cost.json"),
        "research_rerun_s": (
            read_json(folder / "research_rerun.json")["elapsed_s"]
            if (folder / "research_rerun.json").exists()
            else 0.0
        ),
        "research_remeasurement_s": read_json(folder / "build_cost.json")["stages"][
            "cost_measurement"
        ]["elapsed_s"],
        "costs": costs,
        "cost_pass": cost_pass,
        "break_even_tasks": breakeven if collection_quality["passed"] else None,
        "financial_break_even_tasks": breakeven,
        "count_nll": nll,
        "prediction_errors": prediction_errors,
        "decision_changes": mechanism,
        "lifecycle_task_costs": cost_rows,
        "steady_task_costs": continuing,
        "campaign_order": order,
        "selected_update_step": update_report["selected_step"],
        "lifecycle_quality": collection_quality,
        "mean_final_recall": mean([r["recall"] for r in held]),
    }
    write_json(folder / "summary.json", summary)
    (folder / "comparison.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary


def main():
    summaries = [family_report(family) for family in ("cutin", "brake")]
    write_json(ROOT / "summary.json", summaries)
    lines = [
        "# 收益校正风险测试（GCRT）：实验结果\n",
        "车辆切入和前车紧急制动分别建模。36条完整轨迹检验场景选择，"
        "停止与维护效果在预定的留出控制器上评价；全量测试作为成本对照。"
        "发现数按控制器等权平均，表中覆盖率为宏平均 Recall；合并计数覆盖率在详细报告中另列。\n",
        "## 完整预算的发现能力：六种被测驾驶控制器\n",
        "| 功能场景 | 场景测试集 | LRT（动作价值学习）：H200 / Recall | 全预算前瞻：H200 / Recall | GCRT：H200 / Recall |",
        "|---|---|---:|---:|---:|",
    ]
    for summary in summaries:
        label = "车辆切入" if summary["family"] == "cutin" else "前车紧急制动"
        for split, split_name in SPLIT_NAMES.items():
            result = summary["acceptance"][split]
            lines.append(
                f"| {label} | {split_name} | "
                f"{result['original_H200']:.2f} / {100*result['original_recall']:.2f}% | "
                f"{result['current_H200']:.2f} / {100*result['current_recall']:.2f}% | "
                f"{result['H200']:.2f} / {100*result['recall']:.2f}% |"
            )
    lines.extend(
        [
            "\n## 提前停止：同一组留出控制器\n",
            "这里仅比较 IDM、IDM＋MOBIL，固定预算与停止结果使用完全相同的任务。"
            "这组控制器的覆盖率不能直接与上表六种控制器的平均数比较。\n",
            "| 功能场景 | 查询数：固定→停止 | 发现数：固定→停止 | 宏平均 Recall：固定→停止 | 合并发现数保留比例 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        label = "车辆切入" if summary["family"] == "cutin" else "前车紧急制动"
        base, result = (
            summary["summaries"]["fixed"],
            summary["summaries"]["updated_empirical"],
        )
        lines.append(
            f"| {label} | 200 → {result['queries']:.2f} | {base['found']:.2f} → {result['found']:.2f} | "
            f"{100*base['recall']:.2f}% → {100*result['recall']:.2f}% | "
            f"{100*result['discovery_count_retention']:.2f}% |"
        )
    lines.extend(
        [
            "\n## 复用100次的完整成本\n",
            "按预定训练、完整校准和后续评价顺序累计100个任务，启动只计一次，后续使用已发布的停止规则。每18个任务计一个维护批次，末批费用保守整计。覆盖率仍按控制器等权汇总。\n",
            "| 功能场景 | 相对固定200次节省 | 相对全量1024次节省 | 该成本情景的宏平均 / 合并计数 Recall | 全体任务95%保留验收 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for summary in summaries:
        label = "车辆切入" if summary["family"] == "cutin" else "前车紧急制动"
        cost = next(c for c in summary["costs"] if c["reuse"] == 100)
        quality = cost["quality"]
        lines.append(
            f"| {label} | {100*cost['saving_fixed']:.2f}% | {100*cost['saving_exhaustive']:.2f}% | "
            f"{100*quality['macro_recall']:.2f}% / {100*quality['pooled_recall']:.2f}% | {'通过' if quality['passed'] else '未通过'} |"
        )
    lines.extend(
        [
            "\n逐控制器结果、质量验收和全量测试成本见 [车辆切入](cutin/comparison.md)"
            "、[前车紧急制动](brake/comparison.md)。",
            "\n经验校准结果用于检验实际收益；严格校准使用独立控制器块计数，"
            "当前两个校准块不提供非平凡的95%概率早停阈值。",
        ]
    )
    (ROOT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
