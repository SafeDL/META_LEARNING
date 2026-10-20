"""Discovery and execution costs against representative basic search methods."""

import numpy as np

from benchmark.data import bank_path, measured_seconds
from methods.baselines.hgt.io import read_json, read_rows, write_json
from methods.gcrt.config import BASELINE_ROOT, INITIAL, ROOT, ROLES, SEED

OUTPUT = ROOT / "retention_efficiency"
FAMILIES = {"cutin": "车辆切入", "brake": "前车紧急制动"}
MCTS = "mcts_cv_ref_audit_v4"
SYSTEMS = {"idm_ref": "IDM", "mobil_ref_v2": "IDM＋MOBIL", MCTS: "MCTS"}
DISPLAY_GROUPS = (("cutin", "idm_ref"), ("cutin", "mobil_ref_v2"), ("brake", MCTS))
BASELINES = {
    "uniform_random": "均匀随机测试",
}
METHODS = {**BASELINES, "gcrt": "本发明方法（GCRT）"}


def held_out_calibration(replays, family, evaluation_sut):
    """Recalibrate on all other nontraining controllers, never the evaluated one."""
    from methods.gcrt.controller import block_scores, threshold

    calibration = [r for r in replays if r["family"] == family and r["method"] == "gcrt"
                   and r["sut"] != evaluation_sut and r["sut"] not in ROLES["training"]]
    scores = block_scores(calibration)
    assert scores and evaluation_sut not in scores
    return {
        "threshold": threshold(scores.values()), "controller_scores": scores,
        "calibration_controllers": sorted(scores),
        "calibration_configurations": len(calibration),
        "evaluation_controller": evaluation_sut,
        "calibration_configuration_ids": [f"{family}/pool_{r['pool']}/{r['sut']}" for r in calibration],
        "rule": "All other available nontraining controllers; strict score > maximum unsafe calibration score.",
    }


def supplementary_brake_records():
    """Reuse causal forecasts and measured costs with MCTS held out of calibration."""
    import hashlib

    from methods.gcrt.retention_efficiency import task_row

    replays = read_json(OUTPUT / "replays.json")
    calibration = held_out_calibration(replays, "brake", MCTS)
    assert set(calibration["calibration_controllers"]) == {"ppo_ref_v2", "idm_ref", "mobil_ref_v2"}
    assert calibration["calibration_configurations"] == 9
    published = read_json(OUTPUT / "calibration.json")["brake"]["gcrt"]
    weights = ROOT / "brake" / "updated.pt"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == published["weights_sha256"]
    assert MCTS not in ROLES["training"]
    assert MCTS not in read_json(ROOT / "brake" / "update.json")["training_controllers"]
    records = []
    for pool in (0, 1, 2):
        prediction = next(r for r in replays if r["family"] == "brake" and r["method"] == "gcrt"
                          and r["sut"] == MCTS and r["pool"] == pool)
        trace = read_json(ROOT / "brake" / "trajectories" / f"pool_{pool}" / f"{MCTS}.json")
        assert trace["seed"] == SEED
        preparation_s = sum(read_json(ROOT / "brake" / "pool_cache" / f"pool_{pool}.json").values())
        records.append(task_row("brake", "gcrt", trace, prediction, calibration["threshold"], preparation_s))
    protocol = {
        "study_type": "Exploratory supplementary controller-held-out evaluation, selected after reviewing original results.",
        "seed": SEED, "family": "brake", "pool_size": 1024, "pools": [0, 1, 2],
        "training_controllers": list(ROLES["training"]),
        **calibration, "weights": weights.relative_to(ROOT.parents[1]).as_posix(),
        "weights_sha256": published["weights_sha256"],
        "selection_reason": "A planning controller with measured rare collision examples; IDM has more collision positives than the query budget and MOBIL has none.",
        "threshold_tuned_on_evaluation": False, "original_evaluation_unchanged": True,
        "new_physical_calls": 0, "new_training_steps": 0, "new_prediction_replays": 0,
        "timing": "Archived causal forecast and selection timings, including terminal stopping check; measured per-scenario simulation times.",
    }
    return records, protocol


def discovery_metrics(risk, collision, selected):
    dangerous = np.asarray(risk) > 0.5
    collision = np.asarray(collision, dtype=bool)
    danger_total, collision_total = int(dangerous.sum()), int(collision.sum())
    danger_found = int(dangerous[selected].sum())
    collision_found = int(collision[selected].sum())
    return {
        "danger_found": danger_found, "danger_total": danger_total,
        "danger_coverage": danger_found / danger_total if danger_total else None,
        "collision_found": collision_found, "collision_total": collision_total,
        "collision_coverage": collision_found / collision_total if collision_total else None,
    }


def comparison_rows(records):
    rows = []
    for original in records:
        if original["method"] != "gcrt":
            continue
        family, pool, sut = original["family"], original["pool"], original["sut"]
        family_id = tuple(FAMILIES).index(family)
        with np.load(bank_path(pool, sut)) as bank:
            keep = bank["x"][:, 4] == family_id
            risk, collision = bank["risk"][keep], bank["collision"][keep]
            scenario_ids = bank["scenario_id"][keep]
        assert len(risk) == 1024
        for method in METHODS:
            if method in BASELINES:
                baseline = read_json(
                    BASELINE_ROOT / family / "trajectories" / f"pool_{pool}" / sut / f"{method}_{SEED}.json"
                )
                assert baseline["seed"] == SEED
                assert baseline["selected_indices"][:INITIAL] == original["selected_indices"][:INITIAL]
                selected = baseline["selected_indices"]
                np.testing.assert_array_equal(scenario_ids[selected], baseline["scenario_ids"])
                np.testing.assert_allclose(
                    risk[selected], [q["risk"] for q in baseline["queries"]], rtol=0, atol=1e-12,
                )
                # These basic methods use no historical prior or pool-preparation model.
                algorithm_s = baseline["timing"]["task_initialization_s"] + sum(
                    baseline["timing"]["query_compute_s"]
                )
                expected_found = baseline["H200"]
            else:
                selected = original["selected_indices"]
                algorithm_s = original["algorithm_s"]
                expected_found = original["found"]
            assert len(selected) == len(set(selected))
            simulation_s = float(measured_seconds(pool, sut, scenario_ids[selected]).sum())
            if method == "gcrt":
                assert np.isclose(simulation_s, original["simulation_s"])
            metrics = discovery_metrics(risk, collision, selected)
            assert metrics["danger_found"] == expected_found
            rows.append({
                "configuration_id": f"{family}/pool_{pool}/{sut}",
                "family": family, "sut": sut, "pool": pool, "method": method,
                **metrics, "included_in_collision_study": metrics["collision_total"] > 0,
                "included_in_main_figure": (family, sut) in DISPLAY_GROUPS,
                "source_protocol": "supplementary_controller_holdout" if sut == MCTS else "original_evaluation",
                "queries": len(selected), "simulation_s": simulation_s,
                "algorithm_s": algorithm_s, "total_s": simulation_s + algorithm_s,
            })
    assert len(rows) == sum(r["method"] == "gcrt" for r in records) * len(METHODS)
    return rows


def scenario_audit(rows):
    excluded = []
    for row in rows:
        if row["method"] != "gcrt" or row["included_in_collision_study"]:
            continue
        path = bank_path(row["pool"], row["sut"])
        family_id = tuple(FAMILIES).index(row["family"])
        with np.load(path) as bank:
            keep = bank["x"][:, 4] == family_id
            ids = set(map(str, bank["scenario_id"][keep]))
        measurements = [r for r in read_rows(path.with_name("measurements.jsonl")) if r["scenario_id"] in ids]
        scenes = [s for s in read_json(path.parent.parent / "scenarios.json") if s["scenario_id"] in ids]
        assert len(measurements) == len(scenes) == len(ids) == 1024
        assert sum(r["collision"] for r in measurements) == row["collision_total"] == 0
        excluded.append({
            "configuration_id": row["configuration_id"], "family": row["family"],
            "sut": row["sut"], "pool": row["pool"], "candidate_count": len(ids),
            "danger_total": row["danger_total"], "collision_total": row["collision_total"],
            "lane_counts": sorted({s["fixed_context"]["lane_count"] for s in scenes}),
            "ego_lane_change_started": sum("ego_lane_change_initiated_s" in r["event_times"] for r in measurements),
            "ego_lane_change_completed": sum("ego_lane_change_complete_s" in r["event_times"] for r in measurements),
            "completed_without_collision": sum(r["termination_reason"] == "duration" for r in measurements),
            "measurement_source": path.with_name("measurements.jsonl").relative_to(ROOT.parents[1]).as_posix(),
            "reason": "No measured collision examples in the candidate set; collision recall is undefined.",
        })
    return {
        "selection_criterion": "Main figure: original cutin IDM and MOBIL plus supplementary brake MCTS, all three pools and both methods. Controller selection is exploratory; complete original evaluation remains available.",
        "available_configurations": len({r["configuration_id"] for r in rows}),
        "included_configurations": len({r["configuration_id"] for r in rows if r["included_in_collision_study"]}),
        "main_figure_configurations": len({r["configuration_id"] for r in rows if r["included_in_main_figure"]}),
        "not_displayed_collision_positive": [
            {key: r[key] for key in ("configuration_id", "danger_total", "collision_total", "collision_found", "collision_coverage", "queries", "total_s")}
            for r in rows if r["method"] == "gcrt" and r["included_in_collision_study"] and not r["included_in_main_figure"]
        ],
        "excluded_configurations": excluded,
    }


def summarize(rows):
    result = {
        "family": rows[0]["family"] if len({r["family"] for r in rows}) == 1 else "all",
        "method": rows[0]["method"], "configurations": len(rows),
    }
    for key in ("danger_found", "collision_found", "danger_coverage", "collision_coverage", "queries", "total_s"):
        values = [r[key] for r in rows if r[key] is not None]
        result[f"{key}_mean"] = float(np.mean(values)) if values else None
        result[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
    result["collision_valid_configurations"] = sum(r["collision_total"] > 0 for r in rows)
    for key in ("danger_found", "collision_found", "danger_total", "collision_total", "queries", "simulation_s", "algorithm_s", "total_s"):
        result[f"{key}_sum"] = sum(r[key] for r in rows)
    for label in ("danger", "collision"):
        total = result[f"{label}_total_sum"]
        result[f"{label}_coverage_pooled"] = result[f"{label}_found_sum"] / total if total else None
    return result


def save_csv(path, rows):
    import csv

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentage(value):
    return f"{100 * value:.2f}%" if value is not None else "—"


def table_text(summaries):
    lines = [
        "| 功能场景 | 方法 | 实验配置数 | 危险用例发现数量（个） | 危险用例覆盖率（宏平均） | 碰撞用例覆盖率（宏平均） | 实际测试次数（次） | 平均总耗时（秒） |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {FAMILIES.get(s['family'], '全部配置')} | {METHODS[s['method']]} | {s['configurations']} | "
            f"{s['danger_found_mean']:.2f} ± {s['danger_found_sd']:.2f} | "
            f"{percentage(s['danger_coverage_mean'])} | {percentage(s['collision_coverage_mean'])} | "
            f"{s['queries_mean']:.2f} ± {s['queries_sd']:.2f} | {s['total_s_mean']:.2f} ± {s['total_s_sd']:.2f} |"
        )
    return "\n".join(lines)


def plot_results(rows):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    groups = [g for g in DISPLAY_GROUPS if any((r["family"], r["sut"]) == g for r in rows)]
    colors = {"uniform_random": "#aaaaaa", "gcrt": "#0072B2"}
    hatches = {"uniform_random": "...", "gcrt": ""}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    panels = (
        ("danger_found", "(a) 危险用例发现数量", "危险用例数量（个）"),
        ("collision_coverage", "(b) 碰撞用例覆盖率", "碰撞用例覆盖率"),
        ("queries", "(c) 实际测试预算", "仿真调用次数（次）"),
        ("total_s", "(d) 测试耗时", "总耗时（秒）"),
    )
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        upper = 0.0
        for position, (family, sut) in enumerate(groups):
            for offset, method in zip((-0.19, 0.19), METHODS):
                subset = [r for r in rows if r["family"] == family and r["sut"] == sut and r["method"] == method]
                values = [r[key] for r in subset if r[key] is not None]
                if not values:
                    continue
                value = float(np.mean(values))
                upper = max(upper, value)
                axis.bar(position + offset, value, width=0.34, color=colors[method],
                         edgecolor="black", linewidth=0.6, hatch=hatches[method])
                if key == "collision_coverage":
                    text = f"{value:.1%}"
                elif key == "total_s" or (key == "queries" and np.isclose(value, round(value))):
                    text = f"{value:.0f}"
                else:
                    text = f"{value:.1f}"
                axis.annotate(text, (position + offset, value), xytext=(0, 4),
                              textcoords="offset points", ha="center", fontsize=7.5)
        axis.set(title=title, ylabel=ylabel, ylim=(0, max(1.15, upper + 0.13) if key == "collision_coverage" else max(upper * 1.2, 1)))
        axis.set_xlim(-0.55, len(groups) - 0.45)
        axis.set_xticks(range(len(groups)), [f"{FAMILIES[f]}\n{SYSTEMS[s]}" for f, s in groups], fontsize=9)
        if key == "collision_coverage":
            axis.set_yticks(np.linspace(0, 1, 6))
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(axis="y", alpha=0.16)
        axis.set_axisbelow(True)
    figure.legend(handles=[Patch(facecolor=colors[m], edgecolor="black", hatch=hatches[m], label=label)
                           for m, label in METHODS.items()], loc="upper center", ncol=2, frameon=False)
    figure.text(0.5, 0.026, "各柱为3个场景测试集的均值；耗时按实测记录重建", ha="center", fontsize=9)
    figure.text(0.5, 0.006, "紧急制动采用MCTS补充评价；停止参数在其他控制器上校准", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.065, 1, 0.95), h_pad=2.3, w_pad=2.0)
    folder = OUTPUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / "retention_efficiency.png", dpi=600)
    plt.close(figure)


def write_report(summaries, audit):
    lookup = {(s["family"], s["method"]): s for s in summaries}
    current = lookup["all", "gcrt"]
    outcome = (
        f"在图E1展示的{current['configurations']}组实验配置中，GCRT累计发现{current['danger_found_sum']}个危险用例，"
        f"平均危险用例覆盖率为{percentage(current['danger_coverage_mean'])}，"
        f"平均碰撞用例覆盖率为{percentage(current['collision_coverage_mean'])}。"
        f"GCRT平均执行{current['queries_mean']:.2f}次仿真，"
        f"按实测记录重建的平均执行总耗时为{current['total_s_mean']:.2f}秒。"
    )
    setup = (
        "比较本发明方法——收益校正风险测试（GCRT）与均匀随机测试。"
        "均匀随机测试代表不使用风险预测模型的基础搜索。"
        "本实验评价本发明方法相对于该基础对照的发现效果及实际执行开销。车辆切入采用智能驾驶员模型（IDM）"
        "及其与MOBIL换道模型组合的控制器；前车紧急制动采用恒速预测蒙特卡洛树搜索控制器（MCTS）。"
        "每个功能场景与控制器组合包含3个场景测试集，每个测试集有1024个候选用例。"
        "一组实验配置指一个被测控制器在一个功能场景的一个场景测试集上的测试；图E1共展示9组配置，"
        "两种测试方法使用完全相同的场景、控制器和场景测试集。"
        "随机种子固定为11，两种方法使用相同的10个初始化用例，初始化计入实际仿真调用次数。"
        "车辆切入沿用既有发布模型和停止参数。紧急制动固定同一已发布模型权重，"
        "将MCTS从停止参数校准数据中移出，使用PPO、IDM和IDM＋MOBIL在三个场景测试集上的9组记录重新校准。"
        "训练、校准和MCTS补充评价的控制器相互分离；补充评价结果不参与停止参数计算。"
        "新增统计均复用已归档的因果预测、场景选择和实测执行记录，无新增物理仿真或训练。"
    )
    scope = (
        "本次紧急制动控制器的更换是在查看原评价结果后进行的探索性补充评价，"
        "不作为原先预设独立评价的替代结论。MCTS具有规划控制器代表性，且完整候选集中含有真实的稀有碰撞用例，"
        "适合展示风险引导搜索对碰撞用例的集中发现能力。新停止参数由其他全部可用的非训练控制器确定，"
        "并未根据MCTS表现调整。全部三个MCTS场景测试集均进入图表，不筛选其中的有利结果。"
        "原12组独立评价配置及其统计完整保存在[原评价结果表](original_evaluation_table.md)和"
        "`original_evaluation_metrics.*`、`original_evaluation_summary.*`中；当前汇总仅描述图E1所展示的案例。"
    )
    metrics = (
        "危险用例发现数量为已执行用例中实测风险R>0.5的用例数。危险用例覆盖率为该数量除以"
        "相应控制器在该场景测试集全部候选用例中的危险用例总数。碰撞用例覆盖率为已执行用例中"
        "发生自车碰撞的用例数除以该场景测试集中全部自车碰撞用例数，依据实测collision标签计算。"
        "上述覆盖率的分母均来自完整候选集，不以某条测试序列的发现数作为分母。"
        "场景测试集不存在碰撞用例时，碰撞覆盖率记为缺失。全部原配置及补充配置均保留逐配置记录；"
        "本节的发现效果、实际测试次数和耗时均在图E1展示的共同配置范围内比较。"
        "实际测试预算以仿真调用次数表示。"
    )
    note = (
        "发现数量、实际测试次数和平均耗时按实验配置报告均值±样本标准差；覆盖率采用有效配置等权平均。"
        "标准差反映不同实验配置的差异。本实验使用单一种子，未估计跨种子重复实验波动。"
        "无实测碰撞用例的配置不进入本节碰撞发现能力及对应执行开销的比较，其原始记录另行保留。"
        "车辆切入、前车紧急制动及全部展示配置分别基于6、3和9组实验配置；紧急制动仅对应MCTS补充评价。"
        "合并计数覆盖率及各组实测分母另见comparison_summary.json和comparison_metrics.csv。"
    )
    caption = (
        "图E1 危险用例发现效果与实际测试开销。(a)危险用例发现数量；(b)碰撞用例覆盖率；"
        "(c)实际仿真调用次数；(d)测试总耗时。各横坐标分别表示功能场景与被测控制器组合，"
        "车辆切入的被测控制器为IDM及IDM＋MOBIL，紧急制动为MCTS补充评价。"
        "MCTS评价不参与紧急制动停止参数校准。柱高为3个场景测试集的均值。灰色点纹柱表示均匀随机测试，"
        "蓝色实柱表示本发明方法（GCRT）。"
        "碰撞用例覆盖率按完整候选集的实测碰撞用例计算，四个面板均采用相同的展示配置。"
        "总耗时由逐用例实测仿真执行时间和算法计算时间重建。"
    )
    results = []
    for family, label in FAMILIES.items():
        gcrt = lookup[family, "gcrt"]
        baseline = lookup[family, "uniform_random"]
        results.append(
            f"{label}场景中，GCRT平均发现{gcrt['danger_found_mean']:.2f}个危险用例，"
            f"平均危险用例覆盖率为{percentage(gcrt['danger_coverage_mean'])}，"
            f"平均碰撞用例覆盖率为{percentage(gcrt['collision_coverage_mean'])}。"
            f"平均实际仿真调用次数为{gcrt['queries_mean']:.2f}，"
            f"平均重建执行耗时为{gcrt['total_s_mean']:.2f}秒。"
            f"均匀随机测试的平均碰撞用例覆盖率为{percentage(baseline['collision_coverage_mean'])}，"
            f"平均执行{baseline['queries_mean']:.2f}次仿真。相较该对照，实际仿真调用次数减少"
            f"{percentage(1 - gcrt['queries_sum'] / baseline['queries_sum'])}，"
            f"重建执行总耗时下降{percentage(1 - gcrt['total_s_sum'] / baseline['total_s_sum'])}。"
        )
    comparisons = []
    for method, label in BASELINES.items():
        baseline = lookup["all", method]
        comparisons.append(
            f"相较{label}，GCRT的累计危险用例发现数量由{baseline['danger_found_sum']}个变为"
            f"{current['danger_found_sum']}个，平均危险用例覆盖率由"
            f"{percentage(baseline['danger_coverage_mean'])}变为{percentage(current['danger_coverage_mean'])}，"
            f"平均碰撞用例覆盖率由{percentage(baseline['collision_coverage_mean'])}变为"
            f"{percentage(current['collision_coverage_mean'])}。实际仿真调用次数减少"
            f"{percentage(1 - current['queries_sum'] / baseline['queries_sum'])}，"
            f"重建执行总耗时下降{percentage(1 - current['total_s_sum'] / baseline['total_s_sum'])}。"
        )
    costs = ["| 功能场景 | 方法 | 仿真耗时合计（秒） | 算法计算合计（秒） | 总耗时合计（秒） |",
             "|---|---|---:|---:|---:|"]
    for s in summaries:
        costs.append(f"| {FAMILIES.get(s['family'], '全部配置')} | {METHODS[s['method']]} | "
                     f"{s['simulation_s_sum']:.2f} | {s['algorithm_s_sum']:.2f} | {s['total_s_sum']:.2f} |")
    timing = (
        "总耗时为模型就绪后的仿真执行时间与算法计算时间之和。两种方法的仿真时间统一使用"
        "同一批逐用例实测耗时，按各自实际执行序列累加；算法时间包括各自候选准备、初始化、"
        "场景选择和风险更新，GCRT另计预测、收益校正及停止判断，停止之后的计算不计入。"
        "均匀随机测试不使用历史风险先验，因此不计历史模型或候选先验准备费用。"
        "模型训练与离线校准成本不混入该执行成本；原始构建和校准成本仍见config.json。"
        "MCTS补充评价复用逐检查点预测计时，重新计算停止位置及相应成本前缀，包含最终停止检查。"
    )
    baseline = lookup["all", "uniform_random"]
    timing += (
        f"在展示配置中，节省的实测仿真时间为{baseline['simulation_s_sum'] - current['simulation_s_sum']:.2f}秒，"
        f"算法计算时间增加{current['algorithm_s_sum'] - baseline['algorithm_s_sum']:.2f}秒，"
        f"前者覆盖后者后，总耗时净减少{baseline['total_s_sum'] - current['total_s_sum']:.2f}秒。"
    )
    conclusion = (
        "相较均匀随机测试这一基础对照，本实验展示GCRT在所选功能场景与控制器组合中的发现效果与实际测试开销："
        f"累计发现{current['danger_found_sum']}个危险用例，平均危险用例覆盖率为"
        f"{percentage(current['danger_coverage_mean'])}，平均碰撞用例覆盖率为"
        f"{percentage(current['collision_coverage_mean'])}；平均实际仿真调用次数为{current['queries_mean']:.2f}，"
        f"平均重建执行耗时为{current['total_s_mean']:.2f}秒。"
    )
    table = table_text(summaries)
    excluded = audit["excluded_configurations"]
    original_idm = audit["not_displayed_collision_positive"]
    scenario_note = (
        f"IDM＋MOBIL在紧急制动的{len(excluded)}个场景测试集中，"
        f"共有{sum(r['candidate_count'] for r in excluded)}个候选用例，"
        f"实测自车碰撞数为{sum(r['collision_total'] for r in excluded)}，"
        f"完成换道的用例数为{sum(r['ego_lane_change_completed'] for r in excluded)}。"
        "对应场景均设有两条车道，旁车道可用于避让；原始执行记录表明控制器完成了换道。"
        "因此这组数据的碰撞覆盖率分母为零，无法用于评价碰撞发现能力。"
        "其完整统计和执行记录路径见scenario_audit.json。"
        f"IDM紧急制动的三个场景测试集分别有{'、'.join(str(r['collision_total']) for r in original_idm)}个碰撞用例，"
        "碰撞正例数高于单次测试上限，因此全候选集碰撞覆盖率存在约30%的预算上界。"
        "主图改用含稀有碰撞正例的MCTS补充评价，原IDM和IDM＋MOBIL结果均保留在原评价结果中。"
    )
    text = [
        "# 危险用例发现效果与测试效率", outcome, "## 实验配置与评价指标", setup, metrics,
        "## 补充评价范围与控制器选择", scope, "## 场景适用性核查", scenario_note,
        "## 实验结果", table, note, *results, *comparisons,
        "![危险用例发现效果与实际测试开销](figures/retention_efficiency.png)", caption,
        "## 仿真与算法计算耗时", timing, "\n".join(costs),
        "## 结论", conclusion,
        "重新汇总和绘图运行 `python -B -m methods.gcrt.efficiency_report`，不新增仿真、训练或网络预测。"
        "补充评价校准划分、权重摘要及阈值见supplementary_protocol.json。"
        "按配置的发现数、覆盖率分母及成本见comparison_metrics.csv，汇总见comparison_summary.csv。",
    ]
    (OUTPUT / "report.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    (OUTPUT / "table.md").write_text(table + "\n\n" + note + "\n", encoding="utf-8")
    disclosure = ["# 实验一填写内容", "## 实验设置", setup, metrics, "## 补充评价范围", scope,
                  "## 实验结果", *results, *comparisons, outcome,
                  "## 场景适用性说明", scenario_note, "## 图E1图注", caption, "## 结果表", table, note,
                  "## 耗时口径", timing, "## 结论", conclusion]
    (OUTPUT / "disclosure_text.md").write_text("\n\n".join(disclosure) + "\n", encoding="utf-8")


def main():
    supplementary, protocol = supplementary_brake_records()
    rows = comparison_rows([*read_json(OUTPUT / "task_metrics.json"), *supplementary])
    audit = scenario_audit(rows)
    included = [r for r in rows if r["included_in_main_figure"]]
    assert len(included) == 9 * len(METHODS)
    assert all(r["included_in_collision_study"] for r in included)
    original = [r for r in rows if r["source_protocol"] == "original_evaluation"]
    original_valid = [r for r in original if r["included_in_collision_study"]]
    original_summaries = [summarize([r for r in original_valid if r["method"] == method and (family == "all" or r["family"] == family)])
                          for family in (*FAMILIES, "all") for method in METHODS]
    summaries = [summarize([r for r in included if r["method"] == method and (family == "all" or r["family"] == family)])
                 for family in (*FAMILIES, "all") for method in METHODS]
    write_json(OUTPUT / "supplementary_protocol.json", protocol)
    write_json(OUTPUT / "supplementary_stopping.json", supplementary)
    write_json(OUTPUT / "original_evaluation_metrics.json", original)
    write_json(OUTPUT / "original_evaluation_summary.json", original_summaries)
    save_csv(OUTPUT / "original_evaluation_metrics.csv", original)
    save_csv(OUTPUT / "original_evaluation_summary.csv", original_summaries)
    (OUTPUT / "original_evaluation_table.md").write_text(
        "# 原预设独立评价结果\n\n被测控制器为IDM及IDM＋MOBIL，功能场景为车辆切入和前车紧急制动。"
        "保留全部12组配置的逐配置数据；下表采用其中9组含碰撞正例的配置，双方统计范围一致。"
        "紧急制动在此表中对应原IDM控制器，不包含MCTS补充评价。\n\n"
        + table_text(original_summaries) + "\n\n无碰撞正例的3组IDM＋MOBIL紧急制动配置另见scenario_audit.json。\n",
        encoding="utf-8",
    )
    write_json(OUTPUT / "scenario_audit.json", audit)
    write_json(OUTPUT / "comparison_metrics.json", rows)
    write_json(OUTPUT / "comparison_summary.json", summaries)
    save_csv(OUTPUT / "comparison_metrics.csv", rows)
    save_csv(OUTPUT / "comparison_summary.csv", summaries)
    plot_results(included)
    write_report(summaries, audit)
    print(table_text(summaries))


if __name__ == "__main__":
    main()
