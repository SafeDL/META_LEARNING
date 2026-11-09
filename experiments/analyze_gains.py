"""Read-only attribution of completed E2 gains to failure replacement and timing."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from .attribution import csv_report
from .attribution_report import SCENES, TITLES
from .failure_metrics import parameter_distances
from .joint_figures import STYLE
from .protocol import ROOT
from .schema import Scenario
from .storage import read_json, write_json

EXPLORATION = ROOT / "artifacts/runs/joint_response/method_exploration"
OUTPUT = EXPLORATION / "reports"
PREFIXES = (20, 50, 100, 200)


def failure_decomposition(original, candidate, budget):
    """Exact AUC difference: common-failure timing plus added minus lost failures.

    Inputs map a task-local candidate index to its one-based paid query position.
    A failure at query q contributes (B-q+1)/B to discovery AUC.
    """
    common = original.keys() & candidate.keys()
    added = candidate.keys() - original.keys()
    lost = original.keys() - candidate.keys()
    return {
        "original_failures": len(original),
        "candidate_failures": len(candidate),
        "common_failures": len(common),
        "added_failures": len(added),
        "lost_failures": len(lost),
        "delta_failures": len(added) - len(lost),
        "common_timing_auc": sum(original[i] - candidate[i] for i in common) / budget,
        "added_auc": sum(budget - candidate[i] + 1 for i in added) / budget,
        "lost_auc": -sum(budget - original[i] + 1 for i in lost) / budget,
    }


def oracle_limits(truth_count, initial_curve, budget):
    """Evaluator-only upper bounds after the actual shared initialization."""
    found = initial_curve[-1] if initial_curve else 0
    curve = [
        *initial_curve,
        *[
            found + min(step, truth_count - found)
            for step in range(1, budget - len(initial_curve) + 1)
        ],
    ]
    maximum = curve[-1]
    return {"oracle_failures": maximum, "oracle_auc": float(np.mean(curve))}


def positions(saved, candidates, truth):
    result = saved["result"]
    indices = [q["index"] for q in result["queries"]]
    if indices != result["selected_indices"] or len(set(indices)) != len(indices):
        raise ValueError("Gain analysis requires the original unique paid-query sequence")
    found = {}
    for number, query in enumerate(result["queries"], 1):
        index, feedback = query["index"], query["feedback"]
        if (
            feedback["scenario_id"] != candidates[index].scenario_id
            or feedback["collision"] != truth[index]
        ):
            raise ValueError("A paid response differs from its candidate truth")
        if feedback["collision"]:
            found[index] = number
    auc = sum(len(indices) - q + 1 for q in found.values()) / len(indices)
    if len(found) != saved["metrics"]["unique_collisions"] or not np.isclose(
        auc, saved["metrics"]["discovery_auc"], atol=1e-12, rtol=0
    ):
        raise ValueError("Saved discovery metrics differ from the paid query IDs")
    return found


def weight_evidence(result):
    """Score saved prequery experts on their actual common realized sequence."""
    prior = np.asarray(result["group_prior_weights"])
    expert_log_likelihood = np.zeros(len(prior))
    dynamic_loss = fixed_loss = group_loss = 0.0
    weight_error = probability_error = 0.0
    floors = 0
    for query in result["queries"]:
        probabilities = np.asarray(query["prequery_component_probabilities"])
        likelihood = probabilities if query["feedback"]["collision"] else 1 - probabilities
        scores = np.log(prior) + expert_log_likelihood
        weights = np.exp(scores - logsumexp(scores))
        final_weights = np.r_[0.9 * weights, 0.1]
        weight_error = max(
            weight_error, float(np.abs(final_weights - query["prequery_component_weights"]).max())
        )
        probability_error = max(
            probability_error,
            abs(float(final_weights @ probabilities) - query["prequery_probability"]),
        )
        dynamic_loss -= np.log(final_weights @ likelihood)
        fixed_loss -= np.log(0.9 * prior @ likelihood[:-1] + 0.1 * likelihood[-1])
        group_loss -= np.log(weights @ likelihood[:-1])
        floors += int((likelihood[:-1] < 1e-15).sum())
        expert_log_likelihood += np.log(np.maximum(likelihood[:-1], 1e-15))
    identity_error = abs(group_loss + logsumexp(np.log(prior) + expert_log_likelihood))
    if max(weight_error, probability_error) > 1e-10 or (not floors and identity_error > 1e-10):
        raise ValueError("Saved prequery probabilities or weights violate sequential updating")
    return {
        "sequence_dynamic_nll": float(dynamic_loss),
        "sequence_fixed_prior_nll": float(fixed_loss),
        "sequence_nll_change": float(dynamic_loss - fixed_loss),
        "weight_error": weight_error,
        "probability_error": probability_error,
        "likelihood_floor_activations": floors,
        "mixture_identity_error": float(identity_error),
    }


def task_evidence(task):
    original = read_json(ROOT / task["joint"])
    path = EXPLORATION / "e2/adaptive" / task["sut_id"] / f"{task['scenario']}.json.gz"
    candidate = read_json(path)
    if (
        candidate["specification"]["sut_id"] != task["sut_id"]
        or candidate["specification"]["scenario"] != task["scenario"]
    ):
        raise ValueError("The paired records belong to different registered tasks")
    scenes = [Scenario(**s) for s in original["specification"]["candidates"]]
    truth = original["evaluation"]["candidate_collisions"]
    old, new = positions(original, scenes, truth), positions(candidate, scenes, truth)
    initial = original["result"]["selected_indices"][:20]
    if len(original["result"]["queries"]) != 200 or len(candidate["result"]["queries"]) != 200:
        raise ValueError("Gain analysis requires the registered 200-query runs")
    if candidate["result"]["selected_indices"][:20] != initial:
        raise ValueError("The paired runs do not share initialization")
    identity = {
        "sut_id": task["sut_id"],
        "scenario": task["scenario"],
        "family": original["specification"]["bank_contract"]["object"]["family"],
    }
    row = {**identity, **failure_decomposition(old, new, 200)}
    row.update(weight_evidence(candidate["result"]))
    row["delta_auc"] = candidate["metrics"]["discovery_auc"] - original["metrics"]["discovery_auc"]
    if not np.isclose(
        row["delta_auc"],
        sum(row[k] for k in ("common_timing_auc", "added_auc", "lost_auc")),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Failure replacement and timing do not reconstruct discovery AUC")
    initial_curve = np.cumsum([int(truth[i]) for i in initial]).tolist()
    row.update(oracle_limits(sum(truth), initial_curve, 200))
    if max(row["original_failures"], row["candidate_failures"]) > row["oracle_failures"]:
        raise ValueError("A discovery count exceeds the fixed-initialization oracle limit")
    row.update(
        truth_failures=sum(truth),
        candidate_count=len(scenes),
        pool_failure_rate=sum(truth) / len(scenes),
        original_auc=original["metrics"]["discovery_auc"],
        candidate_auc=candidate["metrics"]["discovery_auc"],
    )
    for n in PREFIXES:
        row[f"delta_found_{n}"] = (
            candidate["metrics"]["discovery_curve"][n - 1]
            - original["metrics"]["discovery_curve"][n - 1]
        )
    added, lost = sorted(new.keys() - old.keys()), sorted(old.keys() - new.keys())
    row["added_nearest_original_distance"] = (
        float(
            parameter_distances([scenes[i] for i in added], [scenes[i] for i in old]).min(1).mean()
        )
        if added and old
        else None
    )
    regions = []
    for name in scenes[0].parameters:
        for cohort, indices in (
            ("original", list(old)),
            ("shared", sorted(old.keys() & new.keys())),
            ("added", added),
            ("lost", lost),
        ):
            values = [scenes[i].parameters[name] for i in indices]
            regions.append(
                {
                    **identity,
                    "parameter": name,
                    "cohort": cohort,
                    "count": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "q25": float(np.quantile(values, 0.25)) if values else None,
                    "median": float(np.median(values)) if values else None,
                    "q75": float(np.quantile(values, 0.75)) if values else None,
                }
            )
    references = {
        **identity,
        "original": task["joint"],
        "candidate": str(path.relative_to(ROOT)),
        "added_indices": added,
        "lost_indices": lost,
    }
    return row, regions, references


def aggregate(rows):
    counts = (
        "original_failures",
        "candidate_failures",
        "common_failures",
        "added_failures",
        "lost_failures",
        "delta_failures",
        "truth_failures",
        "oracle_failures",
        "candidate_count",
    )
    averages = (
        "original_auc",
        "candidate_auc",
        "delta_auc",
        "common_timing_auc",
        "added_auc",
        "lost_auc",
        "oracle_auc",
    )
    truth = sum(r["truth_failures"] for r in rows)
    return {
        "tasks": len(rows),
        **{k: sum(r[k] for r in rows) for k in counts},
        **{f"mean_{k}": float(np.mean([r[k] for r in rows])) for k in averages},
        "zero_failure_tasks": sum(r["truth_failures"] == 0 for r in rows),
        "original_at_count_ceiling": sum(
            r["truth_failures"] > 0 and r["original_failures"] == r["oracle_failures"] for r in rows
        ),
        "candidate_at_count_ceiling": sum(
            r["truth_failures"] > 0 and r["candidate_failures"] == r["oracle_failures"]
            for r in rows
        ),
        "pooled_original_coverage": sum(r["original_failures"] for r in rows) / truth
        if truth
        else None,
        "pooled_candidate_coverage": sum(r["candidate_failures"] for r in rows) / truth
        if truth
        else None,
    }


def common_prefix_evidence():
    with (EXPLORATION / "e2/reports/common_feedback.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for kind in SCENES:
        for count in PREFIXES:
            group = [r for r in rows if r["scenario"] == kind and int(r["paid_queries"]) == count]
            if len(group) != 8:
                raise ValueError("A common-prefix comparison is missing a registered ADS")
            result.append(
                {
                    "scenario": kind,
                    "paid_queries": count,
                    "workflow_top200_gain": sum(
                        int(r["joint_top_200_failures"]) - int(r["original_top_200_failures"])
                        for r in group
                    ),
                    "rss_conditional_top200_gain": sum(
                        int(r["joint_top_200_failures"])
                        - int(r["collision_conditioned_top_200_failures"])
                        for r in group
                    ),
                    "rss_conditional_mean_nll_change": float(
                        np.mean(
                            [
                                float(r["joint_nll"]) - float(r["collision_conditioned_nll"])
                                for r in group
                            ]
                        )
                    ),
                    "rss_conditional_mean_brier_change": float(
                        np.mean(
                            [
                                float(r["joint_brier"]) - float(r["collision_conditioned_brier"])
                                for r in group
                            ]
                        )
                    ),
                }
            )
    return result


@plt.rc_context(STYLE)
def figure(groups, prefixes, destination):
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2))
    names = ("CF", "Brake", "Cut-out", "LC")
    x = np.arange(4)
    axes[0].bar(
        x - 0.16,
        [groups[k]["added_failures"] for k in SCENES],
        0.32,
        color="#009E73",
        label="Newly found",
    )
    axes[0].bar(
        x + 0.16,
        [-groups[k]["lost_failures"] for k in SCENES],
        0.32,
        color="#D55E00",
        label="Not retained",
    )
    axes[0].plot(x, [groups[k]["delta_failures"] for k in SCENES], "kD", markersize=3, label="Net")
    axes[0].set(
        title="(a) Failure-set replacement", ylabel="Failure pairs", xticks=x, xticklabels=names
    )
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
    for offset, key, label, color in (
        (-0.24, "common_timing_auc", "Shared timing", "#777777"),
        (0, "added_auc", "Newly found", "#009E73"),
        (0.24, "lost_auc", "Not retained", "#D55E00"),
    ):
        axes[1].bar(
            x + offset, [groups[k][f"mean_{key}"] for k in SCENES], 0.24, label=label, color=color
        )
    axes[1].plot(x, [groups[k]["mean_delta_auc"] for k in SCENES], "kD", markersize=3, label="Net")
    axes[1].set(
        title="(b) Discovery AUC decomposition",
        ylabel="Mean change per task",
        xticks=x,
        xticklabels=names,
    )
    axes[1].legend(frameon=False, fontsize=6.5, ncol=2)
    for kind, label, color, marker in zip(
        SCENES, names, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), ("o", "s", "^", "D")
    ):
        y = [r["workflow_top200_gain"] for r in prefixes if r["scenario"] == kind]
        axes[2].plot(PREFIXES, y, marker=marker, color=color, markersize=3, label=label)
    axes[2].set(
        title="(c) Common-feedback ranking",
        xlabel="Paid feedback prefix",
        ylabel="Change in top-200 failures",
        xticks=PREFIXES,
    )
    axes[2].legend(frameon=False, fontsize=6.5, ncol=2)
    for axis in axes:
        axis.axhline(0, color="#888888", linewidth=0.7)
        axis.grid(axis="y", alpha=0.15)
        axis.margins(y=0.2)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(destination, dpi=600)
    plt.close(fig)


def analyze():
    tasks = read_json(EXPLORATION / "protocol.json")["tasks"]
    if len(tasks) != 32:
        raise ValueError("Gain analysis requires the original complete task matrix")
    rows, regions, references = [], [], []
    for task in tasks:
        row, parameters, reference = task_evidence(task)
        rows.append(row)
        regions.extend(parameters)
        references.append(reference)
    groups = {k: aggregate([r for r in rows if r["scenario"] == k]) for k in SCENES}
    families = {
        k: aggregate([r for r in rows if r["family"] == k])
        for k in sorted({r["family"] for r in rows})
    }
    systems = {
        k: aggregate([r for r in rows if r["sut_id"] == k])
        for k in sorted({r["sut_id"] for r in rows})
    }
    total, prefixes = aggregate(rows), common_prefix_evidence()
    with (EXPLORATION / "e2/reports/kernel_parameters.csv").open(encoding="utf-8-sig") as stream:
        kernels = list(csv.DictReader(stream))
    summary = {
        "status": "complete_read_only_analysis",
        "analysis_date": "2026-11-07",
        "scope": "old-platform development records",
        "new_logical_queries": 0,
        "new_physical_executions": 0,
        "model_updates": 0,
        "task_count": 32,
        "budget": 200,
        "initial_count": 20,
        "algorithm_seed": 11,
        "count_unit": "task-local ADS-candidate failure pairs; no cross-ADS deduplication",
        "total": total,
        "by_scenario": groups,
        "by_family": families,
        "by_system": systems,
        "common_prefix": prefixes,
        "prequery_weights": {
            **{
                k: sum(r[k] for r in rows)
                for k in (
                    "sequence_dynamic_nll",
                    "sequence_fixed_prior_nll",
                    "sequence_nll_change",
                    "likelihood_floor_activations",
                )
            },
            **{
                k: max(r[k] for r in rows)
                for k in ("weight_error", "probability_error", "mixture_identity_error")
            },
            "queries_scored": 200 * len(rows),
            "scope": "same realized adaptive sequence and saved component predictions; not a policy ablation",
        },
        "kernel_parameters": kernels,
        "diagnostic_inputs": [
            "e2/reports/common_feedback.csv",
            "e2/reports/kernel_parameters.csv",
            "e2/reports/paired_differences.csv",
        ],
        "references": references,
    }
    OUTPUT.mkdir(exist_ok=True)
    csv_report(OUTPUT / "gain_tasks.csv", rows)
    csv_report(OUTPUT / "gain_parameters.csv", regions)
    write_json(OUTPUT / "gain_analysis.json", summary)
    figure(groups, prefixes, OUTPUT / "gain_analysis.png")
    weights = summary["prequery_weights"]
    lines = [
        "# 共享与独立潜过程的收益分析",
        "",
        "读取已完成的32个旧平台任务，初始化20、预算200、种子11。新增付费查询、物理执行及模型更新均为0。"
        "本报告为设计资料上的事后诊断，保留全部任务，不改变算法配置或稿件。",
        "",
        "计数单位为各任务内的ADS—候选失效对；不同ADS在相同坐标的响应分别计数。",
        "",
        "| 场景 | 原方法 | 新流程 | 共同发现 | 新增发现 | 未保留 | 净变化 | 平均AUC变化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, title in zip(SCENES, TITLES):
        g = groups[kind]
        lines.append(
            f"| {title} | {g['original_failures']} | {g['candidate_failures']} | "
            f"{g['common_failures']} | {g['added_failures']} | {g['lost_failures']} | "
            f"{g['delta_failures']:+d} | {g['mean_delta_auc']:+.5f} |"
        )
    lines += [
        "",
        f"总净增{total['delta_failures']}来自新增{total['added_failures']}、未保留{total['lost_failures']}；"
        f"共同发现{total['common_failures']}。新集合不是原集合的超集。",
        "",
        "| 旧目标控制器家族 | 完整库失效率 | 原方法发现数 | 新流程发现数 | 净变化 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {name} | {g['truth_failures'] / g['candidate_count']:.2%} | "
            f"{g['original_failures']} | {g['candidate_failures']} | {g['delta_failures']:+d} |"
            for name, g in families.items()
        ],
        "",
        "这些家族按原执行凭据分类。库内失效率来自固定参数范围下的仿真真值，不能解释为自然道路事故率。",
        "规则、优化及搜索控制器合计净增81，两个旧学习控制器净增6；收益并非只来自旧PPO/DQN，"
        "但这些仍是旧平台开发对象，不能替代公开ADS验证。",
        f"完整候选库合计失效率为{total['truth_failures'] / total['candidate_count']:.2%}，"
        f"新流程已付费查询的失效率为{total['candidate_failures'] / (200 * len(rows)):.2%}；"
        "后者由风险选例主动富集，两者均不能作为自然道路事故率。旧学习控制器的完整库失效率"
        "高于其他家族，这仍是核验公开ADS、参数依据及执行语义的理由，不能通过降低采样难度掩盖。",
        "",
        "查询q首次发现的失效对AUC贡献为(B-q+1)/B。两种方法的AUC差可精确拆成共同发现的时间变化、"
        "新增发现的贡献和未保留发现的损失。32任务的平均差为：",
        "",
        f"{total['mean_common_timing_auc']:+.6f} + {total['mean_added_auc']:+.6f} "
        f"{total['mean_lost_auc']:+.6f} = {total['mean_delta_auc']:+.6f}。",
        "",
        "切出任务中共同发现的时间项为负，收益主要来自更换发现集合；换道任务中共同发现更早的时间项"
        "抵消了集合替换的AUC损失。数量和速度应分别解释。",
        "",
        f"全真值失效对数{total['truth_failures']}，真值加权覆盖率由"
        f"{total['pooled_original_coverage']:.2%}变为{total['pooled_candidate_coverage']:.2%}。"
        f"零失效任务{total['zero_failure_tasks']}个，覆盖率仍未定义。",
        "",
        "`gain_tasks.csv`还列出保留实际初始化后的评估端oracle上界。它仅用于解释预算及真值上限，"
        "不作为可访问未查询真值的测试基线。`gain_parameters.csv`给出任务内各集合的参数分位数，"
        "仅作事后区域描述；不将新增集合计数当作多样性。",
        f"固定实际初始化后的总发现上限为{total['oracle_failures']}，原流程与新流程各有"
        f"{total['original_at_count_ceiling']}个非零失效任务达到自身数量上限；达到上限的任务并不相同。",
        "",
        "共同Random前缀的排名诊断与自适应选例轨迹不同。前200名中的真实失效数衡量局部排名，"
        "NLL/Brier衡量概率预测，两者方向不必相同。固定新模型的RSS条件化在部分前缀改善排名，"
        "整体预测损失仍较差；不能据此推断RSS完全无信息，也不能声称RSS已经提高发现效率。",
        "",
        "| 场景 | 反馈数 | 新流程相对原流程Top200失效变化 | 同一新模型RSS条件化Top200变化 | RSS条件化平均NLL变化 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {r['scenario']} | {r['paid_queries']} | {r['workflow_top200_gain']:+d} | "
            f"{r['rss_conditional_top200_gain']:+d} | {r['rss_conditional_mean_nll_change']:+.6f} |"
            for r in prefixes
        ],
        "",
        "Top200变化为八ADS求和，预测损失变化为八ADS平均。去掉RSS条件化时固定的是已经联合拟合的状态，"
        "未去除RSS对历史训练及参数适配的影响，因此不构成从头去RSS的独立消融。",
        "",
        "切出目标核的离线先验共享载荷a=-0.02012、潜函数点相关系数约-0.01189，"
        "RSS共享潜过程方差占比a²约0.0405%；碰撞独立长度尺度约0.124，RSS独立长度尺度约0.680。"
        "弱共享及不同平滑尺度与减少辅助响应干扰的解释相符，但这些是模型潜函数参数，"
        "不是观测碰撞与RSS的相关系数或经验解释方差。结构、尺度及梯度分组尚未单独隔离。",
        "",
        "已知本次碰撞之后RSS的额外价值应面向未查询候选的碰撞判断。固定且正确的联合概率模型下，"
        "额外RSS带来的期望联合对数评分改进等于条件互信息I(Y_U;R_x|Y_x,D_t)。"
        "当前模型失配、RSS压缩和近似推断均可能使实际收益为负；已有排名快照不等于对该互信息的估计。",
        "",
        "逐查询权重只由此前碰撞预测似然更新。以原始参数分量之外的来源先验π_s为起点，"
        "其组内权重满足w_(t+1,s)∝π_s∏_(i≤t)P_(i,s)(y_i)。这给出标准专家混合的逐序列证据恒等式，"
        "不是本文新定理，也不保证另一选例策略下的失效数量或覆盖率。",
        f"复核{weights['queries_scored']}条实际查询，似然下限触发{weights['likelihood_floor_activations']}次，"
        f"最大权重误差{weights['weight_error']:.3g}，组内证据恒等式误差{weights['mixture_identity_error']:.3g}。",
        f"在完全相同的已实现序列和已存分量预测上，动态权重累计NLL为{weights['sequence_dynamic_nll']:.5f}，"
        f"固定初始权重为{weights['sequence_fixed_prior_nll']:.5f}，差为{weights['sequence_nll_change']:+.5f}。"
        "该评分诊断不重跑选例，不等于新结构的固定权重策略对照，也不继承旧结构消融的因果结论。",
        "",
        "现有同发现数距离结果仍以`e2/reports/paired_differences.csv`为准，未显示一致多样性提升。"
        "本次分解解释发现集合与顺序如何变化，不单独识别潜过程结构、尺度和梯度分组各自的因果作用。",
        "",
        "方法定位与最近邻文献的核对见[文献记录](../literature_review.md)。"
        "异质多输出GP及少样本测试已有先例；本文贡献须由跨ADS来源后验、目标适配及有限预算发现证据支撑。",
        "",
    ]
    (OUTPUT / "gain_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    report = analyze()
    print(
        {
            k: report["total"][k]
            for k in (
                "common_failures",
                "added_failures",
                "lost_failures",
                "delta_failures",
                "mean_delta_auc",
            )
        }
    )
