"""Effectiveness, matched discovery costs and amortization by scenario family."""

import math

import numpy as np

from methods.baselines.hgt.io import read_json, write_json

from .by_family import BUDGET, INITIAL, METHODS, REUSE, ROOT, SUT_IDS, TARGET_FRACTIONS

NAMES = {
    "lrt": "局部风险迁移 LRT",
    "cg_brt": "一致性门控贝叶斯风险迁移 CG-BRT",
    "dart": "决策相关风险迁移 DART",
    "hgt": "历史引导高斯过程风险测试 HGT",
    "gp_ei": "高斯过程期望改进 GP-EI",
    "rf_bo": "随机森林贝叶斯优化 RF-BO",
    "uniform_random": "均匀随机测试",
    "no_value_learning": "LRT取消动作价值学习",
}
SYSTEMS = dict(zip(SUT_IDS, ("IDM", "FVDM", "MOBIL", "VI", "MCTS", "PPO")))


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def hit_query(trace, fraction):
    if trace["positives"] == 0:
        return None
    target = math.ceil(fraction * min(trace["positives"], BUDGET))
    return next(
        (
            i + 1
            for i, count in enumerate(trace["found"])
            if i + 1 >= INITIAL and count >= target
        ),
        None,
    )


def construction_cost(ledger, method):
    if method in ("uniform_random", "gp_ei", "rf_bo"):
        return 0.0
    stages = ["kernel"]
    if method not in ("cg_brt", "hgt"):
        stages.append("encoder")
    if method == "lrt":
        stages.extend(("branch_labels", "value"))
    return sum(ledger["stages"][stage]["elapsed_s"] for stage in stages)


def pool_cost(preparation, method):
    if method in ("uniform_random", "gp_ei", "rf_bo"):
        return 0.0
    result = preparation["base"]
    if method not in ("cg_brt", "hgt"):
        result += preparation["encoder"]
    if method in ("lrt", "no_value_learning"):
        result += preparation["archive"]
    return result


def task_cost(trace, query):
    compute = trace["timing"]["task_initialization_s"] + sum(
        trace["timing"]["query_compute_s"][:query]
    )
    return {
        "compute_s": compute,
        "execution_s": sum(trace["execution_s"][:query]),
        "queries": query,
    }


def break_even(upfront_difference, recurring_saving):
    if recurring_saving > 0:
        return max(1, math.floor(upfront_difference / recurring_saving) + 1)
    if recurring_saving == 0 and upfront_difference < 0:
        return 1
    return None


def execution_time_condition(
    upfront_difference, compute_difference, saved_queries, reuse=100
):
    overhead = upfront_difference / reuse + compute_difference
    if saved_queries > 0:
        return {"direction": "above", "seconds": max(0.0, overhead / saved_queries)}
    if saved_queries < 0 and overhead < 0:
        return {"direction": "below", "seconds": overhead / saved_queries}
    return {
        "direction": "all" if saved_queries == 0 and overhead < 0 else "none",
        "seconds": None,
    }


def unilateral_spending(left, right, a, b, upfront_a, upfront_b):
    cost_a, cost_b = task_cost(left, a if a is not None else BUDGET), task_cost(
        right, b if b is not None else BUDGET
    )
    spent_a = upfront_a / 100 + cost_a["compute_s"] + cost_a["execution_s"]
    spent_b = upfront_b / 100 + cost_b["compute_s"] + cost_b["execution_s"]
    achiever = "unresolved"
    if a is not None and b is None and spent_a < spent_b:
        achiever = "lrt"
    elif b is not None and a is None and spent_b < spent_a:
        achiever = "baseline"
    return {
        "lrt_hit_query": a,
        "baseline_hit_query": b,
        "lrt_observed_s": spent_a,
        "baseline_observed_s": spent_b,
        "lower_cost_achiever": achiever,
        "reuse_tasks": 100,
    }


def paired_costs(rows, build, preparation, baseline, fraction):
    unilateral = []
    pairs, outcomes = [], {
        "both_reached": 0,
        "lrt_only": 0,
        "baseline_only": 0,
        "neither": 0,
    }
    for pool in (0, 1, 2):
        for sut in SUT_IDS:
            left, right = rows[pool, sut, "lrt"], rows[pool, sut, baseline]
            a, b = hit_query(left, fraction), hit_query(right, fraction)
            outcome = (
                "both_reached"
                if a is not None and b is not None
                else (
                    "lrt_only"
                    if a is not None
                    else "baseline_only" if b is not None else "neither"
                )
            )
            outcomes[outcome] += 1
            upfront_a = construction_cost(build, "lrt") + pool_cost(
                preparation[pool], "lrt"
            )
            upfront_b = construction_cost(build, baseline) + pool_cost(
                preparation[pool], baseline
            )
            if outcome != "both_reached":
                if outcome != "neither":
                    unilateral.append(
                        {
                            "pool": pool,
                            "sut": sut,
                            "outcome": outcome,
                            **unilateral_spending(
                                left, right, a, b, upfront_a, upfront_b
                            ),
                        }
                    )
                continue
            cost_a, cost_b = task_cost(left, a), task_cost(right, b)
            recurring_a = cost_a["compute_s"] + cost_a["execution_s"]
            recurring_b = cost_b["compute_s"] + cost_b["execution_s"]
            pairs.append(
                {
                    "pool": pool,
                    "sut": sut,
                    "lrt_queries": a,
                    "baseline_queries": b,
                    "saved_queries": b - a,
                    "compute_difference_s": cost_a["compute_s"] - cost_b["compute_s"],
                    "upfront_difference_s": upfront_a - upfront_b,
                    "recurring_saving_s": recurring_b - recurring_a,
                    "reuse": {
                        str(m): {
                            "lrt_s": upfront_a / m + recurring_a,
                            "baseline_s": upfront_b / m + recurring_b,
                            "saving_s": (upfront_b - upfront_a) / m
                            + recurring_b
                            - recurring_a,
                        }
                        for m in REUSE
                    },
                    "break_even_tasks": break_even(
                        upfront_a - upfront_b, recurring_b - recurring_a
                    ),
                    "equal_duration_condition": execution_time_condition(
                        upfront_a - upfront_b,
                        cost_a["compute_s"] - cost_b["compute_s"],
                        b - a,
                    ),
                }
            )

    def pool_average(get):
        return mean(
            [mean([get(p) for p in pairs if p["pool"] == pool]) for pool in (0, 1, 2)]
        )

    upfront = pool_average(lambda p: p["upfront_difference_s"])
    recurring = pool_average(lambda p: p["recurring_saving_s"])
    reuse = {}
    for m in REUSE:
        a = pool_average(lambda p: p["reuse"][str(m)]["lrt_s"])
        b = pool_average(lambda p: p["reuse"][str(m)]["baseline_s"])
        reuse[str(m)] = {
            "lrt_s": a,
            "baseline_s": b,
            "saving_pct": 100 * (b - a) / b if b else None,
        }
    return {
        "baseline": baseline,
        "fraction": fraction,
        "outcomes": outcomes,
        "mean_saved_queries": pool_average(lambda p: p["saved_queries"]),
        "break_even_tasks": break_even(upfront, recurring) if pairs else None,
        "equal_duration_condition": (
            execution_time_condition(
                upfront,
                pool_average(lambda p: p["compute_difference_s"]),
                pool_average(lambda p: p["saved_queries"]),
            )
            if pairs
            else {"direction": "unavailable", "seconds": None}
        ),
        "reuse": reuse,
        "pool_aggregation": "Equal pool weights after averaging jointly reached systems within each pool; pools without pairs are omitted",
        "pairs": pairs,
        "unilateral_spending": unilateral,
    }


def verify_trace(trace):
    indices = trace["selected_indices"]
    if len(indices) != BUDGET or len(set(indices)) != BUDGET:
        raise ValueError("A formal trajectory must contain 200 unique paid queries")
    if len(trace["timing"]["query_compute_s"]) != BUDGET:
        raise ValueError("Missing per-query compute timings")
    expected = np.cumsum([row["risk"] > 0.5 for row in trace["queries"]]).tolist()
    if trace["found"] != expected:
        raise ValueError("Discovery curve differs from the disclosed risks")
    if trace["method"] in ("lrt", "dart", "no_value_learning"):
        charged = sum(d["loss_charged"] for d in trace["decisions"])
        if abs(charged + trace["loss_budget_remaining"] - 20) > 1e-7:
            raise ValueError("Replacement loss ledger does not balance")
        for decision in trace["decisions"][INITIAL:]:
            if (
                decision["selected_index"] != decision["reference_index"]
                and abs(
                    decision["loss_charged"] - decision["posterior_replacement_loss"]
                )
                > 1e-9
            ):
                raise ValueError("An executed deviation was not charged")


def historical_counts(folder):
    counts = {"behavior": 0, "branches": 0, "encoder_support": 0}
    for path in (folder / "historical").rglob("*.json"):
        data = read_json(path)
        if path.stem.endswith("_returns"):
            counts["branches"] += sum(
                row["paid_replay_queries"]
                for state in data.values()
                for row in state.values()
            )
        else:
            counts["behavior"] += len(data["queries"])
    for path in (folder / "encoder/historical_support").glob("*.json"):
        counts["encoder_support"] += len(read_json(path)["queries"])
    encoder = read_json(folder / "encoder/models/training_decision_11.json")
    value = read_json(folder / "models/training_full_11.json")
    counts["encoder_validation"] = sum(
        len(row["by_source"]) * 40 for row in encoder["curve"]
    )
    counts["value_validation"] = sum(len(row["rows"]) * 40 for row in value["curve"])
    counts["total_replay_queries"] = sum(counts.values())
    counts["physical_calls"] = 0
    write_json(folder / "historical_queries.json", counts)
    return counts


def family_summary(folder):
    build = read_json(folder / "build_cost.json")
    preparation = {
        pool: read_json(folder / "pool_cache" / f"pool_{pool}.json")
        for pool in (0, 1, 2)
    }
    rows = {}
    for pool in (0, 1, 2):
        for sut in SUT_IDS:
            for method in METHODS:
                trace = read_json(
                    folder / "trajectories" / f"pool_{pool}" / sut / f"{method}_11.json"
                )
                verify_trace(trace)
                rows[pool, sut, method] = trace
            starts = [
                rows[pool, sut, method]["selected_indices"][:INITIAL]
                for method in METHODS
            ]
            if any(start != starts[0] for start in starts):
                raise ValueError("Initialization differs between methods")
    effects = []
    for label, pools in (("original", (0, 1)), ("third", (2,)), ("all", (0, 1, 2))):
        for sut in (*SUT_IDS, "macro"):
            systems = SUT_IDS if sut == "macro" else (sut,)
            for method in METHODS:
                traces = [
                    rows[pool, system, method] for pool in pools for system in systems
                ]
                effects.append(
                    {
                        "scope": label,
                        "sut": sut,
                        "method": method,
                        "positives": mean([r["positives"] for r in traces]),
                        "H200": mean([r["H200"] for r in traces]),
                        "recall": mean([r["recall"] for r in traces]),
                    }
                )
    acceptance, differences = {}, []
    for scope in ("original", "third"):
        selected = {
            r["method"]: r
            for r in effects
            if r["scope"] == scope and r["sut"] == "macro"
        }
        acceptance[scope] = {
            "H200_retained": selected["lrt"]["H200"] / selected["cg_brt"]["H200"],
            "recall_retained": selected["lrt"]["recall"] / selected["cg_brt"]["recall"],
        }
        acceptance[scope]["accepted"] = min(acceptance[scope].values()) >= 0.9
    for baseline in ("cg_brt", "hgt"):
        for metric in ("H200", "recall"):
            pairs = [
                rows[p, s, "lrt"][metric] - rows[p, s, baseline][metric]
                for p in (0, 1, 2)
                for s in SUT_IDS
            ]
            differences.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "mean_difference": mean(pairs),
                    "wins": sum(v > 1e-12 for v in pairs),
                    "ties": sum(abs(v) <= 1e-12 for v in pairs),
                    "losses": sum(v < -1e-12 for v in pairs),
                }
            )
    mechanism = []
    for pool in (0, 1, 2):
        for sut in SUT_IDS:
            trace = rows[pool, sut, "lrt"]
            mechanism.append(
                {
                    "pool": pool,
                    "sut": sut,
                    "replacements": sum(
                        d["selected_index"] != d["reference_index"]
                        for d in trace["decisions"]
                    ),
                    "spent_loss": 20 - trace["loss_budget_remaining"],
                    "different_queries_without_value": len(
                        set(trace["selected_indices"])
                        - set(rows[pool, sut, "no_value_learning"]["selected_indices"])
                    ),
                }
            )
    costs = [
        paired_costs(rows, build, preparation, method, fraction)
        for fraction in TARGET_FRACTIONS
        for method in METHODS
        if method != "lrt"
    ]
    fixed_costs = []
    for method in METHODS:
        for reuse in REUSE:
            values = []
            for pool in (0, 1, 2):
                for sut in SUT_IDS:
                    current = task_cost(rows[pool, sut, method], BUDGET)
                    values.append(
                        (
                            construction_cost(build, method)
                            + pool_cost(preparation[pool], method)
                        )
                        / reuse
                        + current["compute_s"]
                        + current["execution_s"]
                    )
            fixed_costs.append(
                {"method": method, "reuse": reuse, "mean_s": mean(values)}
            )
    curves = {
        method: np.mean(
            [rows[p, s, method]["found"] for p in (0, 1, 2) for s in SUT_IDS], axis=0
        ).tolist()
        for method in METHODS
    }
    result = {
        "family": folder.name,
        "trajectory_count": len(rows),
        "effects": effects,
        "build": build,
        "construction_s": {m: construction_cost(build, m) for m in METHODS},
        "pool_preparation": preparation,
        "costs": costs,
        "fixed_budget_costs": fixed_costs,
        "paired_differences": differences,
        "mechanism": mechanism,
        "acceptance": acceptance,
        "curves": curves,
    }
    write_json(folder / "summary.json", result)
    return result


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def render(families):
    lines = [
        "# 按功能场景独立测试：发现效果与完整成本",
        "",
        "切入、前车制动分别从本类历史构建模型。每条轨迹使用种子 11、200 次查询和共同的 10 次初始化。"
        "危险定义为实测风险 R>0.5。三个响应池均为已有场景池。",
        "",
        "成本采用同步实测的构建与计算时间，加上逐场景档案耗时重建串行等效执行时间。"
        "主情景为同类模型与同一候选池复用 100 个任务；复用次数是部署情景。",
        "",
        "在线评价采用单进程、单 CPU 线程；本文方法的虚拟积分和网络计算使用 GPU 并同步计时。"
        "RF-BO 沿用 50 棵树、10 折不确定性估计和单线程拟合配置。"
        "硬件与软件版本见 [protocol.json](protocol.json)。",
        "",
    ]
    for result in families:
        title = "切入" if result["family"] == "cutin" else "前车制动"
        lines.extend((f"## {title}", ""))
        for scope, label in (("original", "原有两池平均"), ("third", "第三池")):
            lines.extend(
                (
                    f"### {label}",
                    "",
                    "| 系统 | 危险总数 | 方法 | H200 | Recall |",
                    "|---|---:|---|---:|---:|",
                )
            )
            for row in result["effects"]:
                if row["scope"] == scope:
                    lines.append(
                        f"| {SYSTEMS.get(row['sut'], '宏平均')} | {number(row['positives'], 1)} | {NAMES[row['method']]} | {number(row['H200'])} | {number(100 * row['recall'])}% |"
                    )
            gate = result["acceptance"][scope]
            lines.extend(
                (
                    "",
                    f"相对 CG-BRT，H200 保留 {gate['H200_retained']:.2%}、Recall 保留 {gate['recall_retained']:.2%}；"
                    + (
                        "达到预设 90% 门槛。"
                        if gate["accepted"]
                        else "本类结果未达到预设 90% 门槛。"
                    ),
                    "",
                )
            )
        lines.extend(
            (
                "### 与强对照的配对差异",
                "",
                "三个池共 18 个配对任务；每个系统、每个池权重相同。覆盖率差值以百分点表示。",
                "",
                "| 对照 | 指标 | 本文减对照 | 胜 | 平 | 负 |",
                "|---|---|---:|---:|---:|---:|",
            )
        )
        for row in result["paired_differences"]:
            difference = row["mean_difference"]
            if row["metric"] == "recall":
                difference *= 100
            lines.append(
                f"| {NAMES[row['baseline']]} | {row['metric']} | {difference:+.2f} | {row['wins']} | {row['ties']} | {row['losses']} |"
            )
        lines.extend(
            (
                "",
                "### 从零构建成本",
                "",
                "| 阶段 | 时间（秒） | 占比 |",
                "|---|---:|---:|",
            )
        )
        for stage, row in result["build"]["stages"].items():
            label = {
                "kernel": "基础核",
                "encoder": "编码器及历史支持轨迹",
                "branch_labels": "历史分支与训练特征",
                "value": "价值训练、自身状态补充与验证",
            }[stage]
            lines.append(
                f"| {label} | {row['elapsed_s']:.2f} | {row['elapsed_s'] / result['construction_s']['lrt']:.1%} |"
            )
        lines.extend(
            (
                f"| 本文完整方法合计 | {result['construction_s']['lrt']:.2f} | 100% |",
                "",
                "各方法独立部署时，计入其依赖的全部共享构建费用。下表另列三个候选池的准备成本；更换候选池时重新支付对应费用。",
                "",
                "| 方法 | 一次构建（秒） | 第一池准备（秒） | 第二池准备（秒） | 第三池准备（秒） |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for method in METHODS:
            preparation = " | ".join(
                number(pool_cost(result["pool_preparation"][pool], method), 3)
                for pool in (0, 1, 2)
            )
            lines.append(
                f"| {NAMES[method]} | {number(result['construction_s'][method])} | {preparation} |"
            )
        if "historical_replay" in result:
            counts = result["historical_replay"]
            lines.extend(
                (
                    "",
                    f"历史构建共读取 {counts['total_replay_queries']:,} 次风险记录："
                    f"行为轨迹 {counts['behavior']:,} 次、分支续期 {counts['branches']:,} 次、"
                    f"编码器支持与验证 {counts['encoder_support'] + counts['encoder_validation']:,} 次、"
                    f"价值网络验证 {counts['value_validation']:,} 次。均使用已有历史记录。",
                )
            )
        lines.extend(
            (
                "",
                "### 相同发现目标的成本",
                "",
                "目标取 ceil(比例 × min(池内危险总数, 200))。例如有 3 个危险场景时，主要目标为 3 个；"
                "有 770 个时，主要目标为 160 个。从第 10 次查询开始判定达标。"
                "下表时间只比较双方均达标的配对任务，先在各池内按系统平均，再对有配对任务的池等权汇总；"
                "同时完整列出未达标情况。正的少测次数和节省比例表示本文更省。",
                "",
                "| 比例 | 对照 | 双方达标 | 仅本文达标 | 仅对照达标 | 均未达标 | 平均少测次数 | 本文总秒/任务 | 对照总秒/任务 | 节省比例 | 回本任务数 |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in result["costs"]:
            outcomes, reuse = row["outcomes"], row["reuse"]["100"]
            saving = (
                f"{number(reuse['saving_pct'])}%"
                if reuse["saving_pct"] is not None
                else "—"
            )
            payback = (
                row["break_even_tasks"] or "无持续回本点"
                if outcomes["both_reached"]
                else "无共同达标任务"
            )
            lines.append(
                f"| {row['fraction']:.0%} | {NAMES[row['baseline']]} | {outcomes['both_reached']} | {outcomes['lrt_only']} | {outcomes['baseline_only']} | {outcomes['neither']} | {number(row['mean_saved_queries'])} | {number(reuse['lrt_s'])} | {number(reuse['baseline_s'])} | {saving} | {payback} |"
            )
        lines.extend(
            (
                "",
                "### 单侧达标任务的已观察支出（补充诊断）",
                "",
                "按复用 100 次计入构建费用；比较达标方达到目标时的成本与另一方已完成 200 次的支出。"
                "本表保留单侧达标情形，不推算未执行的测试，也不并入上表双方达标的均值。",
                "",
                "| 对照 | 仅本文达标且支出更低 | 仅对照达标且支出更低 | 其他单侧达标任务 |",
                "|---|---:|---:|---:|",
            )
        )
        for row in result["costs"]:
            if row["fraction"] != 0.8:
                continue
            counts = {
                key: sum(
                    r["lower_cost_achiever"] == key for r in row["unilateral_spending"]
                )
                for key in ("lrt", "baseline", "unresolved")
            }
            lines.append(
                f"| {NAMES[row['baseline']]} | {counts['lrt']} | {counts['baseline']} | {counts['unresolved']} |"
            )
        lines.extend(
            (
                "",
                "### 复用次数敏感性（主要发现目标）",
                "",
                "| 对照 | 复用次数 | 本文秒/任务 | 对照秒/任务 | 节省比例 |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for row in result["costs"]:
            if row["fraction"] != 0.8:
                continue
            for reuse, costs in row["reuse"].items():
                saving = (
                    f"{number(costs['saving_pct'])}%"
                    if costs["saving_pct"] is not None
                    else "—"
                )
                lines.append(
                    f"| {NAMES[row['baseline']]} | {reuse} | {number(costs['lrt_s'])} | {number(costs['baseline_s'])} | {saving} |"
                )
        for row in result["costs"]:
            if row["fraction"] != 0.8 or row["baseline"] not in ("cg_brt", "hgt"):
                continue
            condition = row["equal_duration_condition"]
            description = {
                "above": f"单次执行耗时超过 {number(condition['seconds'])} 秒时，平均总成本更低。",
                "below": f"单次执行耗时低于 {number(condition['seconds'])} 秒时，平均总成本更低。",
                "all": "在统一单次执行耗时条件下，平均总成本已更低。",
                "none": "在当前查询次数和计算开销下，单纯延长单次执行时间不能形成平均成本优势。",
                "unavailable": "没有双方均达标的任务，无法估计成本收支平衡条件。",
            }[condition["direction"]]
            lines.extend(
                (
                    "",
                    f"相对 {NAMES[row['baseline']]}，复用 100 次并按各池等权重算统一单次执行耗时：{description}",
                    "",
                )
            )
        lines.extend(
            (
                f"平均每条轨迹偏离参照 {mean([r['replacements'] for r in result['mechanism']]):.2f} 次；"
                f"取消价值学习后，平均有 {mean([r['different_queries_without_value'] for r in result['mechanism']]):.2f} 个查询发生变化。",
                "",
                f"发现曲线：[图片]({result['family']}/discovery.png)。完整逐任务成本、达标状态和配对差异见本类 summary.json。",
                "",
            )
        )
        if "historical_validation" in result:
            validation = result["historical_validation"]
            lines.extend(
                (
                    f"历史验证选中第 {validation['selected_step']} 次更新的价值网络；"
                    f"H40={validation['validation']['H40']:.2f}，覆盖率={validation['validation']['recall']:.2%}，"
                    f"归一化收益差预测误差={validation['validation']['gain_mse']:.6f}。",
                    (
                        "满足历史检查点条件。"
                        if validation["validation"]["eligible"]
                        else "保留验证误差最小的诊断模型，未达到历史检查点条件。"
                    ),
                    "",
                )
            )
    lines.extend(
        (
            "## 使用说明",
            "",
            "H200 和 Recall 是主要效果指标。等发现目标的成本由完整轨迹前缀复算，"
            "不向选择器提供池内危险总数。未达标任务保留在达标率统计中。负节省比例表示在当前耗时条件下总成本更高。"
            "更换候选池需要重新支付该池准备费用；更换同类被测版本复用既有模型与池缓存。",
            "",
        )
    )
    return "\n".join(lines)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [family_summary(ROOT / family) for family in ("cutin", "brake")]
    for result in families:
        result["historical_replay"] = historical_counts(ROOT / result["family"])
        training = read_json(ROOT / result["family"] / "models/training_full_11.json")
        result["historical_validation"] = next(
            row for row in reversed(training["curve"]) if row["selected"]
        )
        result["historical_validation"]["selected_step"] = result[
            "historical_validation"
        ]["step"]
        write_json(ROOT / result["family"] / "summary.json", result)
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, values in result["curves"].items():
            label = {
                "lrt": "LRT",
                "cg_brt": "CG-BRT",
                "dart": "DART",
                "hgt": "HGT",
                "gp_ei": "GP-EI",
                "rf_bo": "RF-BO",
                "uniform_random": "Random",
                "no_value_learning": "LRT without value learning",
            }[method]
            ax.plot(np.arange(1, BUDGET + 1), values, label=label)
        ax.set(
            xlabel="Paid queries",
            ylabel="Dangerous scenarios found",
            title="Cut-in" if result["family"] == "cutin" else "Lead-vehicle braking",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(ROOT / result["family"] / "discovery.png", dpi=180)
        plt.close(fig)
    (ROOT / "comparison.md").write_text(render(families), encoding="utf-8")
    write_json(
        ROOT / "comparison.json",
        {
            "families": families,
            "physical_calls": 0,
            "trajectories": sum(r["trajectory_count"] for r in families),
        },
    )


if __name__ == "__main__":
    main()
