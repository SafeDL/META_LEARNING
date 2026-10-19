"""Prefix quality, complete deployment cost and evaluator-only continuation audits."""

from zipfile import ZipFile

import numpy as np

from methods.baselines.hgt.io import read_json, write_json
from methods.common.features import prepare_pool
from benchmark.data import bank_path, measured_seconds

from ..adaptive.config import ROOT, SEED
from ..adaptive.controller import StopController
from ..adaptive.data import load_tasks
from ..adaptive.train import load_network
from methods.baselines.cg_brt.reference import ReferencePolicy
from ..posterior import model_set
from .by_family import (
    BUDGET,
    FAMILIES,
    INITIAL,
    PROJECT,
    REUSE,
    SUT_IDS,
    workspace_for,
)
from .family_report import break_even, construction_cost, pool_cost

SYSTEMS = dict(zip(SUT_IDS, ("IDM", "FVDM", "MOBIL", "VI", "MCTS", "PPO")))


def computation(trace):
    timing = trace["timing"]
    return (
        timing["task_initialization_s"]
        + sum(timing["query_compute_s"])
        + timing.get("final_decision_s", 0)
    )


def verify_prefix(trace, fixed):
    n = len(trace["queries"])
    assert INITIAL <= n <= BUDGET
    assert trace["actual_queries"] == n == len(set(trace["selected_indices"]))
    assert trace["selected_indices"] == fixed["selected_indices"][:n]
    assert trace["queries"] == fixed["queries"][:n]
    assert len(trace["disclosures"]) == n
    assert trace["execution_s"] == fixed["execution_s"][:n]
    for row, query in zip(trace["disclosures"], trace["queries"]):
        assert (
            row["collision"] is None
            and row["risk"] == query["risk"]
            and row["index"] == query["index"]
        )
    charged = sum(d["loss_charged"] for d in trace["decisions"])
    assert abs(charged + trace["loss_budget_remaining"] - 20) < 1e-8
    for actual, reference in zip(trace["decisions"], fixed["decisions"]):
        assert actual == reference
    for key in ("query_compute_s", "response_transport_s"):
        values = np.asarray(trace["timing"][key])
        assert (
            values.shape == (n,) and np.isfinite(values).all() and (values >= 0).all()
        )
    if n < BUDGET:
        assert trace["stop_reason"] == "quality_retained"
        assert trace["stopping"][-1]["stop"]
        assert trace["timing"]["final_decision_s"] > 0
    assert trace["H_tau"] == sum(q["risk"] > 0.5 for q in trace["queries"])


def mean_stopping_result(task, forecast):
    # The expected count is real-valued; integer rounding belongs only in the CDF event.
    used = next(
        (r["query"] for r in forecast if max(r["mean"]) <= r["found"] / 9),
        task["budget"],
    )
    found, full = task["found"][used - 1], task["found"][-1]
    return {
        "queries": used,
        "query_fraction": used / task["budget"],
        "found": found,
        "full_found": full,
        "recall": found / task["positives"] if task["positives"] else None,
        "early_loss": int(10 * found < 9 * full),
    }


def audit_fixed(workspace, pool, sut, fixed, network, threshold):
    path = ROOT / workspace.root.name / "diagnostics" / f"pool_{pool}" / f"{sut}.json"
    if path.exists():
        return read_json(path)
    cache = prepare_pool(workspace, pool)
    common, prior = cache["common"], cache["prior"]
    models = model_set(prior)
    anchor = ReferencePolicy(
        common["covariance"],
        common["priors"],
        common["probability"],
        common["fixed_prior"],
        SEED,
        BUDGET,
        INITIAL,
        common["initial_indices"],
    )
    remaining = np.ones(len(common["x"]), dtype=bool)
    controller = StopController(network, threshold)
    forecasts = []
    for t, query in enumerate(fixed["queries"]):
        anchor.recommend(remaining, t)
        if t >= INITIAL:
            prediction = controller.evaluate(
                models,
                remaining,
                anchor,
                fixed["queries"][:t],
                fixed["evidence"][:t],
                BUDGET,
            )
            forecasts.append(
                {
                    "query": t,
                    "found": prediction["found"],
                    "probability": prediction["model_retention_probability"],
                    "mean": prediction["model_remaining_mean"],
                }
            )
        index, risk = query["index"], query["risk"]
        anchor.observe(index, risk, risk > 0.5)
        for model in models:
            model.observe(index, risk)
        remaining[index] = False
    task = {"budget": BUDGET, "positives": fixed["positives"], "found": fixed["found"]}
    mean_result = mean_stopping_result(task, forecasts)
    result = {
        "forecast": forecasts,
        "mean_stopping": mean_result,
        "evaluation_only": True,
    }
    write_json(path, result)
    print(f"AUDIT STOP {workspace.root.name} {pool} {sut}", flush=True)
    return result


def mean(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def family_summary(workspace):
    family = workspace.root.name
    network, threshold = load_network(workspace)
    base_build = construction_cost(read_json(workspace.root / "build_cost.json"), "lrt")
    extra_build = sum(
        r["elapsed_s"]
        for r in read_json(ROOT / family / "build_cost.json")["stages"].values()
    )
    rows = []
    for pool in (0, 1, 2):
        cache = prepare_pool(workspace, pool)
        prep = pool_cost(cache["preparation_s"], "lrt")
        for sut in SUT_IDS:
            fixed = read_json(
                workspace.root / "trajectories" / f"pool_{pool}" / sut / "lrt_11.json"
            )
            trace = read_json(
                ROOT / family / "trajectories" / f"pool_{pool}" / f"{sut}.json"
            )
            verify_prefix(trace, fixed)
            with np.load(bank_path(pool, sut)) as bank:
                keep = bank["x"][:, 4] == workspace.family
                ids, risk = bank["scenario_id"][keep], bank["risk"][keep]
            durations = measured_seconds(pool, sut, ids)
            np.testing.assert_array_equal(
                trace["execution_s"], durations[trace["selected_indices"]]
            )
            np.testing.assert_array_equal(
                [q["risk"] for q in trace["queries"]], risk[trace["selected_indices"]]
            )
            audit = audit_fixed(workspace, pool, sut, fixed, network, threshold)
            for live, offline in zip(trace["stopping"], audit["forecast"]):
                assert live["query"] == offline["query"]
                np.testing.assert_allclose(
                    live["model_retention_probability"],
                    offline["probability"],
                    atol=1e-6,
                    rtol=1e-6,
                )
            full = fixed["H200"]
            n, found = trace["actual_queries"], trace["H_tau"]
            earliest = next(
                t
                for t in range(INITIAL, BUDGET + 1)
                if 10 * fixed["found"][t - 1] >= 9 * full
            )
            worst_case_stop = next(
                t
                for t in range(INITIAL, BUDGET + 1)
                if 9 * (BUDGET - t) <= fixed["found"][t - 1]
            )
            assert n <= worst_case_stop
            adaptive_recurring = computation(trace) + sum(trace["execution_s"])
            fixed_recurring = computation(fixed) + sum(fixed["execution_s"])
            forecast = audit["forecast"][: len(trace["stopping"])]
            scores = [min(r["probability"]) for r in forecast]
            outcomes = [int(10 * r["found"] >= 9 * full) for r in forecast]
            row = {
                "pool": pool,
                "sut": sut,
                "queries": n,
                "found": found,
                "fixed_found": full,
                "positives": fixed["positives"],
                "recall": trace["recall"],
                "fixed_recall": fixed["recall"],
                "discovery_rate": found / n,
                "capacity_completion": (
                    found / min(fixed["positives"], BUDGET)
                    if fixed["positives"]
                    else None
                ),
                "early_loss": int(10 * found < 9 * full),
                "missed_from_reference": full - found,
                "earliest_retention_query": earliest,
                "excess_queries": max(0, n - earliest),
                "worst_case_stop": worst_case_stop,
                "queries_saved_vs_worst_case": worst_case_stop - n,
                "model_retention_brier": float(
                    np.mean((np.asarray(scores) - outcomes) ** 2)
                ),
                "remaining_prediction_mae": float(
                    np.mean(
                        [abs(max(r["mean"]) - (full - r["found"])) for r in forecast]
                    )
                ),
                "compute_s": computation(trace),
                "selected_execution_s": sum(trace["execution_s"]),
                "final_decision_s": trace["timing"].get("final_decision_s", 0),
                "full_scan_s": float(durations.sum()),
                "adaptive_recurring_s": adaptive_recurring,
                "fixed_recurring_s": fixed_recurring,
                "mean_rule_queries": audit["mean_stopping"]["queries"],
                "mean_rule_found": audit["mean_stopping"]["found"],
                "mean_rule_early_loss": audit["mean_stopping"]["early_loss"],
                "mean_rule_recall": (
                    audit["mean_stopping"]["recall"] if fixed["positives"] else None
                ),
                "H10": fixed["found"][min(10, n) - 1],
                "H20": fixed["found"][min(20, n) - 1],
                "H50": fixed["found"][min(50, n) - 1],
                "fixed_H10": fixed["found"][9],
                "fixed_H20": fixed["found"][19],
                "fixed_H50": fixed["found"][49],
                "reuse": {
                    str(m): {
                        "adaptive_s": (base_build + extra_build + prep) / m
                        + adaptive_recurring,
                        "fixed_s": (base_build + prep) / m + fixed_recurring,
                    }
                    for m in REUSE
                },
            }
            rows.append(row)
    scopes = {}
    for name, pools in (("original", (0, 1)), ("third", (2,)), ("all", (0, 1, 2))):
        part = [r for r in rows if r["pool"] in pools]
        result = {
            key: mean(part, key)
            for key in (
                "queries",
                "found",
                "fixed_found",
                "recall",
                "fixed_recall",
                "early_loss",
                "excess_queries",
                "queries_saved_vs_worst_case",
                "model_retention_brier",
                "remaining_prediction_mae",
                "compute_s",
                "selected_execution_s",
                "mean_rule_queries",
                "mean_rule_found",
                "mean_rule_early_loss",
                "mean_rule_recall",
                "H10",
                "H20",
                "H50",
                "fixed_H10",
                "fixed_H20",
                "fixed_H50",
            )
        }
        result["H_retained"] = result["found"] / result["fixed_found"]
        result["recall_retained"] = result["recall"] / result["fixed_recall"]
        result["quality_accepted"] = (
            min(result["H_retained"], result["recall_retained"]) >= 0.9
        )
        result["pooled_recall"] = sum(r["found"] for r in part) / sum(
            r["positives"] for r in part
        )
        result["query_reduction"] = 1 - result["queries"] / BUDGET
        result["reuse"] = {}
        for m in REUSE:
            a = float(np.mean([r["reuse"][str(m)]["adaptive_s"] for r in part]))
            f = float(np.mean([r["reuse"][str(m)]["fixed_s"] for r in part]))
            exhaustive = mean(part, "full_scan_s")
            result["reuse"][str(m)] = {
                "adaptive_s": a,
                "fixed_s": f,
                "full_scan_s": exhaustive,
                "saving_vs_fixed": 1 - a / f,
                "saving_vs_full": 1 - a / exhaustive,
            }
        result["cost_accepted_M100"] = result["reuse"]["100"]["saving_vs_fixed"] > 0
        scopes[name] = result
    recurring_saving = mean(rows, "fixed_recurring_s") - mean(
        rows, "adaptive_recurring_s"
    )
    result = {
        "family": family,
        "threshold": threshold,
        "historical_selection": read_json(ROOT / family / "training.json")["selected"],
        "base_build_s": base_build,
        "stopping_build_s": extra_build,
        "additional_break_even_tasks": break_even(extra_build, recurring_saving),
        "scopes": scopes,
        "tasks": rows,
        "accepted": all(scopes[s]["quality_accepted"] for s in ("original", "third"))
        and scopes["all"]["cost_accepted_M100"],
    }
    write_json(ROOT / family / "summary.json", result)
    return result


def render(families):
    lines = [
        "# 自适应预算风险测试：发现质量与完整成本",
        "",
        "种子为 11；切入、制动分别运行，前 10 次共同初始化，最高 200 次。危险定义为实测风险 R>0.5。",
        "固定预算 LRT 复用已有轨迹；自适应方法真实执行 36 条接口回放。三个池均为已有评价池，新增物理仿真为零。",
        "",
        "停止后发现数保持不变。覆盖率同时报告系统等权平均和全部危险场景合并计数结果。"
        "成本包含原基础构建、新增历史轨迹与停止网络、池准备、终止判断以及实际场景执行的串行等效时间。",
        "",
    ]
    for result in families:
        title = "切入" if result["family"] == "cutin" else "前车制动"
        selection = result["historical_selection"]
        lines += [
            f"## {title}",
            "",
            f"历史验证选中第 {selection['step']} 次更新，停止阈值 {result['threshold']}；"
            f"历史发现数保留 {selection['found']/selection['full_found']:.2%}，"
            f"覆盖率保留 {selection['recall']/selection['full_recall']:.2%}，"
            f"平均查询减少 {1-selection['query_fraction']:.2%}。",
            (
                "本类质量与成本均通过预设验收。"
                if result["accepted"]
                else "本类未通过完整验收，作为诊断候选保留。"
            ),
            "",
            "| 范围 | 平均查询 | 本文发现数 | 固定预算发现数 | 本文 Recall | 固定 Recall | 发现数保留 | 覆盖率保留 | 质量验收 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for scope, label in (
            ("original", "原有两池"),
            ("third", "第三池"),
            ("all", "全部"),
        ):
            r = result["scopes"][scope]
            lines.append(
                f"| {label} | {r['queries']:.2f} | {r['found']:.2f} | {r['fixed_found']:.2f} | {r['recall']:.2%} | {r['fixed_recall']:.2%} | {r['H_retained']:.2%} | {r['recall_retained']:.2%} | {'通过' if r['quality_accepted'] else '未通过'} |"
            )
        all_rows = result["scopes"]["all"]
        lines += [
            "",
            f"合并计数覆盖率 {all_rows['pooled_recall']:.2%}；查询减少 {all_rows['query_reduction']:.2%}。",
            f"相比只利用剩余查询上限即可推出的保守停止点，收益预测平均再少测 {all_rows['queries_saved_vs_worst_case']:.2f} 次；该诊断不增加训练或查询。",
            "",
            "| 池 | 系统 | 危险总数 | 查询数 | 发现数 / 固定发现数 | Recall | 提前停止未保留90% | 比最早保留目标多测 |",
            "|---|---|---:|---:|---:|---:|---|---:|",
        ]
        for r in result["tasks"]:
            lines.append(
                f"| {r['pool']} | {SYSTEMS[r['sut']]} | {r['positives']} | {r['queries']} | {r['found']} / {r['fixed_found']} | {r['recall']:.2%} | {'是' if r['early_loss'] else '否'} | {r['excess_queries']} |"
            )
        lines += [
            "",
            "### 少量反馈与停止诊断",
            "",
            "| 方法 | H10 | H20 | H50 | 终止发现数 | 平均查询 | 提前停止未保留90%的任务比例 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| 固定预算 LRT | {all_rows['fixed_H10']:.2f} | {all_rows['fixed_H20']:.2f} | {all_rows['fixed_H50']:.2f} | {all_rows['fixed_found']:.2f} | 200 | 0% |",
            f"| 完整自适应方法 | {all_rows['H10']:.2f} | {all_rows['H20']:.2f} | {all_rows['H50']:.2f} | {all_rows['found']:.2f} | {all_rows['queries']:.2f} | {all_rows['early_loss']:.2%} |",
            f"| 均值停止（评价端前缀诊断） | — | — | — | {all_rows['mean_rule_found']:.2f} | {all_rows['mean_rule_queries']:.2f} | {all_rows['mean_rule_early_loss']:.2%} |",
            "",
            "H10 为共同初始化结果。均值停止复用同一网络和固定轨迹，不增加独立训练或正式轨迹。",
            f"停止前续期发现数预测的平均绝对误差为 {all_rows['remaining_prediction_mae']:.2f} 个；质量保留事件的 Brier 分数为 {all_rows['model_retention_brier']:.4f}。这些是评价端误差诊断。",
            "",
            "### 完整成本",
            "",
            f"原基础模型构建 {result['base_build_s']/60:.2f} 分钟；新增历史续期与停止训练 {result['stopping_build_s']/60:.2f} 分钟。",
            f"相对固定预算 LRT，新增构建的平均回本任务数：{result['additional_break_even_tasks'] or '无持续回本点'}。",
            "",
            "| 复用次数 | 本文秒/任务 | 固定LRT秒/任务 | 全量扫描秒/任务 | 相对固定节省 | 相对全量节省 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for m, c in all_rows["reuse"].items():
            lines.append(
                f"| {m} | {c['adaptive_s']:.2f} | {c['fixed_s']:.2f} | {c['full_scan_s']:.2f} | {c['saving_vs_fixed']:.2%} | {c['saving_vs_full']:.2%} |"
            )
        lines += [
            "",
            f"复用100次成本验收：{'通过' if all_rows['cost_accepted_M100'] else '未通过'}。"
            "质量和成本分别判断，所有任务均纳入结果。相对全量扫描的节省对应上表实际覆盖率。",
            "",
        ]
    return "\n".join(lines)


def main():
    families = [family_summary(workspace_for(f)) for f in FAMILIES]
    (ROOT / "comparison.md").write_text(render(families), encoding="utf-8")
    historical_count, historical_queries = 0, 0
    for family in FAMILIES:
        for split in ("train", "validation"):
            tasks = load_tasks(family, split)
            assert len(tasks) == 18
            for task in tasks:
                cumulative = np.r_[0, task["found"]]
                assert len(task["found"]) == task["budget"]
                assert [r["step"] for r in task["records"]] == list(
                    range(task["initial"], task["budget"])
                )
                for record in task["records"]:
                    t = record["step"]
                    assert record["found"] == cumulative[t]
                    assert record["features"].shape == (3, len(record["lengths"]), 136)
                    assert record["returns"] == [
                        int(cumulative[t + b] - cumulative[t])
                        for b in record["lengths"]
                    ]
                historical_count += 1
                historical_queries += task["budget"]
    write_json(
        ROOT / "comparison.json",
        {
            "families": families,
            "formal_trajectories": 36,
            "physical_calls": 0,
            "all_families_accepted": all(f["accepted"] for f in families),
        },
    )
    with ZipFile(ROOT / "frozen_inputs.zip") as archive:
        for name in archive.namelist():
            assert (PROJECT / name).read_bytes() == archive.read(name), name
    write_json(
        ROOT / "verification.json",
        {
            "status": "complete",
            "formal_trajectories": 36,
            "historical_trajectories": historical_count,
            "historical_replay_queries": historical_queries,
            "paid_queries": sum(r["queries"] for f in families for r in f["tasks"]),
            "strict_policy_prefixes": True,
            "replacement_ledgers": True,
            "raw_risk_and_execution_times": True,
            "collision_disclosures": 0,
            "terminal_decision_time_included": True,
            "frozen_policy_and_weights": True,
            "performance_acceptance": {f["family"]: f["accepted"] for f in families},
            "physical_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
