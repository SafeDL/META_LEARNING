"""Recompute all comparisons from shared measured banks and paid traces."""

import itertools

import numpy as np

from methods.common.catalog import NAMES as LABELS
from benchmark.config import OUTPUT, SUT_IDS
from benchmark.evaluation import metrics
from methods.baselines.cg_brt.config import THRESHOLD
from methods.baselines.hgt.io import read_json, write_json

from .config import BUDGET, INITIAL_QUERIES, METRICS, POLICY, PROJECT, ROOT, SEEDS


COLLISION_METRICS = ("F200", "collision_recall_pct")


def measured_row(path, info, pool, sut, seed, bank):
    trace = read_json(path)
    selected = trace["selected_indices"]
    if len(selected) != BUDGET or len(set(selected)) != BUDGET:
        raise ValueError(f"Invalid or repeated queries: {path}")
    observations = trace.get("disclosures") or trace["queries"]
    if len(observations) != BUDGET:
        raise ValueError(f"Incomplete disclosure record: {path}")
    counts = {"R": 0, "C": 0}
    for index, observation in zip(selected, observations):
        if index not in range(len(bank["risk"])) or observation["index"] != index:
            raise ValueError(f"Invalid query or disclosure order: {path}")
        if observation.get("risk") is not None:
            counts["R"] += 1
            if observation["risk"] != float(bank["risk"][index]):
                raise ValueError(f"Disclosed R differs from the bank: {path}")
        if observation.get("collision") is not None:
            counts["C"] += 1
            if bool(observation["collision"]) != bool(bank["collision"][index]):
                raise ValueError(f"Disclosed C differs from the bank: {path}")
    expected = {
        key: BUDGET if key in info["recorded_feedback"] else 0 for key in counts
    }
    if counts != expected:
        raise ValueError(f"Recorded feedback channels differ: {path}")
    high = bank["risk"] > THRESHOLD
    measured, collision = metrics(high, selected), metrics(bank["collision"], selected)
    total = int(high.sum())
    initial_positive = int(high[selected[:INITIAL_QUERIES]].sum())
    return {
        "method": info["method"],
        "pool": pool,
        "sut": sut,
        "seed": seed,
        "high_risk_recall_pct": 100 * measured["recall_200"],
        "H200": measured["F200"],
        "total_high_risk": total,
        "budget_recall_ceiling_pct": 100 * min(BUDGET, total) / total,
        "initialization_recall_ceiling_pct": 100
        * min(total, BUDGET - INITIAL_QUERIES + initial_positive)
        / total,
        "collision_recall_pct": 100 * collision["recall_200"],
        "F200": collision["F200"],
        "recorded_feedback_counts": counts,
        "trace": path.relative_to(PROJECT).as_posix(),
    }


def aggregate(rows, infos):
    return {
        info["method"]: {
            metric: float(
                np.mean(
                    [row[metric] for row in rows if row["method"] == info["method"]]
                )
            )
            for metric in (*METRICS, *COLLISION_METRICS)
        }
        for info in infos
    }


def sign_randomization(deltas):
    observed = float(np.mean(deltas))
    null = [
        np.mean(np.asarray(signs) * deltas)
        for signs in itertools.product((-1, 1), repeat=len(deltas))
    ]
    return sum(value >= observed - 1e-15 for value in null) / len(null)


def main():
    catalog = read_json(ROOT / "baseline_catalog.json")
    current = {
        "method": POLICY,
        "group": "current method",
        "offline_history": "six historical SUTs",
        "feedback_used": "R only",
        "recorded_feedback": "R",
        "seeds": list(SEEDS),
    }
    full = [current, *[info for info in catalog if len(info["seeds"]) == 5]]
    supplemental = [current, *[info for info in catalog if len(info["seeds"]) == 1]]
    rows = []
    for pool in (0, 1):
        coordinates = None
        for sut in SUT_IDS:
            with np.load(OUTPUT / f"pool_{pool}/{sut}/responses.npz") as stored:
                bank = {name: stored[name] for name in ("x", "risk", "collision")}
                if coordinates is None:
                    coordinates = bank["x"].copy()
                np.testing.assert_array_equal(coordinates, bank["x"])
                for info in (current, *catalog):
                    for seed in info["seeds"]:
                        method = info["method"]
                        path = (
                            ROOT / f"driving/pool_{pool}/{sut}/{POLICY}_{seed}.json"
                            if method == POLICY
                            else OUTPUT
                            / f"pool_{pool}/{sut}/baselines/{method}_{seed}.json"
                        )
                        row = measured_row(path, info, pool, sut, seed, bank)
                        if method == POLICY:
                            initial = read_json(
                                OUTPUT
                                / f"pool_{pool}/{sut}/baselines/frozen_original_{seed}.json"
                            )
                            if (
                                read_json(path)["selected_indices"][:10]
                                != initial["selected_indices"][:10]
                            ):
                                raise ValueError(
                                    "Current method changed shared initialization"
                                )
                        rows.append(row)
    full_ids = {info["method"] for info in full}
    supplemental_ids = {info["method"] for info in supplemental}
    full_rows = [row for row in rows if row["method"] in full_ids]
    supplemental_rows = [
        row for row in rows if row["method"] in supplemental_ids and row["seed"] == 11
    ]
    comparisons = {}
    for info in full[1:]:
        method = info["method"]
        comparisons[method] = {}
        for metric in METRICS:
            deltas = np.asarray(
                [
                    np.mean(
                        [
                            row[metric]
                            for row in full_rows
                            if row["method"] == POLICY and row["sut"] == sut
                        ]
                    )
                    - np.mean(
                        [
                            row[metric]
                            for row in full_rows
                            if row["method"] == method and row["sut"] == sut
                        ]
                    )
                    for sut in SUT_IDS
                ]
            )
            comparisons[method][metric] = {
                "sut_deltas": dict(zip(SUT_IDS, deltas.tolist())),
                "mean_delta": float(np.mean(deltas)),
                "one_sided_signflip_p": sign_randomization(deltas),
            }
    candidate = [row for row in full_rows if row["method"] == POLICY]
    by_sut = {
        sut: {
            "pool_high_risk_counts": {
                str(pool): next(
                    row["total_high_risk"]
                    for row in candidate
                    if row["sut"] == sut and row["pool"] == pool
                )
                for pool in (0, 1)
            },
            **{
                key: float(
                    np.mean([row[key] for row in candidate if row["sut"] == sut])
                )
                for key in (
                    "H200",
                    "high_risk_recall_pct",
                    "budget_recall_ceiling_pct",
                    "initialization_recall_ceiling_pct",
                )
            },
        }
        for sut in SUT_IDS
    }
    ceiling = {
        key: float(np.mean([row[key] for row in candidate]))
        for key in (
            "high_risk_recall_pct",
            "budget_recall_ceiling_pct",
            "initialization_recall_ceiling_pct",
        )
    }
    ceiling["micro_recall_pct"] = (
        100
        * sum(row["H200"] for row in candidate)
        / sum(row["total_high_risk"] for row in candidate)
    )
    report = {
        "status": "complete",
        "positive_rule": "measured R > 0.5, strictly greater",
        "aggregation": "macro mean over SUTs, pools and available seeds; five-seed and seed11 tables separated",
        "full_method_metadata": full,
        "supplemental_method_metadata": supplemental,
        "full_aggregate": aggregate(full_rows, full),
        "supplemental_aggregate": aggregate(supplemental_rows, supplemental),
        "trajectory_metrics": rows,
        "paired_sut_comparisons": comparisons,
        "coverage_upper_bounds": ceiling,
        "by_sut": by_sut,
        "verification": {
            "total_unique_trace_files_checked": len(rows),
            "full_five_seed_trace_files": len(full_rows),
            "supplemental_seed11_rows": len(supplemental_rows),
            "paid_query_records_checked": BUDGET * len(rows),
            "unique_queries_and_shared_banks_verified": True,
            "disclosed_responses_verified": True,
            "current_selector_collision_disclosures": 0,
            "current_common_initialization_verified": True,
            "new_selector_runs_during_comparison": 0,
            "new_physical_calls": 0,
        },
    }
    write_json(ROOT / "comparison.json", report)
    write_report(report)
    print(report["verification"])
    print(report["full_aggregate"][POLICY])
    print(ceiling)


def write_report(report):
    lines = [
        "# CG-BRT统一基准结果",
        "",
        "高风险标签严格定义为实测R>0.5。CG-BRT只接收已请求的R，C不进入选择器。全表使用同一六SUT、两个2048点场景池和预算200；十次初始化计入预算。历史信息与反馈通道按方法实际记录，各方法初始化规则并不全部相同。",
        "",
        "评价比较200次测试的高风险发现数H200和覆盖率Recall；覆盖率先逐轨迹计算，再宏平均。方法全称与反馈条件见[方法索引](../../methods/README.md)。",
        "",
    ]
    for title, metadata_key, aggregate_key in (
        ("五种子完整比较", "full_method_metadata", "full_aggregate"),
        ("种子11补充比较", "supplemental_method_metadata", "supplemental_aggregate"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| 方法 | 历史信息 | 推断反馈 | H200 | 覆盖率 % |",
                "|---|---|---|---:|---:|",
            ]
        )
        metadata = {info["method"]: info for info in report[metadata_key]}
        for method, measured in sorted(
            report[aggregate_key].items(),
            key=lambda item: item[1][METRICS[0]],
            reverse=True,
        ):
            values = " | ".join(f"{measured[key]:.3f}" for key in METRICS)
            info = metadata[method]
            lines.append(
                f"| {LABELS.get(method, method)} | {info['offline_history']} | {info['feedback_used']} | {values} |"
            )
        lines.append("")
    lines.extend(
        [
            "额外48个IDM/FVDM配置的工程参照会在历史信息列明确披露。补充表仅种子11；风险混合控制有的将C写入评价记录，但推断只用R。各实现并非只改变一个组件，因此不能把补充表当作单因素消融。",
            "",
            "## 当前方法的预算覆盖上限",
            "",
            "| SUT | 两池高风险总数 | H200 | 实际覆盖率 % | 预算理论上限 % | 固定初始化后上限 % |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for sut, measured in report["by_sut"].items():
        counts = "/".join(
            str(value) for value in measured["pool_high_risk_counts"].values()
        )
        values = " | ".join(
            f"{measured[key]:.3f}"
            for key in (
                "H200",
                "high_risk_recall_pct",
                "budget_recall_ceiling_pct",
                "initialization_recall_ceiling_pct",
            )
        )
        lines.append(f"| {sut} | {counts} | {values} |")
    lines.extend(
        [
            "",
            "覆盖率为H200/N_H，理论上限为min(200,N_H)/N_H；固定初始化后为min(N_H,190+H10)/N_H。这些是理想计数上限，未附加决策规则限制。",
            "",
            "## 辅助碰撞评价（同一五种子轨迹）",
            "",
            "| 方法 | 碰撞发现数 F200 | 碰撞覆盖率 % |",
            "|---|---:|---:|",
        ]
    )
    for method, measured in sorted(
        report["full_aggregate"].items(),
        key=lambda item: item[1][COLLISION_METRICS[0]],
        reverse=True,
    ):
        values = " | ".join(f"{measured[key]:.3f}" for key in COLLISION_METRICS)
        lines.append(f"| {LABELS.get(method, method)} | {values} |")
    lines.extend(
        [
            "",
            "[comparison.json](comparison.json)包含全部轨迹指标及六SUT块的配对检验。只报告现有基准证据，排名与显著性分别判断；没有新增基线轨迹或物理测试。",
            "",
            "```powershell",
            "conda run -n metadrive python -B -m methods.baselines.cg_brt.evaluation.compare",
            "```",
            "",
        ]
    )
    (ROOT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
