"""Discovery, paired retention and evaluator-only replacement diagnostics."""

import numpy as np

from methods.common.catalog import DART_VARIANTS
from benchmark.evaluation import metrics
from methods.baselines.hgt.io import read_json, write_json

from .config import (
    ACCEPTANCE_METRICS,
    RETENTION_FLOOR,
    ROOT,
    SEEDS,
    SUT_IDS,
    bank_root,
    pools,
    variants,
)
from .evaluate import trace_path


def measured(stage, pool, sut, policy, seed, bank):
    trace = read_json(trace_path(stage, pool, sut, policy, seed))
    selected = trace["selected_indices"]
    if len(selected) != 200 or len(set(selected)) != 200:
        raise ValueError("Trace must contain exactly 200 distinct paid queries")
    for q, index in zip(trace["queries"], selected):
        if q["index"] != index or q["risk"] != float(bank["risk"][index]):
            raise ValueError("Risk disclosure differs from the response bank")
    high = bank["risk"] > 0.5
    score = metrics(high, selected)
    decisions = trace["decisions"]
    replacements = (
        []
        if policy == "cg_brt"
        else [d for d in decisions if d["reference_index"] != d["selected_index"]]
    )
    actual_loss = sum(
        int(high[d["reference_index"]] and not high[d["selected_index"]])
        for d in replacements
    )
    predicted_loss = sum(d.get("posterior_replacement_loss", 0.0) for d in replacements)
    return {
        "stage": stage,
        "pool": pool,
        "sut": sut,
        "policy": policy,
        "seed": seed,
        "H200": score["F200"],
        "high_risk_recall_pct": 100 * score["recall_200"],
        "replacements": len(replacements),
        "actual_same_history_loss": actual_loss,
        "posterior_same_history_loss": float(predicted_loss),
        "charged_loss": float(sum(d.get("loss_charged", 0.0) for d in decisions)),
        "rejected_candidates": sum(d.get("rejected_candidates", 0) for d in decisions),
        "mean_information_value_mc": float(
            np.mean([d.get("information_value_mc", 0.0) for d in decisions])
        ),
        "selector_elapsed_s": trace.get("selector_elapsed_s", 0.0),
        "diagnostic_labels_available_only_to_evaluator": True,
    }


def main(stage="development"):
    rows = []
    for pool in pools(stage):
        for sut in SUT_IDS:
            with np.load(
                bank_root(stage) / f"pool_{pool}" / sut / "responses.npz"
            ) as stored:
                bank = {"risk": stored["risk"]}
            for policy in variants(stage):
                for seed in SEEDS:
                    rows.append(measured(stage, pool, sut, policy, seed, bank))
    keys = (
        *ACCEPTANCE_METRICS,
        "replacements",
        "actual_same_history_loss",
        "posterior_same_history_loss",
        "charged_loss",
        "rejected_candidates",
        "mean_information_value_mc",
        "selector_elapsed_s",
    )
    aggregate = {
        p: {k: float(np.mean([r[k] for r in rows if r["policy"] == p])) for k in keys}
        for p in variants(stage)
    }
    ratios = {
        k: aggregate["full"][k] / aggregate["cg_brt"][k] for k in ACCEPTANCE_METRICS
    }
    by_sut = {
        sut: {
            p: {
                k: float(
                    np.mean(
                        [r[k] for r in rows if r["sut"] == sut and r["policy"] == p]
                    )
                )
                for k in ACCEPTANCE_METRICS
            }
            for p in variants(stage)
        }
        for sut in SUT_IDS
    }
    rng = np.random.default_rng(970031)
    # Resample system blocks; repeated selector seeds are not independent systems.
    deltas = np.array(
        [
            [by_sut[s]["full"][k] - by_sut[s]["cg_brt"][k] for k in ACCEPTANCE_METRICS]
            for s in SUT_IDS
        ]
    )
    resampled = deltas[rng.integers(0, len(SUT_IDS), (10000, len(SUT_IDS)))].mean(1)
    intervals = {
        k: np.quantile(resampled[:, i], [0.025, 0.975]).tolist()
        for i, k in enumerate(ACCEPTANCE_METRICS)
    }
    result = {
        "method_names": DART_VARIANTS,
        "stage": stage,
        "aggregate": aggregate,
        "by_sut": by_sut,
        "rows": rows,
        "retention": ratios,
        "retention_floor": RETENTION_FLOOR,
        "evaluation_metrics": list(ACCEPTANCE_METRICS),
        "evaluation_revision": "2026-10-13: user selected discovery count and coverage only; existing weights and trajectories retained",
        "effect_accepted": all(v >= RETENTION_FLOOR for v in ratios.values()),
        "paired_delta_95pct_system_block_bootstrap": intervals,
        "statistical_noninferiority_claim": False,
        "mechanism_implemented": True,
        "mechanism_changes_paid_decisions": aggregate["full"]["replacements"] > 0,
    }
    write_json(ROOT / f"{stage}_comparison.json", result)
    lines = [
        f"# DART：决策相关风险迁移（{stage}）",
        "",
        "评价仅比较 H200 和高风险覆盖率（Recall）；两项均保留配对 CG-BRT 的至少 90%。",
        "",
        "2026-10-13 按用户要求更新评价指标；已有模型权重、实测响应和查询轨迹保持不变。",
        "",
        "| 方法 | H200 | 覆盖率 % | 替换次数 | 实际同历史损失 | 预测损失 | 时间 s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p, a in aggregate.items():
        lines.append(
            f"| {DART_VARIANTS[p]} | {a['H200']:.3f} | {a['high_risk_recall_pct']:.3f} | "
            f"{a['replacements']:.2f} | {a['actual_same_history_loss']:.2f} | {a['posterior_same_history_loss']:.3f} | {a['selector_elapsed_s']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"DART 相对 CG-BRT 的保留比例：{ratios}",
            "",
            "效果已验收。" if result["effect_accepted"] else "机制已实现、效果未验收。",
            "",
            "实际同历史损失仅由评价端计算，未向选择器提供未查询标签。",
            "置信区间以六个系统为块；10% 为工程门槛，不声明统计非劣效。",
        ]
    )
    (ROOT / f"{stage}_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    from .plots import plot_report

    plot_report(result)
    print(result["retention"], flush=True)
    return result


if __name__ == "__main__":
    main()
