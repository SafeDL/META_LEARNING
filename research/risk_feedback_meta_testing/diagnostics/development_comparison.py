"""Join complete new-method results with the seven frozen same-pool references."""
from collections import defaultdict

import numpy as np

from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.statistics import bootstrap_indices
from research.risk_feedback_meta_testing.config import METHODS, RESULTS, SEEDS
from research.risk_feedback_meta_testing.evaluate import verify_or_lock


REFERENCE_NAMES = (
    "candidate", "previous_best", "frozen_original", "ras_frt_uq",
    "matched_collision_gp", "fixed_behavior", "discrete_history",
)
METRICS = ("normalized_area", "recall", "F50", "F100", "F150", "F200")


def main():
    verify_or_lock()
    development = read_json(RESULTS / "development_summary.json")
    reference = read_json(RESULTS / "existing_reference_results.json")
    protocol = read_json(RESULTS / "protocol.json")
    names = protocol["split"]["development_holdout"]
    profiles = [row for row in protocol["profiles"] if row["name"] in names]
    if development["run_count"] != 600 or reference["reused_runs"] != 840:
        raise ValueError("The full fixed comparison must finish before reporting")
    records = development["records"] + reference["records"]
    grouped = defaultdict(list)
    seen = set()
    for row in records:
        key = row["profile"], row["replicate"], row["seed"], row["method"]
        if key in seen or row["profile"] not in names:
            raise ValueError("Unpaired or duplicate comparison record")
        seen.add(key)
        grouped[row["method"], row["profile"]].append(row)
    expected_runs = {(name, replicate, seed, method)
                     for name in names for replicate in range(2)
                     for seed in SEEDS for method in (*METHODS, *REFERENCE_NAMES)}
    if seen != expected_runs:
        raise ValueError("Methods do not use the same configured pools and seeds")
    per_profile, aggregate = {}, {}
    for method in (*METHODS, *REFERENCE_NAMES):
        values = []
        for profile in profiles:
            local = grouped[method, profile["name"]]
            if len(local) != 10:
                raise ValueError("Each system needs two pools and five seeds")
            values.append({
                "profile": profile["name"], "controller": profile["controller"],
                **{metric: float(np.mean([row[metric] for row in local]))
                   for metric in METRICS},
            })
        per_profile[method] = values
        aggregate[method] = {
            metric: float(np.mean([row[metric] for row in values]))
            for metric in METRICS
        }
        expected = (development if method in METHODS else reference)["aggregate"][method]
        if any(not np.isclose(aggregate[method][metric], expected[metric], atol=1e-12,
                              rtol=0) for metric in METRICS):
            raise ValueError("Joined means differ from their source report")
    indices = bootstrap_indices(profiles)
    comparisons = {}
    for method in REFERENCE_NAMES:
        contrasts = {}
        for metric in ("normalized_area", "recall"):
            delta = np.asarray([a[metric] - b[metric] for a, b in zip(
                per_profile["meta_lookahead"], per_profile[method])])
            contrasts[metric] = {
                "mean_difference": float(delta.mean()),
                "stratified_bootstrap_95_ci_descriptive":
                    np.quantile(delta[indices].mean(1), [0.025, 0.975]).tolist(),
                "per_profile_differences": delta.tolist(),
            }
        comparisons[method] = contrasts
    summary = {
        "role": "complete same-pool development table; frozen references have different historical information",
        "new_runs": 600, "reused_frozen_runs": 840, "statistical_units": 12,
        "aggregate": aggregate, "per_profile": per_profile,
        "meta_lookahead_minus_frozen_references": comparisons,
        "not_prospective_significance_evidence": True,
        "feedback_difference": "RAS sees queried collision labels; new methods see queried risk only",
    }
    write_json(RESULTS / "development_comparison.json", summary)
    table = ["# 完整开发结果与冻结基线参照", "",
             "五个新方法与七个冻结参照使用相同的 12 个 SUT、2 池和5种子。",
             "冻结参照未完整吸收新增训练档案；新方法的机制归因使用同数据 A/B/C/D/E 消融。",
             "RAS 按原协议读取查询碰撞；新方法仅读取查询风险。以下不是新配置上的独立确认。", "",
             "| 方法 | 角色 | 发现面积 (%) | 终点召回 (%) | F50 | F100 | F150 | F200 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for method, row in aggregate.items():
        role = "同数据新对照" if method in METHODS else "冻结工程参照"
        table.append(f"| {method} | {role} | {100*row['normalized_area']:.3f} | "
                     f"{100*row['recall']:.3f} | {row['F50']:.2f} | "
                     f"{row['F100']:.2f} | {row['F150']:.2f} | {row['F200']:.2f} |")
    table += ["", "同配置内先汇总池与种子，再以系统配置为统计单位。",
              "完整配对工程差值与描述区间见 `development_comparison.json`；",
              "同信息机制消融见 `development_summary.json`。", ""]
    (RESULTS / "development_comparison.md").write_text(
        "\n".join(table), encoding="utf-8")
    print("COMPLETE DEVELOPMENT TABLE", len(aggregate), len(seen), flush=True)


if __name__ == "__main__":
    main()
