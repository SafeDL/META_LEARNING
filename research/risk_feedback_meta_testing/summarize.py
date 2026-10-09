"""Rebuild development results from physical labels and exact disclosures."""
import numpy as np

from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.statistics import bootstrap_indices

from .config import BUDGET, COHORT, METHODS, RESULTS, SEEDS
from .evaluate import EVALUATION, verify_or_lock


def main():
    verify_or_lock()
    protocol = read_json(RESULTS / "protocol.json")
    names = protocol["split"]["development_holdout"]
    profiles = [profile for profile in protocol["profiles"]
                if profile["name"] in names]
    rows, grouped = [], {method: [] for method in METHODS}
    for profile in profiles:
        local = {method: [] for method in METHODS}
        for replicate in range(2):
            with np.load(COHORT / profile["name"] / f"pool_{replicate}" /
                         "responses.npz") as bank:
                for seed in SEEDS:
                    for method in METHODS:
                        saved = read_json(EVALUATION / profile["name"] /
                                          f"pool_{replicate}" / f"{method}_{seed}.json")
                        selected = saved["selected_indices"]
                        if len(selected) != len(set(selected)) or len(selected) != BUDGET:
                            raise ValueError("Invalid development query sequence")
                        observed = [{"index": index, "continuous_risk": float(bank["risk"][index])}
                                    for index in selected]
                        if saved["observations"] != observed:
                            raise ValueError("Risk transcript differs from physical bank")
                        if len(saved["queries"]) != BUDGET:
                            raise ValueError("Hypothetical observations entered the real query log")
                        for number, (index, record) in enumerate(zip(selected, saved["queries"]), 1):
                            if (record["index"] != index or record["query_number"] != number
                                    or record["continuous_risk"] != float(bank["risk"][index])):
                                raise ValueError("Selector query record differs")
                            if record.get("constraint_enabled", False):
                                if record["predicted_terminal_count"] < record["reference_terminal_count"]:
                                    raise ValueError("Model terminal constraint violated")
                        curve = np.cumsum(bank["collision"][selected])
                        total = int(bank["collision"].sum())
                        if curve.tolist() != saved["curve"]:
                            raise ValueError("Discovery curve differs from true collisions")
                        row = {
                            "profile": profile["name"], "controller": profile["controller"],
                            "replicate": replicate, "seed": seed, "method": method,
                            "normalized_area": float(curve.mean() / total) if total else 0.0,
                            "recall": float(curve[-1] / total) if total else 1.0,
                            "missed_failures": int(total - curve[-1]),
                            "budget_upper_bound_attainment": float(curve[-1] / min(BUDGET, total))
                            if total else 1.0,
                            "all_failures_found": bool(curve[-1] == total),
                            "selector_elapsed_s": saved["selector_elapsed_s"],
                            **{f"F{count}": int(curve[count - 1])
                               for count in (10, 30, 50, 100, 150, 200)},
                        }
                        for key in ("recall", "F50", "F100", "F200"):
                            if row[key] != saved[key]:
                                raise ValueError(f"Cached metric differs: {key}")
                        rows.append(row)
                        local[method].append(row)
        for method, values in local.items():
            grouped[method].append({
                "profile": profile["name"], "controller": profile["controller"],
                **{key: float(np.mean([row[key] for row in values]))
                   for key in values[0]
                   if key not in ("profile", "controller", "replicate", "seed", "method")},
            })
    aggregate = {
        method: {key: float(np.mean([row[key] for row in values]))
                 for key in values[0] if key not in ("profile", "controller")}
        for method, values in grouped.items()
    }
    indices = bootstrap_indices(profiles)
    comparisons = {}
    pairs = (("meta_greedy", "supervised_greedy"),
             ("supervised_lookahead", "supervised_greedy"),
             ("meta_lookahead", "meta_greedy"),
             ("meta_lookahead", "supervised_lookahead"),
             ("meta_lookahead", "supervised_greedy"),
             ("meta_lookahead", "meta_unconstrained"))
    for candidate, reference in pairs:
        key = f"{candidate}_minus_{reference}"
        comparisons[key] = {}
        for metric in ("normalized_area", "recall"):
            difference = np.asarray([a[metric] - b[metric]
                                     for a, b in zip(grouped[candidate], grouped[reference])])
            comparisons[key][metric] = {
                "mean_difference": float(difference.mean()),
                "bootstrap_95_ci_descriptive": np.quantile(
                    difference[indices].mean(1), [0.025, 0.975]).tolist(),
                "per_profile_differences": difference.tolist(),
            }
    summary = {
        "role": "Fixed development comparison; not prospective significance evidence",
        "run_count": len(rows), "statistical_units": len(profiles),
        "risk_disclosures_verified": len(rows) * BUDGET,
        "records": rows, "per_profile": grouped, "aggregate": aggregate,
        "comparisons": comparisons, "new_physical_measurements": 0,
    }
    write_json(RESULTS / "development_summary.json", summary)
    table = ["# 风险反馈元适配：开发比较", "",
             "历史数据已公开，以下是开发结果；独立确认尚未完成。", "",
             "| 方法 | 发现面积 (%) | 终点召回 (%) | F50 | F100 | F150 | F200 | 选择耗时 (秒) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, values in aggregate.items():
        table.append(f"| {method} | {100*values['normalized_area']:.3f} | "
                     f"{100*values['recall']:.3f} | {values['F50']:.2f} | "
                     f"{values['F100']:.2f} | {values['F150']:.2f} | "
                     f"{values['F200']:.2f} | {values['selector_elapsed_s']:.3f} |")
    table += ["", "配对区间与完整配置、运行结果见 `development_summary.json`。",
              "统计单位是系统配置，池和模型种子先在同配置内汇总。", ""]
    (RESULTS / "development_report.md").write_text("\n".join(table), encoding="utf-8")
    print("DEVELOPMENT RESULTS VERIFIED", len(rows), len(profiles), aggregate, flush=True)


if __name__ == "__main__":
    main()
