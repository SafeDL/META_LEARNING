"""Recompute available comparisons without treating partial banks as completed."""
import numpy as np

from methods.history_guided_testing.archive import RiskArchive
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.scenarios import parameter_cells

from .baselines import RETAINED
from .config import CHECKPOINTS, OUTPUT, POOL_SEEDS, SEEDS, SUT_IDS
from .evaluation import metrics


def main():
    rows, missing = [], []
    lines = [
        "# 代表 SUT 的现有完整指标索引", "",
        "当前为开发结果。表中重复数明确标注；未完成池与方法不填补，不作显著优势确认。",
        "新机制对照共享风险与碰撞反馈；冻结参照保持原反馈。扩展行为模型使用过额外历史配置，属于工程参照。", "",
        "| 池 | SUT | 方法 | 完成/5 | 全池碰撞 | Area % | Recall % | F10 | F30 | F50 | F100 | F150 | F200 | 遗漏 | 上界 % | 风险 QDScore | 选择秒 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pool_id in range(len(POOL_SEEDS)):
        for sut_id in SUT_IDS:
            folder = OUTPUT / f"pool_{pool_id}" / sut_id
            if not (folder / "responses.npz").exists():
                missing.append({"pool": pool_id, "sut_id": sut_id, "reason": "bank not complete"})
                continue
            with np.load(folder / "responses.npz") as bank:
                cells = parameter_cells(bank["x"])
                for method in RETAINED:
                    category = "baselines"
                    runs = []
                    for seed in SEEDS:
                        path = folder / category / f"{method}_{seed}.json"
                        if not path.exists():
                            continue
                        run = read_json(path)
                        selected = run["selected_indices"]
                        if len(selected) != 200 or len(set(selected)) != 200:
                            raise ValueError("Incorrect selection budget in result")
                        rebuilt = metrics(bank["collision"], selected)
                        for key in ("area_200", "recall_200", "F200", "checkpoints", "curve"):
                            if run[key] != rebuilt[key]:
                                raise ValueError(f"Metric differs from full-pool events: {path}, {key}")
                        expected = [{"index": index, "risk": float(bank["risk"][index]),
                                     "collision": bool(bank["collision"][index])}
                                    for index in selected]
                        if category == "baselines":
                            for record in expected:
                                record["risk" if method == "ras_frt_uq" else "collision"] = None
                        if run["disclosures"] != expected:
                            raise ValueError(f"Disclosure log differs from permitted actual feedback: {path}")
                        archive = RiskArchive()
                        for index in selected:
                            archive.observe(cells[index], float(bank["risk"][index]))
                        runs.append({**rebuilt, "selector_elapsed_s": run["selector_elapsed_s"],
                                     **archive.metrics()})
                    if len(runs) != len(SEEDS):
                        missing.append({"pool": pool_id, "sut_id": sut_id, "method": method,
                                        "completed": len(runs), "expected": len(SEEDS)})
                    if not runs:
                        continue
                    value = {
                        "pool": pool_id, "sut_id": sut_id, "method": method, "runs": len(runs),
                        "total_collisions": runs[0]["total_collisions"],
                        **{key: float(np.mean([run[key] for run in runs]))
                           for key in ("F200", "missed", "risk_qd_score", "selector_elapsed_s")},
                        **{key: float(np.mean([run[key] for run in runs])) if runs[0][key] is not None else None
                           for key in ("area_200", "recall_200", "terminal_ceiling_attainment")},
                        "checkpoints": {str(t): float(np.mean([run["checkpoints"][str(t)] for run in runs]))
                                        for t in CHECKPOINTS},
                    }
                    rows.append(value)
                    fields = [str(pool_id), sut_id, method, f"{len(runs)}/5", str(value["total_collisions"])]
                    fields += [f"{100 * value[key]:.3f}" if value[key] is not None else "无碰撞池"
                               for key in ("area_200", "recall_200")]
                    fields += [f"{value['checkpoints'][str(t)]:.1f}" for t in CHECKPOINTS]
                    fields += [f"{value['missed']:.1f}",
                               f"{100 * value['terminal_ceiling_attainment']:.2f}" if value["terminal_ceiling_attainment"] is not None else "—",
                               f"{value['risk_qd_score']:.3f}", f"{value['selector_elapsed_s']:.3f}"]
                    lines.append("| " + " | ".join(fields) + " |")
    lines += ["", f"未完成项：{len(missing)}，详见 `comparison.json`。风险 QDScore 为辅助指标，不参与成功判定。"]
    write_json(OUTPUT / "baseline_details.json", {
        "stage": "frozen baselines",
        "complete": not missing, "rows": rows, "missing": missing,
        "significant_advantage_established": False,
    })
    (OUTPUT / "baseline_details.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    baseline_rows = [row for row in rows if row["method"] in RETAINED]
    complete_baselines = (len(baseline_rows) == len(POOL_SEEDS) * len(SUT_IDS) * len(RETAINED)
                          and all(row["runs"] == len(SEEDS) for row in baseline_rows))
    aggregate = []
    if complete_baselines:
        for method in RETAINED:
            grouped = [row for row in baseline_rows if row["method"] == method]
            aggregate.append({
                "method": method,
                "area_pct": 100 * float(np.mean([row["area_200"] for row in grouped])),
                "recall_pct": 100 * float(np.mean([row["recall_200"] for row in grouped])),
                "F200": float(np.mean([row["F200"] for row in grouped])),
                "checkpoints": {str(t): float(np.mean([row["checkpoints"][str(t)] for row in grouped]))
                                for t in CHECKPOINTS},
            })
    write_json(OUTPUT / "baseline_comparison.json", {
        "complete": complete_baselines, "rows": baseline_rows, "aggregate": aggregate,
        "scope": "six fixed representative SUTs, two shared pools, five selection seeds",
        "information_differences": "behavior models used additional public configurations; original feedback permissions retained",
    })
    baseline_lines = ["# 六 SUT 的完整现有基线对比", "",
                      f"完成状态：{'全部 840 条曲线完成并重算核对' if complete_baselines else '仍有缺失项'}。",
                      "两项主指标先在 SUT 内汇总两池和五种子，再对六个固定 SUT 等权汇总。",
                      "驾驶模型族相关，种子与池不计为新的驾驶算法样本；额外历史配置的工程参照按协议单列。", "",
                      "| 方法 | Area % | Recall % | F10 | F30 | F50 | F100 | F150 | F200 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        fields = [row["method"], f"{row['area_pct']:.3f}", f"{row['recall_pct']:.3f}"]
        fields += [f"{row['checkpoints'][str(t)]:.2f}" for t in CHECKPOINTS]
        baseline_lines.append("| " + " | ".join(fields) + " |")
    baseline_lines += ["", "完整逐池／逐 SUT 结果及开销见 [指标索引](comparison.md)，"
                       "原始 840 条选择与披露日志保留在各池的 `baselines/` 中。"]
    (OUTPUT / "baseline_comparison.md").write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")
    print("REBUILT COMPARISON", len(rows), "rows", len(missing), "missing items", flush=True)


if __name__ == "__main__":
    main()
