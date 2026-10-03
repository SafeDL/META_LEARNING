"""Evaluate all methods on the original S01 failure ground truth."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audit import protected_hashes
from .benchmark import BUDGET, CHECKPOINTS, SEEDS
from .common import read_json, write_json
from .diagnostics import prediction_metrics
from .experiment import BASELINES, IMPLEMENTATION
from .qd import RiskArchive
from .s01 import ROOT, TARGET, cells_for, response_rows


def metrics(run, scenes, rows):
    indices = np.asarray(run["selected_indices"], int)
    labels = np.asarray([r["ego_collision"] for r in rows], bool)
    risks = np.asarray([r["risk"] for r in rows])
    cells = cells_for(scenes)
    assert len(indices) == len(set(indices)) == BUDGET
    assert all(0 <= i < len(scenes) for i in indices)
    for i, observation in zip(indices, run["queries"]):
        assert observation["index"] == i and observation["scenario_id"] == scenes[i]["scenario_id"]
        assert observation["label"] == int(labels[i]) and observation["risk"] == risks[i]
    count, dangerous = int(labels.sum()), len(set(cells[labels]))
    assert count == 116 and dangerous == 33
    out = {"target_collision_count": count, "target_collision_cells": dangerous}
    archive = RiskArchive()
    for t, i in enumerate(indices, 1):
        archive.observe(int(cells[i]), float(risks[i]))
        if t in CHECKPOINTS:
            found = int(labels[indices[:t]].sum())
            found_cells = len(set(cells[indices[:t]][labels[indices[:t]]]))
            out.update({f"F{t}": found, f"query_collision_rate{t}": found / t,
                        f"recall{t}": found / count, f"cells{t}": found_cells,
                        f"cell_coverage{t}": found_cells / dangerous,
                        f"risk_qd_score{t}": archive.metrics()["risk_qd_score"]})
    out["repeat_collision_queries200"] = out["F200"] - out["cells200"]
    out["grid_sensitivity"] = {}
    for bins in (3, 4, 5):
        grid = cells_for(scenes, bins)
        truth = len(set(grid[labels]))
        found = len(set(grid[indices][labels[indices]]))
        out["grid_sensitivity"][str(bins)] = {"dangerous_cells": truth, "found_cells": found,
                                             "coverage200": found / truth}
    if "final_mean" in run:
        unqueried = np.ones(len(rows), bool)
        unqueried[indices] = False
        z = np.log(np.clip(risks, 1e-4, 1 - 1e-4)) - np.log1p(-np.clip(risks, 1e-4, 1 - 1e-4))
        out["prediction"] = prediction_metrics(np.asarray(run["final_mean"]),
                                               np.asarray(run["final_latent_var"]), z, labels, unqueried)
    return out


def csv_at(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    scenes, rows = response_rows("D")
    names = {**BASELINES, "SRD_TNP_BQD": "SRD-TNP-BQD"}
    output = ROOT / "comparison"
    output.mkdir(exist_ok=True)
    table, records = [], []
    for method, label in names.items():
        runs = [read_json(output / "runs" / f"{method}_seed_{seed}.json")
                for seed in ((11,) if method == "knn_history" else SEEDS)]
        if method == "SRD_TNP_BQD":
            assert all(run["implementation"] == IMPLEMENTATION for run in runs)
            assert all(run["frozen_modules"] and run["frozen_history_cache"] for run in runs)
            assert all(run["mean_readout"]["D_used"] is False for run in runs)
        values = [metrics(run, scenes, rows) for run in runs]
        records.extend({"method": method, "seed": run["seed"], **value} for run, value in zip(runs, values))
        report = {"method": label, "seeds": len(runs)}
        for key in ("F50", "F100", "F200", "recall200", "cell_coverage50", "cell_coverage100",
                    "cell_coverage200", "cells200", "risk_qd_score200"):
            samples = [v[key] for v in values]
            report[key] = float(np.mean(samples))
            report[key + "_std"] = float(np.std(samples, ddof=1)) if len(samples) > 1 else None
        table.append(report)
    csv_at(output / "comparison.csv", table)
    write_json(output / "per_seed.json", records)
    truth = {"target": TARGET, "candidate_count": 2048, "collisions": 116,
             "collision_rate": 116 / 2048, "collision_cells": 33,
             "F50_upper_bound": 50, "F100_upper_bound": 100, "F200_upper_bound": 116,
             "recall200_upper_bound": 1., "cell_coverage200_upper_bound": 1.}
    write_json(output / "ground_truth.json", truth)
    write_json(output / "summary.json", {"status": "COMPLETE", "setting": "original RAS-FRT-UQ S01 A/D",
                                        "target": TARGET, "implementation": IMPLEMENTATION,
                                        "methods": table,
                                        "historical_reference": "../reference/first_batch/comparison/summary.json",
                                        "target_and_scenarios_changed": False})
    from methods.ras_frt_uq.s01 import MODEL_ROOT
    original = read_json(MODEL_ROOT / "original_replay.json")["runs"]
    ras_equal = []
    for seed in SEEDS:
        current = read_json(output / "runs" / f"ras_frt_uq_seed_{seed}.json")
        reference = next(r for r in original if r["seed"] == seed)
        ras_equal.append(current["selected_indices"] == reference["selected_indices"])
    before = read_json(ROOT / "audit/restoration_before.json")
    after = protected_hashes()
    assert before == after, "frozen benchmark, measurements or paper weights changed"
    assert all(ras_equal), "original RAS trajectories changed"
    development = read_json(ROOT / "reference/development_history.json")["matched_history_development"]
    current = next(row for row in table if row["method"] == "SRD-TNP-BQD")
    for key, value in development["candidate"].items():
        np.testing.assert_allclose(current[key], value, rtol=0, atol=1e-12)
    write_json(output / "verification.json",
               {"status": "PASS", "original_RAS_5x200_queries_identical": ras_equal,
                "promoted_main_metrics_unchanged": True,
                "current_main_frozen_h_kernel_and_A_only_readout": True,
                "protected_files_byte_identical": True, "protected_file_count": len(after),
                "protected_reference": "audit/restoration_before.json (current storage layout)",
                "current_main_replay_queries": len(SEEDS) * BUDGET,
                "comparison_replay_queries": len(records) * BUDGET,
                "new_physical_executions": 0})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for row in table:
        axes[0].plot((50, 100, 200), [row[f"F{t}"] for t in (50, 100, 200)],
                     marker="o", label=row["method"])
    positions = np.arange(len(table))
    axes[1].barh(positions, [row["cell_coverage200"] * 100 for row in table])
    axes[1].set_yticks(positions, [row["method"] for row in table], fontsize=8)
    axes[1].invert_yaxis()
    axes[0].set(xlabel="Target queries", ylabel="Collisions discovered", title="Original S01: 116 failures / 2048")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set(xlabel="Failure-cell coverage at 200 queries (%)", xlim=(0, 105))
    for axis in axes:
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)
    lines = ["# 当前 SRD-TNP-BQD 与九个基线", "",
             "默认本文方法已收敛为匹配历史源、独立均值读出、联合均值校准、冻结 h／差异核和原混合采集。",
             "A/D 各 2048 个 S01 cut-in 场景；原五个 IDM 源与 fvdm_safety_speed_23_mps 目标保持不变。",
             "完整 D 真值为 116 次碰撞、33 个碰撞 cell。每条轨迹预算 200，五种子均值；kNN 为一次确定性运行。", "",
             "|方法|F50|F100|F200|碰撞召回@200|碰撞 cell 覆盖@200|风险 QDScore@200|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"|{row['method']}|{row['F50']:.2f}|{row['F100']:.2f}|{row['F200']:.2f}|"
                     f"{row['recall200']:.2%}|{row['cell_coverage200']:.2%}|{row['risk_qd_score200']:.2f}|")
    lines.extend(["", "新主线使用原 A 上补充的 2048 条匹配源响应；原九个基线仍使用原历史条件。",
                  "本文 F50 高于九个正式基线；F100 和最终碰撞召回仍略低于 RAS-FRT-UQ。",
                  "RAS-FRT-UQ 保留原二值碰撞反馈、模型和融合权重；五种子各 200 次选例与 Git 原记录完全一致。",
                  "本文和其余八个基线使用连续风险反馈，碰撞标签只用于评价；因此与 RAS 的比较不能单独证明连续反馈的贡献。",
                  "第一批结果与旧消融仅作为[历史参照](../reference/first_batch/README.md)保留，不混作当前主线的消融。",
                  "BAS 使用全候选池的预期 Bernoulli 方差下降，事件为 R>0.5。RF surrogate BO 使用 RF、十折 Jackknife 和 EI。",
                  "RF 参数预先固定；固定池枚举替代论文的连续 DE 搜索，未执行论文的 TPE 调参。两者均为明确披露的候选池适配。",
                  "原 D 曾参与 RAS 融合开发，当前成绩属于同库开发证据，不能解释为独立盲测结论。",
                  f"当前表复用 {len(records)} 条已完成轨迹，共 {len(records) * BUDGET} 次计费回放；此次整理没有新增物理执行或训练。",
                  "原源准备 10240 次、补充源 2048 次、D 真值 2048 次物理执行分别披露。新主线当前为固定池回放结果；旧在线复核属于第一批。",
                  "", "![比较曲线](comparison.png)", "",
                  "[逐种子指标](per_seed.json) · [当前核验](verification.json) · [历史开发摘要](../reference/development_history.json)",
                  "· [基线实现、出处与适配范围](../../../methods/srd_tnp_bqd/README.md)"])
    (output / "README.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print("Current S01 main chain, retained RAS and frozen-benchmark verification PASS")


if __name__ == "__main__":
    main()
