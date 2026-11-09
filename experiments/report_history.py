"""Describe the full executed historical RSS data without new simulations or training."""

import csv
import gzip
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from testing_methods.common.response import archive_fingerprints

from .joint_figures import STYLE, save_figure
from .prepare_joint import DATASET_ROOT
from .protocol import RUNS_ROOT
from .rss import measurement_contract, trajectory_margin
from .storage import load_archive, read_json, write_json

OUTPUT = RUNS_ROOT / "joint_response/reports/history_response"
TITLES = {
    "car_following": "Car following",
    "front_brake": "Front braking",
    "cut_out": "Lead cut-out",
    "lane_change_rear": "Ego lane change",
}
OUTCOMES = ((True, "Collision", "#D55E00", "-"), (False, "Safe pass", "#0072B2", "--"))


def executed_rows(archive):
    """Recompute initial/minimum margins from each archived record's actual trajectory."""
    batch_by_bank = {
        str(Path(item["bank"]).resolve()): item["batch"]
        for item in archive.provenance["contract"]["batches"]
    }
    rows = []
    for index, record_id in enumerate(archive.provenance["record_ids"]):
        evidence = archive.provenance["executions"][record_id]
        bank = Path(evidence["bank"]).resolve()
        with gzip.open(bank / evidence["execution"], "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        trace, feedback = payload["trajectory"], payload["feedback"]
        scene = archive.scenarios[index]
        if (
            archive.role != "history"
            or evidence["original_role"] != "history"
            or payload["call"] != evidence["call"]
            or payload["scenario"] != asdict(scene)
            or feedback["scenario_id"] != scene.scenario_id
            or evidence["scenario_id"] != scene.scenario_id
            or not trace
            or trace[0]["time_s"] != 0
            or type(feedback["collision"]) is not bool
            or feedback["collision"] != bool(archive.collision[index])
            or feedback["collision"] != any(frame["collision"] for frame in trace)
        ):
            raise ValueError(f"Historical response differs from its executed record: {record_id}")
        measured = trajectory_margin(trace)
        margin = measured["minimum_margin"]
        if not math.isclose(margin, archive.continuous_margin[index], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Archived RSS differs from actual trajectory: {record_id}")
        rows.append(
            {
                "scenario": scene.kind,
                "source_id": archive.source_id,
                "batch": batch_by_bank[str(bank)],
                "record_id": record_id,
                "scenario_id": scene.scenario_id,
                "collision": feedback["collision"],
                "rss_margin": margin,
                "initial_margin": measured["initial_margin"],
                "minimum_at_initial": math.isclose(
                    margin, measured["initial_margin"], rel_tol=0, abs_tol=1e-12
                ),
                "critical_time_s": measured["critical_time_s"],
                "critical_actor": measured["critical_actor"],
                "execution": str(bank / evidence["execution"]),
            }
        )
    if len(rows) != len(archive.scenarios):
        raise ValueError("Historical record IDs must cover the full archive")
    return rows


def describe(rows):
    count = len(rows)
    margins = np.array([r["rss_margin"] for r in rows], dtype=float)
    initial = sum(r["minimum_at_initial"] for r in rows)
    negative = int((margins < 0).sum())
    return {
        "records": count,
        "negative_margin_records": negative,
        "negative_margin_fraction": negative / count if count else None,
        "minimum_at_initial_records": initial,
        "minimum_at_initial_fraction": initial / count if count else None,
        "quantiles": dict(
            zip(
                ("min", "q05", "q25", "median", "q75", "q95", "max"),
                np.quantile(margins, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
                strict=True,
            )
        )
        if count
        else None,
    }


def grouped_statistics(rows, fields):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[name] for name in fields), []).append(row)
    return [
        {
            **dict(zip(fields, key, strict=True)),
            "all": describe(selected),
            "collision": describe([r for r in selected if r["collision"]]),
            "safe": describe([r for r in selected if not r["collision"]]),
        }
        for key, selected in groups.items()
    ]


@plt.rc_context(STYLE)
def figures(rows, root):
    kinds = list(dict.fromkeys(r["scenario"] for r in rows))
    figure, axes = plt.subplots(
        math.ceil(len(kinds) / 2), 2, figsize=(180 / 25.4, 132 / 25.4), squeeze=False
    )
    for panel, (axis, kind) in enumerate(zip(axes.flat, kinds)):
        for collision, label, color, style in OUTCOMES:
            values = np.sort(
                [
                    r["rss_margin"]
                    for r in rows
                    if r["scenario"] == kind and r["collision"] == collision
                ]
            )
            if len(values):
                axis.step(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    where="post",
                    color=color,
                    linestyle=style,
                    linewidth=1.3,
                    label=f"{label} (n={len(values):,})",
                )
        axis.axvline(0, color="#777777", linewidth=0.8, linestyle=":")
        axis.set(
            title=f"({chr(97 + panel)}) {TITLES[kind]}", xlabel="Minimum RSS margin", ylabel="ECDF"
        )
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(
            loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=8
        )
    for axis in list(axes.flat)[len(kinds) :]:
        axis.set_visible(False)
    figure.tight_layout(pad=1.2)
    distribution = root / "rss_distributions.png"
    save_figure(figure, distribution)

    stats = grouped_statistics(rows, ("scenario",))
    figure, axis = plt.subplots(figsize=(180 / 25.4, 85 / 25.4))
    positions = np.arange(len(stats))
    for offset, (_, label, color, _), key in zip(
        (-0.18, 0.18), OUTCOMES, ("collision", "safe"), strict=True
    ):
        values = [s[key]["minimum_at_initial_fraction"] for s in stats]
        bars = axis.bar(
            positions + offset,
            [100 * value if value is not None else np.nan for value in values],
            width=0.34,
            color=color,
            label=label,
        )
        axis.bar_label(
            bars,
            labels=[f"{100 * v:.1f}%" if v is not None else "" for v in values],
            fontsize=8,
            padding=3,
        )
    axis.set_xticks(positions, [TITLES[s["scenario"]] for s in stats])
    maximum = max(
        100 * item[key]["minimum_at_initial_fraction"]
        for item in stats
        for key in ("collision", "safe")
        if item[key]["minimum_at_initial_fraction"] is not None
    )
    axis.set(
        ylabel="Minimum attained at t=0 (%)", ylim=(0, max(5, 5 * math.ceil(maximum * 1.2 / 5)))
    )
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2)
    figure.tight_layout(pad=1.2)
    initial = root / "initial_minimum.png"
    save_figure(figure, initial)
    return [distribution.name, initial.name]


def report(manifest_path=DATASET_ROOT / "manifest.json", output=OUTPUT):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = read_json(manifest_path)
    if manifest["rss_contract"] != measurement_contract():
        raise ValueError("Historical diagnostic uses another RSS definition")
    rows, fingerprints = [], {}
    for entry in manifest["archives"]:
        archive = load_archive(manifest_path.parent / entry["path"])
        fingerprint = archive_fingerprints([archive])[archive.source_id]
        if (
            fingerprint != entry["fingerprint"]
            or len(archive.scenarios) != entry["records"]
            or archive.source_id != entry["source_id"]
            or archive.role != entry["role"]
            or any(scene.kind != entry["scenario"] for scene in archive.scenarios)
        ):
            raise ValueError("Historical diagnostic archive differs from its manifest")
        rows.extend(executed_rows(archive))
        fingerprints[entry["path"]] = fingerprint
        print(f"historical RSS: {len(rows)} verified records", flush=True)
    if len(rows) != manifest["total_records"] or len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Historical diagnostic requires all unique executed records")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "records.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "records": len(rows),
        "archive_fingerprints": fingerprints,
        "rss_contract": measurement_contract(),
        "minimum_at_initial": "initial equals minimum over all actual recorded times; includes later ties; absolute tolerance 1e-12",
        "quantiles": "empirical sample quantiles with numpy linear interpolation",
        "scenarios": grouped_statistics(rows, ("scenario",)),
        "sources": grouped_statistics(rows, ("scenario", "source_id")),
        "batches": grouped_statistics(rows, ("scenario", "batch")),
        "new_simulations": 0,
        "method_queries": 0,
        "figures": figures(rows, output),
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# 全量历史 RSS 诊断",
        "",
        f"逐条读取 {len(rows):,} 条已执行历史轨迹，核对碰撞标签与已归档 RSS；不新增仿真、方法查询或训练。",
        "",
        "| 场景 | 记录数 | 碰撞数 | 安全通过负裕度 | 最小值已在初始时刻达到 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["scenarios"]:
        all_rows, safe = item["all"], item["safe"]
        lines.append(
            f"| {item['scenario']} | {all_rows['records']} | {item['collision']['records']} | "
            f"{safe['negative_margin_records']}／{safe['records']} | "
            f"{all_rows['minimum_at_initial_records']}／{all_rows['records']} "
            f"（{100 * all_rows['minimum_at_initial_fraction']:.2f}%） |"
        )
    lines.extend(
        [
            "",
            "初始最小值允许后续时刻并列达到相同值，不等于本次执行没有后续风险变化。碰撞与安全通过的分布分别归一化为经验累积分布，不按类别样本数缩放，也不使用 RSS 符号替代碰撞。",
            "",
            "[逐记录数据](records.csv)、[分场景／来源／批次统计与分位数](summary.json)、[碰撞与安全通过的 RSS 分布](rss_distributions.png)、[初始最小裕度比例](initial_minimum.png)。",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    report()
