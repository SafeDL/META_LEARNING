"""Figures and one full-rate paired replay from measured FBRT episodes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from highway_env_benchmark.envs.fbrt_env import run_episode
from highway_env_benchmark.envs.fbrt_scenarios import FBRTScenario
from method_chains.core_mine.idm_revision_pilot import REFERENCE
from method_chains.failure_memory_regression.experiment import BUDGET, CHECKPOINTS, METHODS, ROOT

OUTPUT = ROOT / "core"


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_results() -> dict:
    output = OUTPUT
    reference = {row["scenario_id"]: row for row in rows(output / "reference_archive.csv")}
    candidates = {row["scenario_id"] for row in rows(output / "candidate_pool.csv")}
    target = {(row["build"], row["scenario_id"]): row
              for row in rows(output / "target_response_bank.csv")}
    campaigns: dict[tuple[str, str, str, str], list[str]] = {}
    queries = rows(output / "queries.csv")
    for row in queries:
        key = (row["seed"], row["build"], row["method"], row["repeat"])
        campaigns.setdefault(key, []).append(row["scenario_id"])
        scenario_id = row["scenario_id"]
        assert scenario_id in candidates
        assert reference[scenario_id]["reference_pass"] == "True"
        measured = target[(row["build"], scenario_id)]["ego_collision"] == "True"
        assert measured == (row["ego_collision"] == "True")
        assert measured == (row["regression"] == "True")
    for selected in campaigns.values():
        assert len(selected) == BUDGET
        assert len(set(selected)) == len(selected)
    payload = {"candidate_count": len(candidates), "campaign_count": len(campaigns),
               "logical_query_count": len(queries), "target_bank_count": len(target),
               "all_queries_revealed_only_eligible_measured_outcomes": True,
               "all_campaigns_used_distinct_scenarios_within_budget": True}
    (output / "validation.json").write_text(json.dumps(payload, indent=2) + "\n",
                                             encoding="utf-8")
    return payload


def plot_three_vehicle(trace: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for name in ("ego", "lead", "static"):
        axes[0].plot([row["time_s"] for row in trace],
                     [row[name]["x_m"] for row in trace], label=name)
        axes[1].plot([row["time_s"] for row in trace],
                     [row[name]["y_m"] for row in trace], label=name)
    axes[0].set_ylabel("Longitudinal position (m)")
    axes[1].set(xlabel="Time (s)", ylabel="Lateral position (m)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_figures() -> None:
    output = OUTPUT
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    archive = rows(output / "reference_archive.csv")
    candidates = rows(output / "candidate_pool.csv")
    boundaries = rows(output / "boundary_memory.csv")
    queries = rows(output / "queries.csv")
    first_seed = min(int(row["seed"]) for row in archive)
    for template in {row["template_id"] for row in candidates}:
        subset = [row for row in archive if int(row["seed"]) == first_seed and
                  json.loads(row["scenario"])["template_id"] == template]
        fig, ax = plt.subplots(figsize=(5, 4))
        for label, color in (("True", "#b13a3a"), ("False", "#2c7a50")):
            samples = [json.loads(row["scenario"]) for row in subset
                       if row["ego_collision"] == label]
            if samples:
                second = {"fbrt_cutin": "lane_change_duration_s",
                          "fbrt_lead_emergency_brake": "lead_deceleration_mps2",
                          "fbrt_stop_hold_go": "lead_deceleration_mps2",
                          "fbrt_cutout_static": "static_target_ttc_s"}[template]
                ax.scatter([item["initial_clearance_m"] for item in samples],
                           [item[second] for item in samples], c=color,
                           label="reference collision" if label == "True" else "reference pass")
        by_id = {row["scenario_id"]: json.loads(row["scenario"]) for row in subset}
        for patch in boundaries:
            if int(patch["seed"]) != first_seed or patch["template_id"] != template:
                continue
            failed = by_id[patch["failed_id"]]
            passed = by_id[patch["passed_id"]]
            xs = [failed["initial_clearance_m"], passed["initial_clearance_m"]]
            ys = [failed[second], passed[second]]
            ax.plot(xs, ys, color="#555555", alpha=0.4, linewidth=1)
            ax.scatter([sum(xs) / 2], [sum(ys) / 2], marker="x",
                       c="#222222", s=25)
        ax.set_xlabel("Initial bumper clearance (m)")
        ax.set_ylabel(second)
        ax.set_title(template)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{template}_reference.png", dpi=160)
        plt.close(fig)
    for build in sorted({row["build"] for row in queries}):
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in ("FBRT-Static", "FBRT-Adaptive", "FBRT-RegionBandit",
                       "HistoryRank-UCB", "HistoryMargin", "FailureDistance",
                       "ART-Maximin", "Random"):
            subset = [row for row in queries if row["build"] == build and
                      row["method"] == method]
            campaigns = {}
            for row in subset:
                campaigns.setdefault((row["seed"], row["repeat"]), []).append(row)
            detected = []
            for rank in range(1, BUDGET + 1):
                detected.append(sum(any(entry["regression"] == "True" and
                                        int(entry["rank"]) <= rank for entry in campaign)
                                    for campaign in campaigns.values()) / max(len(campaigns), 1))
            ax.plot(range(1, BUDGET + 1), detected, label=method)
        ax.set(xlabel="Target queries", ylabel="Fraction detecting a regression",
               ylim=(-0.03, 1.03), title=build)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{build}_detection.png", dpi=160)
        plt.close(fig)


def make_replay(reuse_existing: bool = False) -> dict | None:
    output = OUTPUT
    existing_path = output / "replay" / "paired_trace.json"
    if reuse_existing and existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing["scene"]["template_id"] == "fbrt_cutout_static":
            plot_three_vehicle(existing["reference_trace"],
                               existing_path.parent / "three_vehicle_trajectory.png")
        return {"scenario_id": existing["scene"]["scenario_id"],
                "build": existing["build"],
                "reference_collision": existing["reference"]["ego_collision"],
                "target_collision": existing["target"]["ego_collision"],
                "collision_time_s": existing["target"]["collision_time_s"],
                "collision_partner": existing["target"]["collision_partner"],
                "frames": max(len(existing["reference_trace"]),
                              len(existing["target_trace"]))}
    queries = rows(output / "queries.csv")
    matches = [row for row in queries if row["method"] == "FBRT-Static" and
               row["regression"] == "True" and row["repeat"] == "0"]
    if not matches:
        return None
    chosen = min(matches, key=lambda row: (int(row["seed"]), int(row["rank"])))
    candidate = next(row for row in rows(output / "candidate_pool.csv")
                     if row["scenario_id"] == chosen["scenario_id"])
    scene = FBRTScenario(candidate["scenario_id"], candidate["template_id"],
                         float(candidate["initial_clearance_m"]),
                         lane_change_duration_s=float(candidate["lane_change_duration_s"])
                         if candidate["lane_change_duration_s"] else None,
                         lead_deceleration_mps2=float(candidate["lead_deceleration_mps2"])
                         if candidate["lead_deceleration_mps2"] else None,
                         static_target_ttc_s=float(candidate["static_target_ttc_s"])
                         if candidate["static_target_ttc_s"] else None)
    seed = int(chosen["seed"])
    reference, ref_trace = run_episode(REFERENCE, scene, seed, with_trace=True)
    target, target_trace = run_episode(REFERENCE, scene, seed,
                                       fault=chosen["build"], with_trace=True)
    replay_dir = output / "replay"
    replay_dir.mkdir(exist_ok=True)
    payload = {"scene": scene.as_record(), "build": chosen["build"],
               "seed": seed, "reference": reference, "target": target,
               "reference_trace": ref_trace, "target_trace": target_trace}
    (replay_dir / "paired_trace.json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    times = [row["time_s"] for row in ref_trace]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, trace in (("reference", ref_trace), (chosen["build"], target_trace)):
        ax.plot([row["time_s"] for row in trace],
                [row["ego"]["speed_mps"] for row in trace], label=f"{name} ego")
    if scene.template_id == "fbrt_stop_hold_go":
        ax.plot(times, [row["lead"]["speed_mps"] for row in ref_trace],
                label="lead", linestyle="--")
    ax.set(xlabel="Time (s)", ylabel="Speed (m/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(replay_dir / "speed.png", dpi=160)
    plt.close(fig)
    if scene.template_id == "fbrt_cutout_static":
        plot_three_vehicle(ref_trace, replay_dir / "three_vehicle_trajectory.png")
    frames = []
    for index in range(max(len(ref_trace), len(target_trace))):
        canvas = Image.new("RGB", (900, 320), "white")
        draw = ImageDraw.Draw(canvas)
        for panel, (label, trace) in enumerate((("reference", ref_trace),
                                                (chosen["build"], target_trace))):
            offset = panel * 450
            draw.text((offset + 12, 8), label, fill="black")
            draw.rectangle((offset + 5, 35, offset + 440, 145),
                           fill="#dddddd", outline="#777777")
            draw.line((offset + 5, 90, offset + 440, 90), fill="white", width=3)
            row = trace[min(index, len(trace) - 1)]
            focus_x = row["ego"]["x_m"]
            for name, state in row.items():
                if name == "time_s":
                    continue
                x = offset + 80 + (state["x_m"] - focus_x) * 4
                y = 122 - state["y_m"] * 16
                color = "#d63b32" if state["crashed"] else (
                    "#2163aa" if name == "ego" else "#e69f31" if name == "lead" else "#555555")
                draw.rectangle((x - 9, y - 6, x + 9, y + 6), fill=color)
                draw.text((x - 9, y + 8), name, fill="black")
            draw.text((offset + 12, 165), f"t={row['time_s']:.2f}s", fill="black")
            draw.text((offset + 12, 190),
                      f"ego speed={row['ego']['speed_mps']:.1f} m/s", fill="black")
            draw.text((offset + 12, 215),
                      "ego collision" if row["ego"]["crashed"] else "no ego collision",
                      fill="#b13a3a" if row["ego"]["crashed"] else "#2c7a50")
        frames.append(canvas)
    frames[0].save(replay_dir / "paired.gif", save_all=True,
                   append_images=frames[1:], duration=50, loop=0, optimize=True)
    return {"scenario_id": scene.scenario_id, "build": chosen["build"],
            "reference_collision": reference["ego_collision"],
            "target_collision": target["ego_collision"],
            "collision_time_s": target["collision_time_s"],
            "collision_partner": target["collision_partner"],
            "frames": len(frames)}


def make_stop_plot(reuse_existing: bool) -> bool:
    output = OUTPUT
    trace_path = output / "replay" / "stop_hold_go_trace.json"
    if reuse_existing and trace_path.exists():
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    else:
        candidate = next(row for row in rows(output / "candidate_pool.csv")
                         if row["template_id"] == "fbrt_stop_hold_go")
        scene = FBRTScenario(candidate["scenario_id"], candidate["template_id"],
                             float(candidate["initial_clearance_m"]),
                             lead_deceleration_mps2=float(candidate["lead_deceleration_mps2"]))
        outcome, trace = run_episode(REFERENCE, scene,
                                     int(scene.scenario_id.split(":", 1)[0]), with_trace=True)
        payload = {"scene": scene.as_record(), "outcome": outcome, "trace": trace}
        trace_path.parent.mkdir(exist_ok=True)
        trace_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    trace = payload["trace"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([row["time_s"] for row in trace],
            [row["lead"]["speed_mps"] for row in trace], label="lead")
    ax.plot([row["time_s"] for row in trace],
            [row["ego"]["speed_mps"] for row in trace], label="ego")
    ax.set(xlabel="Time (s)", ylabel="Speed (m/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(trace_path.parent / "stop_hold_go_speed.png", dpi=160)
    plt.close(fig)
    return True


def complete_report(replay: dict | None, stop_plot: bool) -> None:
    output = OUTPUT
    summary = rows(output / "summary_by_revision.csv")
    templates = rows(output / "summary_by_template.csv")
    boundaries = rows(output / "boundary_memory.csv")
    queries = rows(output / "queries.csv")
    ledger_path = output / "compute_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["replay_episodes"] = (2 if replay else 0) + int(stop_plot)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    task_count = len({(row["seed"], row["build"]) for row in summary})
    candidate_count = len(rows(output / "candidate_pool.csv"))
    lines = ["", "## 原型做通了什么", "",
             f"四类功能场景均完成参考版与三个局部修改版的实际执行；"
             f"共同候选池有 {candidate_count} 个参考通过场景。"
             f"历史失败和邻近通过样本形成 {len(boundaries)} 个局部配对，"
             "这些配对进入 FBRT 的选例分数。目标结果只在每次逻辑查询时揭示。",
             "", "## 跨种子预算结果", "",
             f"每个受测版本和种子构成一项检测任务，共 {task_count} 项。随机方法每项重放 20 次，表中给出平均检测任务数；其他方法每项运行一次。",
             "", f"| 方法 | @5 检测/{task_count} | @10 检测/{task_count} | @20 检测/{task_count} | @50 检测/{task_count} | @50 平均回归碰撞数 | @10 平均回归功能数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    detection_at_five = {}
    for method in METHODS:
        cells = []
        for budget in CHECKPOINTS:
            subset = [row for row in summary if row["method"] == method and
                      int(row["budget"]) == budget]
            repeats = 20 if method == "Random" else 1
            detections = sum(row["regression_detected"] == "True" for row in subset) / repeats
            cells.append(f"{detections:.2f}")
            if budget == 5:
                detection_at_five[method] = detections
            if budget == 10:
                functions = sum(int(row["regression_template_count"])
                                for row in subset) / len(subset)
            if budget == BUDGET:
                collisions = sum(int(row["regression_collision_count"])
                                 for row in subset) / len(subset)
        lines.append(f"| {method} | {' | '.join(cells)} | {collisions:.2f} | {functions:.2f} |")
    lines.extend(["", f"@5 中，FBRT-Static 检出 {detection_at_five['FBRT-Static']:.2f}/{task_count}，"
                  f"FBRT-Adaptive 检出 {detection_at_five['FBRT-Adaptive']:.2f}/{task_count}，"
                  f"HistoryMargin 和 FailureDistance 分别为 "
                  f"{detection_at_five['HistoryMargin']:.2f}/{task_count} 与 "
                  f"{detection_at_five['FailureDistance']:.2f}/{task_count}。"
                  "这说明原型可以在少量新版本查询中发现真实回归；"
                  "简单历史排序在当前场景中同样有效。",
                  "@10 的功能数是发现回归碰撞的不同功能场景数，不等于独立软件缺陷数。"
                  "FBRT-RegionBandit 把历史失效边界的局部排序与新版本反馈的跨功能分配结合；"
                  "它在本轮开发结果库中的平均碰撞数略高于 HistoryRank-UCB，"
                  "但该差异来自同一结果库上的方法开发，不能当作独立验证或统计优势。"])
    first_hit = next((row for row in queries
                      if row["method"] == "FBRT-Static" and
                      row["selection_reason"] == "boundary" and
                      row["regression"] == "True" and row["patch_id"]), None)
    if first_hit is not None:
        patch = next(row for row in boundaries
                     if row["seed"] == first_hit["seed"] and
                     row["patch_id"] == first_hit["patch_id"])
        lines.extend(["", "## 一条可追溯的选例链", "",
                      f"参考版实测失败场景 `{patch['failed_id']}` 与附近实测通过场景 "
                      f"`{patch['passed_id']}` 构成局部配对；FBRT-Static 根据这段历史，"
                      f"在第 {first_hit['rank']} 次查询选中 `{first_hit['scenario_id']}`。"
                      f"同一场景参考版通过，`{first_hit['build']}` 实测发生 ego 碰撞。"])
    lines.extend(["", "## 按功能观察", "",
                  f"下表把 {len({row['seed'] for row in templates})} 个种子的参考结果与三个目标修改的回归碰撞合计；参考结果只计一次，不因目标版本重复。",
                  "", "| 功能 | 参考通过 | 参考碰撞 | 目标回归碰撞（3 版本合计） | 局部边界对 |",
                  "|---|---:|---:|---:|---:|"])
    for template in sorted({row["template_id"] for row in templates}):
        subset = [row for row in templates if row["template_id"] == template]
        first_by_seed = {row["seed"]: row for row in subset}
        passes = sum(int(row["reference_passes"]) for row in first_by_seed.values())
        failures = sum(int(row["reference_failures"]) for row in first_by_seed.values())
        regressions = sum(int(row["target_regressions"]) for row in subset)
        patch_count = sum(row["template_id"] == template for row in boundaries)
        lines.append(f"| {template} | {passes} | {failures} | {regressions} | {patch_count} |")
    lines.extend(["", "停车—保持—起步若无参考碰撞，就按已记录的历史裕度回退；其目标回归仍属于真实执行结果。",
                  "局部修改按车辆状态触发，报告不把每个场景随机种子解释为独立软件缺陷。",
                  "", "## 物理成本与回放", "",
                  f"本配置使用参考 {ledger['physical_reference_episodes_used']} 次、目标 "
                  f"{ledger['physical_target_episodes_used']} 次实测 episode；"
                  + ("本次从保存的结果库离线重放，新增仿真 0 次。" if ledger.get(
                      "offline_replay_from_measured_bank") else
                     f"本次调用新跑参考 {ledger['new_reference_episodes']} 次、目标 "
                     f"{ledger['new_target_episodes']} 次。")
                  + "逐方法逻辑查询见 `compute_ledger.json`。"])
    if replay:
        lines.append(f"配对回放：`{replay['scenario_id']}`；参考无碰撞，"
                     f"`{replay['build']}` 于 {replay['collision_time_s']} s 与 "
                     f"`{replay['collision_partner']}` 碰撞。`replay/paired.gif` 每 0.05 s 一帧，"
                     "`replay/paired_trace.json` 保留全部状态。")
    if stop_plot:
        lines.append("停车—保持—起步的参考版速度曲线在 `replay/stop_hold_go_speed.png`，"
                     "对应 20 Hz 轨迹在 `replay/stop_hold_go_trace.json`。")
    lines.extend(["", "当前结论对应本轮受控 IDM 修改和 highway-env 功能场景。", ""])
    report_path = output / "report.md"
    existing = report_path.read_text(encoding="utf-8")
    for marker in ("\n## 原型做通了什么", "\n## 跨种子预算结果"):
        if marker in existing:
            existing = existing.split(marker, 1)[0]
            break
    report_path.write_text(existing.rstrip() + "\n" + "\n".join(lines),
                           encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-replay", action="store_true")
    args = parser.parse_args()
    validate_results()
    make_figures()
    replay = make_replay(reuse_existing=args.reuse_replay)
    stop_plot = make_stop_plot(reuse_existing=args.reuse_replay)
    complete_report(replay, stop_plot)
    print(json.dumps(replay, ensure_ascii=False))


if __name__ == "__main__":
    main()
