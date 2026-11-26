"""Animate registered, avoidable collision representatives without simulation."""

import math
from itertools import combinations

import numpy as np

from simulator.highway.observation import VehicleState, observe

from experiments.scene_design.audit_interactions import check_dynamics
from experiments.execution_events import analyze_events
from experiments.protocol import ROOT
from experiments.plotting.replay_safe import SCENES, render_case, render_contact_sheets, road_bounds
from experiments.schema import Scenario
from experiments.storage import read_json, write_json

DESTINATION = ROOT / "figures/scene_replays/collision"
RESULTS = ROOT / "results/scene_replays/collision"
FRAME_LABELS = ("Initial", "Approach / reveal", "Before collision", "First collision")


def select_cases():
    cases = []
    for directory in SCENES:
        summary = read_json(
            directory
            / (
                "secondary_summary.json"
                if directory.parent.name == "cut_out_rear_gap"
                else "summary.json"
            )
        )
        for row in summary["rows"]:
            secondary = None
            if "representative_collision" in row:
                ordinal = int(row["representative_collision"].rsplit(":", 1)[-1])
                assert row["representative_collision_avoidable"]
                reference = directory / "joint_reference" / f"{ordinal:03d}.json.gz"
                mechanism = "lead_braking"
            else:
                representative = row["avoidable_representative"]
                ordinal = (
                    representative["ordinal"]
                    if "ordinal" in representative
                    else int(representative["scenario_id"].rsplit(":", 1)[-1])
                )
                reference = (
                    directory
                    / "reference"
                    / f"{ordinal:03d}_{representative['mode']}.json.gz"
                )
                mechanism = representative.get("mechanism", "direct_interaction")
                if mechanism == "verified_secondary":
                    secondary = next(
                        c for c in row["secondary_checks"] if c["ordinal"] == ordinal
                    )
                    assert secondary["confirmed"] and secondary["counterfactual_safe"]
            path = directory / row["sut_id"] / f"{ordinal:03d}.json.gz"
            payload, safe_reference = read_json(path), read_json(reference)
            assert payload["feedback"]["collision"]
            assert safe_reference["scenario"] == payload["scenario"]
            assert not safe_reference["summary"]["collision"]
            assert not any(f["collision"] for f in safe_reference["trajectory"])
            assert not safe_reference["summary"]["violations"]
            assert safe_reference["trajectory"][-1]["time_s"] == (
                14
                if payload["scenario"]["kind"] == "car_following_braking_onset"
                else 10
            )
            cases.append(
                (
                    row["sut_id"],
                    path,
                    {
                        "mechanism": mechanism,
                        "reference": reference.relative_to(ROOT).as_posix(),
                        "secondary_check": secondary,
                    },
                )
            )
    assert len(cases) == 64
    return cases


def inspect_collision(payload):
    trace = payload["trajectory"]
    lower, upper = road_bounds(payload["scenario"])
    crash = next(i for i, frame in enumerate(trace) if frame["collision"])
    partner = trace[crash]["collision_partner"]
    assert payload["feedback"]["collision"] and not trace[0]["collision"]
    assert trace[crash]["time_s"] == payload["summary"]["collision_time_s"]
    assert partner == payload["summary"]["collision_partner"]
    issues, clearance, road, background_road, overlap_times = [], [], [], [], []
    for index, frame in enumerate(trace):
        assert all(
            math.isfinite(a[k])
            for a in frame["actors"]
            for k in ("x", "y", "speed", "heading", "length", "width")
        )
        assert all(a["speed"] >= -1e-8 for a in frame["actors"])
        assert frame["collision"] == (index >= crash)
        assert frame["collision_partner"] == (partner if index >= crash else None)
        actors = [VehicleState(**a) for a in frame["actors"]]
        bodies = {a.actor_id: a.polygon() for a in actors}
        ego = bodies["ego"]
        others = [b for name, b in bodies.items() if name != "ego"]
        assert partner in bodies
        clearance.append(min(ego.distance(b) for b in others))
        road.append(min(ego.bounds[1] - lower, upper - ego.bounds[3]))
        background_road.append(
            min(min(b.bounds[1] - lower, upper - b.bounds[3]) for b in others)
        )
        if any(ego.intersects(b) for b in others):
            overlap_times.append(frame["time_s"])
        if index == 0 and any(
            a.intersects(b) for a, b in combinations(bodies.values(), 2)
        ):
            issues.append("initial_actor_overlap")
        if index < crash and any(a.intersects(b) for a, b in combinations(others, 2)):
            issues.append("background_overlap_before_collision")
        if {a.actor_id for a in observe(frame["time_s"], actors).vehicles} != set(
            frame["visible_ids"]
        ):
            issues.append("visible_actor_set_mismatch")
    # Native impact resolution can alter poses and actions after a crash. The
    # ordinary bicycle/control equations are checked only before that transition.
    check_dynamics(payload)
    if min(road[:crash]) < -1e-8:
        issues.append("ego_outside_road_before_collision")
    if min(background_road[:crash]) < -1e-8:
        issues.append("background_outside_road_before_collision")
    events = analyze_events(Scenario(**payload["scenario"]), trace)
    onset = events["measured_background_onset_s"]
    if onset is None or onset > trace[crash]["time_s"]:
        issues.append("collision_before_scripted_event")
    reveal = next(
        (i for i, f in enumerate(trace) if "obstacle" in f["visible_ids"]), None
    )
    if payload["scenario"]["kind"] == "cut_out_rear_gap":
        if reveal is None or not 0 < trace[reveal]["time_s"] < trace[crash]["time_s"]:
            issues.append("collision_before_obstacle_reveal")
    event = reveal if reveal is not None else min(crash, max(1, crash // 2))
    ego_states = [next(a for a in f["actors"] if a["actor_id"] == "ego") for f in trace]
    return {
        "passed": not issues,
        "issues": sorted(set(issues)),
        "physics_frames": len(trace),
        "precollision_integration_steps_checked": crash - 1,
        "duration_s": trace[-1]["time_s"],
        "collision_time_s": trace[crash]["time_s"],
        "collision_partner": partner,
        "clearance_at_native_collision_m": clearance[crash],
        "first_geometric_overlap_s": overlap_times[0] if overlap_times else None,
        "minimum_body_clearance_m": min(clearance),
        "minimum_road_margin_m": min(road),
        "minimum_precollision_road_margin_m": min(road[:crash]),
        "minimum_background_road_margin_m": min(background_road),
        "closest_time_s": trace[int(np.argmin(clearance))]["time_s"],
        "obstacle_first_visible_s": (
            trace[reveal]["time_s"] if reveal is not None else None
        ),
        "ego_speed_initial_final_mps": [
            ego_states[0]["speed"],
            ego_states[-1]["speed"],
        ],
        "ego_lateral_range_m": [
            min(a["y"] for a in ego_states),
            max(a["y"] for a in ego_states),
        ],
        "ego_final_y_m": ego_states[-1]["y"],
        "events": events,
        "keyframe_indices": [0, event, crash - 1, crash],
        "clearance_series_m": clearance,
    }


def write_index(results):
    lines = [
        "# 危险场景碰撞 GIF 回放",
        "",
        "四类当前开发场景 × 八类 ADS × 目标与历史配置，共 64 个 GIF。采用结果表中已登记、具有同坐标合法避撞参考的危险代表记录；不是随机抽样，也不据此估计碰撞率。逐例来源、参数、参考路径和检查值见 [summary.json](summary.json)。",
        "",
        "全部动画按原始 20 Hz 执行轨迹重建，每帧 50 ms，末帧停留 500 ms；保留原生碰撞后的所有已记录帧。红框及 COLLISION 表示原生碰撞事件，碰撞前显示 Before collision，并标明实际碰撞对象。原生预测冲击与当前车身几何重叠分别记录，不能因为画面中存在微小间距而修改碰撞标签。下方总览使用独立纵向比例，主视图横纵等比例；星号和空心车身表示当前未被 ADS 观测到。",
        "",
        "本次仅读取既有档案，没有新增仿真、训练或正式方法评估。切出场景的 UCT、Beam 使用已有移除障碍对照确认的次生碰撞，不能解释为直接撞障碍。",
        "",
        "## GIF 索引",
        "",
        "每格 T 为目标配置，H 为历史配置，数字为原始记录编号。",
        "",
        "| ADS | 跟车制动 | 受限切入 | 切出露障碍 | 侧向侵入 |",
        "|---|---|---|---|---|",
    ]
    for family in (
        "idm_mobil",
        "fvdm_lane",
        "linear_mpc",
        "nonlinear_trajectory",
        "uct",
        "beam",
        "ppo",
        "dqn",
    ):
        cells = [family]
        for kind in dict.fromkeys(r["scenario"]["kind"] for r in results):
            matches = [
                r
                for r in results
                if r["scenario"]["kind"] == kind
                and r["sut_id"].split("_", 1)[1].removesuffix("_native") == family
            ]
            cells.append(
                " / ".join(
                    f"[{'T' if r['sut_id'].startswith('target_') else 'H'} {r['scenario']['scenario_id'].rsplit(':', 1)[-1]}](../../../{r['gif']})"
                    for r in matches
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 复现",
        "",
        "```powershell",
        "python -B -m experiments.plotting.replay_collision",
        "```",
        "",
    ]
    (RESULTS / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    results = []
    for sut_id, path, evidence in select_cases():
        result = render_case(
            sut_id, path, inspect_record=inspect_collision, destination=DESTINATION
        )
        result.update(evidence)
        results.append(result)
        check = result["inspection"]
        print(
            f"{len(results):02d}/64 {result['scenario']['kind']} {sut_id} {path.stem}: "
            f"{check['collision_time_s']} s -> {check['collision_partner']}; issues={check['issues']}",
            flush=True,
        )
    sheets = render_contact_sheets(results, DESTINATION, FRAME_LABELS)
    write_json(
        RESULTS / "summary.json",
        {
            "selection": "registered collision representative with an existing same-coordinate legal avoidance reference; no random sampling",
            "new_simulator_executions": 0,
            "cases": results,
            "contact_sheets": sheets,
            "all_passed": all(r["inspection"]["passed"] for r in results),
            "physics_frames": sum(r["inspection"]["physics_frames"] for r in results),
            "gif_frame_pixels_and_timestamps_verified": all(
                r["gif_pixels_and_timestamps_verified"] for r in results
            ),
        },
    )
    write_index(results)


if __name__ == "__main__":
    main()
