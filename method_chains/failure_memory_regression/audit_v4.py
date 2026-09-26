"""Auditable v4 execution, geometry checks, and trajectory GIFs.

The v2 scenario bank is frozen. New physical calls use a new contract and an
isolated audit episode file while sharing the global physical-call ledger.
"""

from __future__ import annotations

import json
import math
import csv
from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from method_chains.failure_memory_regression.archive_v2 import read_jsonl
from method_chains.failure_memory_regression.experiment_v2 import (
    GLOBAL_PHYSICAL_CAP, ROOT, _ensure_physical_episode, _load_ledger,
)


CONFIG = Path("configs/fbrt/ego_initial_state_audit_v3.yaml")
OUT = ROOT / "validation_audit_v4"
BUILD_MOBIL = "mobil_ref_v2"
BUILD_MUTANT = "mobil_rear_guard_off_v2"
BUILD_PPO = "ppo_ref_v2"
BUILD_DELAY = "ppo_obs_age020_v2"
BUILD_VI = "vi_ttc_ref_audit_v4"
BUILD_MCTS = "mcts_cv_ref_audit_v4"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_frozen_bank(cases: list[dict], episodes: list[dict]) -> dict:
    errors = []
    by_id = {case["scenario_id"]: case for case in cases}
    if len(cases) != 80 or len(by_id) != 80:
        errors.append("frozen_bank_case_count_or_uniqueness")
    if len(episodes) != 320:
        errors.append("frozen_bank_episode_count")
    for case in cases:
        a, c = case["active_parameters"], case["fixed_context"]
        for key, value in a.items():
            lower, upper = case["research_bounds"][key]
            if not lower <= value <= upper:
                errors.append(f"{case['scenario_id']}:out_of_bounds:{key}")
        if c.get("lane_count") != 2 or not 0 < c["ego_speed_mps"] <= 35:
            errors.append(f"{case['scenario_id']}:road_or_ego_speed")
        template = case["template_id"]
        if "initial_clearance_m" in a and a["initial_clearance_m"] <= 0:
            errors.append(f"{case['scenario_id']}:initial_lead_overlap")
        if template == "fbrt_lane_change_rear":
            if a["rear_clearance_m"] <= 0 or c["lead_clearance_m"] <= 0:
                errors.append(f"{case['scenario_id']}:initial_traffic_overlap")
            if c["ego_speed_mps"] + a["rear_closing_speed_mps"] > 35:
                errors.append(f"{case['scenario_id']}:rear_speed_above_research_bound")
        if template == "fbrt_cutout_static":
            # At event onset, the lead-to-static bumper gap is the specified TTC
            # times the lead speed. This is a geometric construction, not a crash claim.
            gap = a["lead_to_static_ttc_start_s"] * c["lead_speed_mps"]
            if not 40 <= gap <= 90:
                errors.append(f"{case['scenario_id']}:static_gap")
    unique = {(row["scenario_id"], row["build_id"]) for row in episodes}
    if len(unique) != len(episodes):
        errors.append("duplicate_build_case_execution")
    if {row["scenario_id"] for row in episodes} != set(by_id):
        errors.append("episode_scenario_set_mismatch")
    if any(row.get("inconclusive") for row in episodes):
        errors.append("inconclusive_bank_episode")
    return {
        "case_count": len(cases), "episode_count": len(episodes),
        "selected_templates": dict(Counter(case["catalogue_id"] for case in cases)),
        "bounds_and_initial_geometry_errors": errors,
        "all_static_checks_pass": not errors,
        "old_s02_first_exit_zero_count": sum(
            row["template_id"] == "fbrt_cutout_static" and
            row.get("event_times", {}).get("first_exit_s") == 0.0
            for row in episodes),
        "old_s02_ppo_lead_collision_count": sum(
            row["template_id"] == "fbrt_cutout_static" and
            row["build_id"].startswith("ppo_") and
            row.get("ego_collision") and row.get("collision_partner_role") == "lead"
            for row in episodes),
    }


def _new_case(old: dict, bounds: dict, label: str, overrides: dict) -> dict:
    case = deepcopy(old)
    a = case["active_parameters"]
    c = case["fixed_context"]
    defaults = {
        "ego_initial_x_m": 120.0,
        "ego_initial_speed_mps": float(c["ego_speed_mps"]),
        "ego_initial_lateral_offset_m": 0.0,
        "ego_initial_heading_offset_rad": 0.0,
        "ego_initial_lane_id": 0,
    }
    a.update(defaults)
    a.update(overrides)
    all_bounds = {**{k: v for k, v in bounds.items() if k != "ego_initial_speed_mps_by_template"},
                  "ego_initial_speed_mps": bounds["ego_initial_speed_mps_by_template"][
                      case["template_id"]]}
    for key, limits in all_bounds.items():
        if not limits[0] <= a[key] <= limits[1]:
            raise ValueError(f"{label}:{key}={a[key]} outside {limits}")
    if a["ego_initial_lane_id"] != int(a["ego_initial_lane_id"]):
        raise ValueError("ego_initial_lane_id must be discrete")
    case["research_bounds"].update(all_bounds)
    case["parameterization_version"] = "research_v3_ego_initial"
    case["context_id"] = old["context_id"] + ":ego_initial_v3"
    case["scenario_id"] = f"audit-v3:{old['catalogue_id']}:{label}"
    case["audit_label"] = label
    case["audit_parent_scenario_id"] = old["scenario_id"]
    return case


def _audit_specs(cases: list[dict], config: dict) -> list[tuple[str, dict, str]]:
    base = {case["catalogue_id"]: case for case in cases
            if case["sampling_kind"] == "anchor_like" and
            case["scenario_id"].endswith(":0")}
    bounds = config["initial_state_bounds"]
    specs: list[tuple[str, dict, str]] = []
    for card in config["audit"]["scene_representatives"]:
        case = _new_case(base[card], bounds, "baseline", {})
        specs.extend((card + ":baseline", case, build)
                     for build in (BUILD_MOBIL, BUILD_PPO))
        if card == "S02":
            specs.append((card + ":baseline", case, BUILD_DELAY))
        if card == "S05":
            specs.append((card + ":baseline", case, BUILD_MUTANT))
        if card in {"S02", "S05"}:
            specs.extend((card + ":baseline", case, build)
                         for build in (BUILD_VI, BUILD_MCTS))
        for corner, declared in config["audit"]["all_template_initial_state_corners"].items():
            overrides = {key: value for key, value in declared.items() if key != "speed_bound"}
            speed_limits = bounds["ego_initial_speed_mps_by_template"][case["template_id"]]
            overrides["ego_initial_speed_mps"] = speed_limits[
                0 if declared["speed_bound"] == "lower" else 1]
            corner_case = _new_case(base[card], bounds, "corner_" + corner, overrides)
            specs.extend((card + ":corner_" + corner, corner_case, build)
                         for build in (BUILD_MOBIL, BUILD_PPO))
    for label, values in config["audit"]["ego_sweep"].items():
        case = _new_case(base["S05"], bounds, label, values)
        specs.extend(("S05:" + label, case, build)
                     for build in (BUILD_MOBIL, BUILD_MUTANT))
    for label, values in config["audit"]["ppo_ego_sweep"].items():
        case = _new_case(base["S02"], bounds, label, values)
        specs.extend(("S02:" + label, case, build)
                     for build in (BUILD_PPO, BUILD_DELAY))
    return specs


def _load_trace(row: dict) -> list[dict]:
    return read_jsonl(ROOT / row["trajectory_path"])


def _draw_panel(draw: ImageDraw.ImageDraw, left: int, top: int, row: dict,
                state: dict, action: str, font: ImageFont.ImageFont) -> None:
    width = 600
    draw.rectangle((left, top, left + width - 1, top + 418),
                   fill="#f8fafc", outline="#cbd5e1", width=2)
    name = row["build_id"]
    draw.text((left + 12, top + 8), name, fill="#0f172a", font=font)
    draw.text((left + 12, top + 28),
              f"t={state['time_s']:.2f}s  ego={state['ego']['speed_mps']:.1f} m/s  action={action}",
              fill="#334155", font=font)
    crashed_now = bool(state["ego"]["crashed"])
    partner = row.get("collision_partner_role") if crashed_now else None
    if crashed_now and not partner:
        co_crashed = [role for role, actor in state.items()
                      if role not in {"ego", "time_s"} and
                      isinstance(actor, dict) and actor.get("crashed")]
        if len(co_crashed) == 1:
            partner = co_crashed[0] + "*"
    draw.text((left + 12, top + 48),
              f"collision_now={crashed_now} partner={partner}",
              fill="#b91c1c" if crashed_now else "#475569", font=font)
    runtime = row["ego_runtime"]
    draw.text((left + 12, top + 68),
              f"initial x={runtime['initial_x_m']:.1f} y={runtime['initial_y_m']:.2f} "
              f"v={runtime['initial_speed_mps']:.1f} yaw={runtime['initial_heading_rad']:.3f}",
              fill="#334155", font=font)
    road_top, road_bottom = top + 88, top + 360
    draw.rectangle((left + 2, road_top, left + width - 3, road_bottom), fill="#475569")
    lane_centers = [top + 156, top + 292]
    draw.line((left + 2, top + 224, left + width - 3, top + 224),
              fill="#f8fafc", width=2)
    for k in range(12):
        x0 = left + k * 55
        draw.line((x0, top + 224, min(x0 + 27, left + width - 3), top + 224),
                  fill="#facc15", width=3)
    camera = state["ego"]["x_m"]
    for role, actor in state.items():
        if role == "time_s" or not isinstance(actor, dict):
            continue
        x = left + 155 + (actor["x_m"] - camera) * 4.1
        y = lane_centers[0] + actor["y_m"] / 4.0 * (lane_centers[1] - lane_centers[0])
        if not left - 20 < x < left + width + 20:
            continue
        color = {"ego": "#38bdf8", "lead": "#f97316", "rear": "#a78bfa",
                 "static": "#ef4444"}.get(role, "#94a3b8")
        if actor.get("crashed"):
            color = "#dc2626"
        box = (round(x - 11), round(y - 5), round(x + 11), round(y + 5))
        draw.rectangle(box, fill=color, outline="#0f172a", width=2)
        draw.text((x - 15, y - 22), role, fill="#ffffff", font=font)
    draw.text((left + 12, top + 377),
              "blue ego  orange lead  purple rear  red static  |  *trace inferred",
              fill="#334155", font=font)


def _action_at(row: dict, stamp: float) -> str:
    actions = row.get("ego_actions", [])
    current = "native" if not actions else "pending"
    for item in actions:
        if float(item["time_s"]) > stamp + 1e-9:
            break
        current = item["action"]
    return current


def _render_pair(path: Path, left_row: dict, right_row: dict) -> dict:
    left_trace, right_trace = _load_trace(left_row), _load_trace(right_row)
    last = max(left_trace[-1]["time_s"], right_trace[-1]["time_s"])
    stamps = [round(x * 0.2, 3) for x in range(int(math.ceil(last / 0.2)) + 1)]
    font = ImageFont.load_default()
    frames = []
    for stamp in stamps:
        image = Image.new("RGB", (1200, 420), "white")
        draw = ImageDraw.Draw(image)
        left_state = left_trace[min(round(stamp / 0.05), len(left_trace) - 1)]
        right_state = right_trace[min(round(stamp / 0.05), len(right_trace) - 1)]
        _draw_panel(draw, 0, 0, left_row, left_state,
                    _action_at(left_row, left_state["time_s"]), font)
        _draw_panel(draw, 600, 0, right_row, right_state,
                    _action_at(right_row, right_state["time_s"]), font)
        frames.append(image.quantize(colors=64, method=Image.Quantize.FASTOCTREE))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=200, loop=0, optimize=True, disposal=2)
    return {"file": path.name, "frames": len(frames), "duration_s": round(last, 2),
            "left_execution_id": left_row["execution_id"],
            "right_execution_id": right_row["execution_id"]}


def run_audit() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["schema"] != "fbrt_ego_initial_state_audit_v3":
        raise ValueError("unexpected audit configuration")
    frozen_cases = read_jsonl(ROOT / "compact_bank" / "scenario_cases.jsonl")
    frozen_episodes = read_jsonl(ROOT / "compact_bank" / "episodes.jsonl")
    frozen = _validate_frozen_bank(frozen_cases, frozen_episodes)
    if not frozen["all_static_checks_pass"]:
        raise ValueError(f"frozen scenario geometry failed: {frozen['bounds_and_initial_geometry_errors']}")
    specs = _audit_specs(frozen_cases, config)
    distinct_cases = {case["scenario_id"]: case for _, case, _ in specs}
    _write_json(OUT / "audit_cases.json", list(distinct_cases.values()))
    manifest_path = OUT / "ego_initial_state_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        keys = ["scenario_id", "catalogue_id", "audit_label", "ego_initial_x_m",
                "ego_initial_speed_mps", "ego_initial_lane_id",
                "ego_initial_lateral_offset_m", "ego_initial_heading_offset_rad"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for case in distinct_cases.values():
            writer.writerow({**{key: case[key] for key in keys[:3]},
                             **{key: case["active_parameters"][key] for key in keys[3:]}})
    bank_path = OUT / "episodes.jsonl"
    by_id = {row["execution_id"]: row for row in read_jsonl(bank_path)}
    ledger = _load_ledger()
    outcomes = {}
    newly_executed = 0
    for label, case, build in specs:
        row, status = _ensure_physical_episode(
            build, case, "validation_audit", GLOBAL_PHYSICAL_CAP,
            bank_path, by_id, ledger, capture_trace=True)
        if row is None:
            raise RuntimeError(f"audit physical execution denied: {label} {build} {status}")
        newly_executed += status == "executed"
        outcomes[(label, build)] = row
    runtime_errors = []
    measurement_warnings = []
    inferred_partners = {}
    for (label, build), row in outcomes.items():
        runtime = row["ego_runtime"]
        initial = row["scenario"]["active_parameters"]
        trace = _load_trace(row)
        if row.get("ego_collision") and not row.get("collision_partner_role"):
            co_crashed = [role for role, actor in trace[-1].items()
                          if role not in {"ego", "time_s"} and
                          isinstance(actor, dict) and actor.get("crashed")]
            if len(co_crashed) == 1:
                inferred_partners[(label, build)] = co_crashed[0]
                measurement_warnings.append(
                    f"{label}:{build}:raw_collision_partner_missing;"
                    f"terminal_co_crashed_actor={co_crashed[0]}")
            else:
                measurement_warnings.append(
                    f"{label}:{build}:raw_collision_partner_missing;"
                    f"co_crashed_actors={co_crashed}")
        if abs(runtime["initial_x_m"] - initial["ego_initial_x_m"]) > 1e-6:
            runtime_errors.append(f"{label}:{build}:ego_x_not_applied")
        if abs(runtime["initial_speed_mps"] - initial["ego_initial_speed_mps"]) > 1e-6:
            runtime_errors.append(f"{label}:{build}:ego_speed_not_applied")
        if abs(runtime["initial_heading_rad"] - initial["ego_initial_heading_offset_rad"]) > 1e-6:
            runtime_errors.append(f"{label}:{build}:ego_heading_not_applied")
        expected_y = 4.0 * initial["ego_initial_lane_id"] + initial["ego_initial_lateral_offset_m"]
        if abs(runtime["initial_y_m"] - expected_y) > 1e-6:
            runtime_errors.append(f"{label}:{build}:ego_lane_or_lateral_not_applied")
        first = trace[0]
        if any(actor["crashed"] for role, actor in first.items() if role != "time_s"):
            runtime_errors.append(f"{label}:{build}:initial_collision")
        if build.startswith("mobil_"):
            if (runtime["vehicle_class"] != "RearGuardIDMVehicle" or
                    runtime["native_act_calls"] < 1 or runtime["mobil_decision_calls"] < 1):
                runtime_errors.append(f"{label}:{build}:native_policy_not_exercised")
        else:
            expected_policy = {
                BUILD_VI: "ValueIterationPolicy", BUILD_MCTS: "MCTSCVPolicy",
            }.get(build, "PPOPolicy")
            if (runtime["vehicle_class"] != "MDPVehicle" or
                    runtime["policy_class"] != expected_policy or
                    row["ego_action_count"] < 1 or
                    (expected_policy == "PPOPolicy" and not runtime["checkpoint_sha256"])):
                runtime_errors.append(f"{label}:{build}:external_policy_not_exercised")
        if label.startswith("S02:"):
            event = row.get("event_times", {}).get("first_exit_s")
            if event is None or event < 1.0:
                runtime_errors.append(f"{label}:{build}:cutout_event_time_invalid")
        if label == "S08:baseline":
            phases = {event["phase"] for event in row.get("lead_events", [])}
            if "BRAKE_AFTER_MERGE" not in phases:
                runtime_errors.append(f"{label}:{build}:brake_event_absent")
    translation_residual = {}
    for build in (BUILD_MOBIL, BUILD_MUTANT):
        baseline = _load_trace(outcomes[("S05:baseline", build)])
        shifted = _load_trace(outcomes[("S05:position_shift", build)])
        if len(baseline) != len(shifted):
            runtime_errors.append(f"S05:{build}:translation_changes_duration")
            continue
        residual = max(
            abs(right[role]["x_m"] - left[role]["x_m"] - 20.0)
            for left, right in zip(baseline, shifted)
            for role in ("ego", "lead", "rear"))
        residual = max(residual, *(
            abs(right[role][key] - left[role][key])
            for left, right in zip(baseline, shifted)
            for role in ("ego", "lead", "rear")
            for key in ("y_m", "speed_mps")))
        translation_residual[build] = residual
        if residual > 1e-6:
            runtime_errors.append(f"S05:{build}:translation_changes_relative_dynamics")
    gif_specs = [(f"scene_{card}.gif", outcomes[(card + ":baseline", BUILD_MOBIL)],
                  outcomes[(card + ":baseline", BUILD_PPO)])
                 for card in config["audit"]["scene_representatives"]]
    gif_specs.extend((f"ego_corner_{card}.gif",
                      outcomes[(card + ":corner_low", BUILD_PPO)],
                      outcomes[(card + ":corner_high", BUILD_PPO)])
                     for card in config["audit"]["scene_representatives"])
    gif_specs.extend((f"policy_vi_mcts_{card}.gif",
                      outcomes[(card + ":baseline", BUILD_VI)],
                      outcomes[(card + ":baseline", BUILD_MCTS)])
                     for card in ("S02", "S05"))
    gif_specs.extend([
        ("policy_mobil_guard_pair.gif", outcomes[("S05:baseline", BUILD_MOBIL)],
         outcomes[("S05:baseline", BUILD_MUTANT)]),
        ("policy_ppo_delay_pair.gif", outcomes[("S02:baseline", BUILD_PPO)],
         outcomes[("S02:baseline", BUILD_DELAY)]),
        ("ego_speed_low_vs_high.gif", outcomes[("S05:speed_low", BUILD_MUTANT)],
         outcomes[("S05:speed_high", BUILD_MUTANT)]),
        ("ego_lane_0_vs_1.gif", outcomes[("S05:baseline", BUILD_MUTANT)],
         outcomes[("S05:adjacent_lane", BUILD_MUTANT)]),
        ("ego_x_120_vs_140.gif", outcomes[("S05:baseline", BUILD_MUTANT)],
         outcomes[("S05:position_shift", BUILD_MUTANT)]),
        ("ego_lateral_0_vs_020.gif", outcomes[("S05:baseline", BUILD_MUTANT)],
         outcomes[("S05:lateral_offset", BUILD_MUTANT)]),
        ("ego_heading_0_vs_003.gif", outcomes[("S05:baseline", BUILD_MUTANT)],
         outcomes[("S05:heading_offset", BUILD_MUTANT)]),
        ("ego_s02_lane_0_vs_1.gif", outcomes[("S02:baseline", BUILD_PPO)],
         outcomes[("S02:lane_1", BUILD_PPO)]),
        ("ego_s02_speed25_lane_0_vs_1.gif", outcomes[("S02:speed_high", BUILD_PPO)],
         outcomes[("S02:lane_1_speed_high", BUILD_PPO)]),
        ("policy_ppo_delay_lane1.gif", outcomes[("S02:lane_1", BUILD_PPO)],
         outcomes[("S02:lane_1", BUILD_DELAY)]),
    ])
    gifs = [_render_pair(OUT / "gifs" / name, left, right)
            for name, left, right in gif_specs]
    summary = {
        "frozen_bank": frozen,
        "new_audit_episodes": len(specs),
        "newly_executed_this_run": newly_executed,
        "global_physical_episodes": ledger["new_physical_episodes"],
        "global_physical_cap": GLOBAL_PHYSICAL_CAP,
        "runtime_errors": runtime_errors,
        "runtime_checks_pass": not runtime_errors,
        "measurement_warnings": measurement_warnings,
        "position_translation_max_residual": translation_residual,
        "outcomes": [{"label": label, "build_id": build,
                      "scenario_id": row["scenario_id"],
                      "execution_id": row["execution_id"],
                      "ego_collision": row["ego_collision"],
                      "collision_partner_role": row.get("collision_partner_role"),
                      "inferred_collision_partner_role": inferred_partners.get((label, build)),
                      "ego_action_count": row["ego_action_count"],
                      "rear_guard_bypassed_count": row["rear_guard_bypassed_count"],
                      "runtime": row["ego_runtime"],
                      "event_times": row["event_times"]}
                     for (label, build), row in outcomes.items()],
        "gifs": gifs,
    }
    _write_json(OUT / "audit_manifest.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2))
