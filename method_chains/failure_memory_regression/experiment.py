"""Run shared physical episodes and replay budgeted FBRT campaigns."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from scipy.stats import qmc

from highway_env_benchmark.envs.fbrt_env import run_episode
from highway_env_benchmark.envs.fbrt_scenarios import (
    ACTIVE_PARAMETERS, BOUNDS, FBRTScenario,
)
from method_chains.core_mine.idm_revision_pilot import REFERENCE
from method_chains.core_mine.local_fault_idm import FAULTS
from method_chains.failure_memory_regression.boundary_memory import build_patches
from method_chains.failure_memory_regression.selectors import TargetOracle, run_selector


ROOT = Path("results/method_chains/failure_memory_regression/standard_aligned")
SEEDS = (4179801, 4179802, 4179803)
METHODS = ("Random", "ART-Maximin", "HistoryMargin", "FailureDistance",
           "HistoryRank-UCB", "FBRT-Static", "FBRT-Adaptive", "FBRT-RegionBandit")
BUDGET = 50
CHECKPOINTS = (5, 10, 20, 50)
STANDARD_URL = "https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=C3FD7FF23C6D06A9F7459DCD73E68905"
SIMULATION_CONTRACT = "highway-env;20Hz;functional-scripts;idm-reference"


def initial_scenarios(seed: int) -> list[FBRTScenario]:
    scenes = []
    for offset, (template, bounds) in enumerate(BOUNDS.items()):
        samples = qmc.Sobol(d=2, scramble=True, seed=seed + offset).random_base2(m=5)
        second_name = ACTIVE_PARAMETERS[template][1]
        range_name = "wide_" if bounds[0][1] > 60.0 else ""
        for index, sample in enumerate(samples):
            x = bounds[0][0] + float(sample[0]) * (bounds[0][1] - bounds[0][0])
            y = bounds[1][0] + float(sample[1]) * (bounds[1][1] - bounds[1][0])
            scenes.append(FBRTScenario(f"{seed}:{template}:{range_name}initial:{index}", template,
                                       x, **{second_name: y}))
    return scenes


def probe_scenarios(seed: int) -> list[FBRTScenario]:
    scenes = []
    coordinates = ((0.0, 0.0), (0.0, 1.0), (0.12, 0.25), (0.12, 0.75))
    for template, bounds in BOUNDS.items():
        range_name = "wide_" if bounds[0][1] > 60.0 else ""
        for index, (x, y) in enumerate(coordinates):
            scenes.append(FBRTScenario(
                f"{seed}:{template}:{range_name}probe:{index}", template,
                bounds[0][0] + x * (bounds[0][1] - bounds[0][0]),
                **{ACTIVE_PARAMETERS[template][1]: bounds[1][0] + y *
                   (bounds[1][1] - bounds[1][0])}))
    return scenes


def midpoint_scenarios(seed: int, scenes: list[FBRTScenario],
                       source: dict[str, dict]) -> list[FBRTScenario]:
    patches = build_patches(scenes, source)
    output = []
    by_id = {scene.scenario_id: scene for scene in scenes}
    for template, bounds in BOUNDS.items():
        local = [patch for patch in patches if patch.template_id == template]
        if not local:
            for index, (x, y) in enumerate(((0.25, 0), (0.25, 1), (0.35, 0.25),
                                             (0.35, 0.75))):
                output.append(FBRTScenario(
                    f"{seed}:{template}:core_extra:{index}", template,
                    bounds[0][0] + x * (bounds[0][1] - bounds[0][0]),
                    **{ACTIVE_PARAMETERS[template][1]: bounds[1][0] + y *
                       (bounds[1][1] - bounds[1][0])}))
            continue
        for index in range(4):
            patch = local[index % len(local)]
            failed = by_id[patch.failed_id]
            passed = by_id[patch.passed_id]
            fraction = (0.5, 0.25, 0.75, 0.5)[index]
            values = tuple(a + fraction * (b - a) for a, b in
                           zip(failed.active_values(), passed.active_values()))
            output.append(FBRTScenario(
                f"{seed}:{template}:core_boundary:{index}", template, values[0],
                **{ACTIVE_PARAMETERS[template][1]: values[1]}))
    return output


def _job(args: tuple[str, FBRTScenario, int]) -> dict:
    build, scene, seed = args
    outcome, _ = run_episode(REFERENCE, scene, seed, None if build == "idm_ref" else build)
    return {"build": build, "seed": seed, "scenario_id": scene.scenario_id,
            "scenario": scene.as_record(), "contract": SIMULATION_CONTRACT,
            **outcome}


def cache_key(build: str, scene: FBRTScenario, seed: int) -> str:
    return f"{build}|{seed}|{scene.scenario_id}"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {f"{row['build']}|{row['seed']}|{row['scenario_id']}": row for row in records}


def ensure_episodes(scenes: list[FBRTScenario], build: str, seed: int,
                    cache: dict[str, dict], path: Path, workers: int,
                    ledger: dict) -> dict[str, dict]:
    for scene in scenes:
        previous = cache.get(cache_key(build, scene, seed))
        if previous is not None and (previous["contract"] != SIMULATION_CONTRACT or
                                     previous["scenario"] != scene.as_record()):
            raise RuntimeError(f"Cached episode contract differs: {scene.scenario_id}")
    missing = [scene for scene in scenes if cache_key(build, scene, seed) not in cache]
    ledger["cache_hits"] += len(scenes) - len(missing)
    if missing:
        jobs = [(build, scene, seed) for scene in missing]
        with ProcessPoolExecutor(max_workers=workers) as executor, path.open(
                "a", encoding="utf-8") as handle:
            for row in executor.map(_job, jobs, chunksize=2):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                cache[f"{row['build']}|{row['seed']}|{row['scenario_id']}"] = row
                field = "new_reference_episodes" if build == "idm_ref" else "new_target_episodes"
                ledger[field] += 1
    return {scene.scenario_id: cache[cache_key(build, scene, seed)] for scene in scenes}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scene_from_record(row: dict) -> FBRTScenario:
    return FBRTScenario(row["scenario_id"], row["template_id"],
                        float(row["initial_clearance_m"]),
                        **{name: float(row[name]) for name in ACTIVE_PARAMETERS[row["template_id"]][1:]})


def campaigns(seed: int, scenes: list[FBRTScenario], candidates: list[FBRTScenario],
              source: dict[str, dict], target_banks: dict[str, dict[str, dict]],
              patches: list, ledger: dict) -> tuple[list[dict], list[dict]]:
    queries_out = []
    summary_out = []
    for build, target in target_banks.items():
        for method in METHODS:
            for repeat in range(20 if method == "Random" else 1):
                oracle = TargetOracle(target)
                queries = run_selector(method, candidates, scenes, source, patches, oracle,
                                       BUDGET, random_seed=seed + repeat)
                queries_out.extend({"seed": seed, "build": build,
                                    "repeat": repeat, **row} for row in queries)
                ledger["logical_queries_per_campaign"][f"{seed}:{build}:{method}:{repeat}"] = len(queries)
                for checkpoint in CHECKPOINTS:
                    prefix = queries[:checkpoint]
                    ranks = [row["rank"] for row in prefix if row["regression"]]
                    total_regressions = sum(row["ego_collision"] for row in target.values())
                    summary_out.append({"seed": seed, "build": build,
                                        "method": method, "repeat": repeat,
                                        "budget": checkpoint, "actual_queries": len(prefix),
                                        "regression_detected": bool(ranks),
                                        "first_regression_rank": min(ranks) if ranks else None,
                                        "regression_collision_count": len(ranks),
                                        "regression_template_count": len({
                                            row["template_id"] for row in prefix if row["regression"]}),
                                        "regression_recall": len(ranks) / total_regressions,
                                        "observed_regressions_in_pool": total_regressions})
    return queries_out, summary_out


def write_report_header(output: Path, summary: list[dict], candidate_count: int,
                        patch_count: int, reference_count: int, target_count: int,
                        offline: bool) -> None:
    report = ["# FBRT core 结果", "",
              "场景为规范启发的 highway-env 研究场景；参考通过指完整时长无 ego 碰撞。",
              "所有选择器仅通过逐次查询接口看到目标结果；方法共用同一实测结果库。", "",
              f"候选数：{candidate_count}；历史局部边界对：{patch_count}；",
              f"本配置使用的实测参考 episode：{reference_count}；实测目标 episode：{target_count}。",
              "本次从已保存的实测结果库重放选例，新增仿真 episode：0。" if offline else
              "本次运行包含实测仿真 episode；具体成本见 compute_ledger.json。", "",
              f"## 每个种子与目标的 @{BUDGET} 首次回归查询位置", "",
              "| 种子 | 目标 | 方法 | 检测 | 首次位置 | 碰撞数 |",
              "|---|---|---|---:|---:|---:|"]
    for row in summary:
        if row["budget"] == BUDGET and row["repeat"] == 0:
            report.append(f"| {row['seed']} | {row['build']} | {row['method']} | "
                          f"{int(row['regression_detected'])} | "
                          f"{row['first_regression_rank'] or '未发现'} | "
                          f"{row['regression_collision_count']} |")
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def with_trigger_geometry(row: dict) -> dict:
    scene = row["scenario"]
    if scene["template_id"] != "fbrt_cutout_static":
        return row
    # VT1 runs at exactly 20 m/s until the 1 s trigger; verified against replay.
    return {**row,
            "trigger_clearance_m": scene["lead_speed_mps"] * scene["static_target_ttc_s"],
            "trigger_ttc_s": scene["static_target_ttc_s"],
            "trigger_geometry_source": "fixed_speed_script_at_event_start"}


def write_mapping() -> None:
    mapping = ROOT / "standard_mapping.md"
    mapping.write_text(
        "# 场景与规范来源\n\n"
        "本实验是规范启发的功能对齐研究，不是标准认证。物理参数范围为研究设置。\n\n"
        "| template_id | functional_name | reference_standard | verified_parent_section | specific_clause_status | source_url | borrowed_behavior_skeleton | research_parameter_ranges | deviations |\n"
        "|---|---|---|---|---|---|---|---|---|\n" +
        "\n".join(
            f"| {template} | {name} | GB/T 41798-2022；GB/T 47025-2026（仿真参考） | "
            f"6.4 周边车辆响应／6.5 自动紧急避险的功能框架 | 未核对具体子条款 | "
            f"{STANDARD_URL} | {skeleton}；中汽研解读 https://www.castc.net/news/9807.cshtml | "
            f"{BOUNDS[template]} | highway-env 研究简化，非标准规定数值 |"
            for template, name, skeleton in (
                ("fbrt_cutin", "前方车辆切入", "相邻前车切入本车道"),
                ("fbrt_lead_emergency_brake", "单车道前车紧急制动", "前车刹停"),
                ("fbrt_stop_hold_go", "前车停保持起步", "前车停车、保持、起步"),
                ("fbrt_cutout_static", "切出后静止车辆", "前车切出、露出静止车辆"))) + "\n",
        encoding="utf-8")


def write_contracts() -> None:
    payload = {template: {"active_parameters": ACTIVE_PARAMETERS[template],
                          "bounds": BOUNDS[template],
                          "lane_count": 1 if template == "fbrt_lead_emergency_brake" else 2,
                          "physics_hz": 20, "control_hz": 20,
                          "event_start_s": 1.0}
               for template in BOUNDS}
    (ROOT / "scenario_contracts.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                   encoding="utf-8")


def audit() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    write_mapping()
    write_contracts()
    manifest = {"reference": asdict(REFERENCE), "targets": list(FAULTS),
                "simulator": SIMULATION_CONTRACT,
                "prior_local_fault_summary": "results/method_chains/core_mine/studies/local_fault_pilot/summary.json",
                "prior_records_not_reused_as_new_scenario_results": True}
    (ROOT / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                  encoding="utf-8")


def run(workers: int) -> dict:
    start = time.monotonic()
    audit()
    cache_path = ROOT / "episode_cache.jsonl"
    cache = load_cache(cache_path)
    ledger = {"new_reference_episodes": 0, "new_target_episodes": 0,
              "replay_episodes": 0,
              "cache_hits": 0, "logical_queries_per_campaign": {},
              "wall_clock_seconds": 0}
    all_reference = []
    all_candidates = []
    all_patches = []
    all_targets = []
    all_queries = []
    all_summary = []
    for seed in SEEDS:
        initial = initial_scenarios(seed)
        source = ensure_episodes(initial, "idm_ref", seed, cache, cache_path,
                                 workers, ledger)
        probes = probe_scenarios(seed)
        source.update(ensure_episodes(probes, "idm_ref", seed, cache, cache_path,
                                      workers, ledger))
        scenes = initial + probes
        midpoints = midpoint_scenarios(seed, scenes, source)
        source.update(ensure_episodes(midpoints, "idm_ref", seed, cache, cache_path,
                                      workers, ledger))
        scenes.extend(midpoints)
        patches = build_patches(scenes, source)
        candidates = [scene for scene in scenes if source[scene.scenario_id]["completed"]
                      and not source[scene.scenario_id]["ego_collision"]]
        all_reference.extend(with_trigger_geometry({**row, "reference_pass": bool(
            row["completed"] and not row["ego_collision"])})
            for row in source.values())
        all_candidates.extend(scene.as_record() for scene in candidates)
        all_patches.extend({"seed": seed, **patch.as_record()} for patch in patches)
        target_banks = {}
        for build in FAULTS:
            target = ensure_episodes(candidates, build, seed, cache, cache_path,
                                     workers, ledger)
            target_banks[build] = target
            all_targets.extend(with_trigger_geometry(row) for row in target.values())
        queries, summary = campaigns(seed, scenes, candidates, source, target_banks,
                                     patches, ledger)
        all_queries.extend(queries)
        all_summary.extend(summary)
        print(json.dumps({"seed_finished": seed, "candidates": len(candidates),
                          "patches": len(patches),
                          "target_regressions": {build: sum(
                              row["ego_collision"] for row in bank.values())
                              for build, bank in target_banks.items()}}), flush=True)
    output = ROOT / "core"
    output.mkdir(exist_ok=True)
    for filename, rows in (("reference_archive.csv", all_reference),
                           ("boundary_memory.csv", all_patches),
                           ("candidate_pool.csv", all_candidates),
                           ("target_response_bank.csv", all_targets),
                           ("queries.csv", all_queries),
                           ("summary_by_revision.csv", all_summary)):
        write_csv(output / filename, rows)
    template_summary = []
    for seed in SEEDS:
        for template in BOUNDS:
            refs = [row for row in all_reference if row["seed"] == seed and
                    row["scenario"]["template_id"] == template]
            for build in FAULTS:
                targets = [row for row in all_targets if row["seed"] == seed and
                           row["build"] == build and
                           row["scenario"]["template_id"] == template]
                template_summary.append({"seed": seed, "template_id": template,
                                         "build": build, "reference_failures": sum(
                                             row["ego_collision"] for row in refs),
                                         "reference_passes": sum(row["completed"] for row in refs),
                                         "target_regressions": sum(row["ego_collision"]
                                                                   for row in targets)})
    write_csv(output / "summary_by_template.csv", template_summary)
    ledger["physical_reference_episodes_used"] = len(all_reference)
    ledger["physical_target_episodes_used"] = len(all_targets)
    ledger["physical_cache_records_total"] = len(cache)
    ledger["wall_clock_seconds"] = round(time.monotonic() - start, 2)
    (output / "compute_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n",
                                                  encoding="utf-8")
    write_report_header(output, all_summary, len(all_candidates), len(all_patches),
                        len(all_reference), len(all_targets), offline=False)
    return {"candidates": len(all_candidates),
            "patches": len(all_patches), "ledger": ledger,
            "report": str(output / "report.md")}


def replay_measured_bank() -> dict:
    """Re-evaluate selection budgets without running or altering physical episodes."""
    output = ROOT / "core"
    archive = read_csv(output / "reference_archive.csv")
    candidates_by_id = {row["scenario_id"] for row in read_csv(output / "candidate_pool.csv")}
    target_rows = read_csv(output / "target_response_bank.csv")
    ledger_path = output / "compute_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["logical_queries_per_campaign"] = {}
    all_queries = []
    all_summary = []
    for seed in SEEDS:
        source_rows = [row for row in archive if int(row["seed"]) == seed]
        scenes = [scene_from_record(json.loads(row["scenario"])) for row in source_rows]
        source = {row["scenario_id"]: {
            "completed": row["completed"] == "True",
            "ego_collision": row["ego_collision"] == "True",
            "min_ttc": float(row["min_ttc"]),
            "min_clearance": float(row["min_clearance"])} for row in source_rows}
        candidates = [scene for scene in scenes if scene.scenario_id in candidates_by_id]
        patches = build_patches(scenes, source)
        target_banks = {
            build: {row["scenario_id"]: {
                "ego_collision": row["ego_collision"] == "True",
                "semantic_valid": row["semantic_valid"] == "True"}
                for row in target_rows if int(row["seed"]) == seed and row["build"] == build}
            for build in FAULTS}
        for build, bank in target_banks.items():
            if set(bank) != {scene.scenario_id for scene in candidates}:
                raise ValueError(f"Incomplete measured target bank: {seed}:{build}")
        queries, summary = campaigns(seed, scenes, candidates, source, target_banks,
                                     patches, ledger)
        all_queries.extend(queries)
        all_summary.extend(summary)
    write_csv(output / "queries.csv", all_queries)
    write_csv(output / "summary_by_revision.csv", all_summary)
    write_report_header(output, all_summary, len(candidates_by_id),
                        len(read_csv(output / "boundary_memory.csv")), len(archive),
                        len(target_rows), offline=True)
    ledger["offline_replay_from_measured_bank"] = True
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return {"candidates": len(candidates_by_id), "campaigns": len(
        ledger["logical_queries_per_campaign"]), "logical_queries": len(all_queries),
        "new_physical_episodes": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replay-measured-bank", action="store_true")
    args = parser.parse_args()
    result = replay_measured_bank() if args.replay_measured_bank else run(args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
