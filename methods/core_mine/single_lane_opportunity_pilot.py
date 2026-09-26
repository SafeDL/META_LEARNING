"""Development-only source-safe opportunity check for one-lane interactions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from highway_sim_env.envs.cutin_env import CutInScenario
from highway_sim_env.envs.single_lane_longitudinal import (
    SingleLaneExternalCutInEnv, SingleLaneLongitudinalEnv,
)
from methods.core_mine import sparse_sut_experiment as sparse
from methods.core_mine.fvdm_revision_pilot import BUILDS
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


ROOT = Path("results/method_chains/core_mine/studies/single_lane_opportunity_pilot")
SEED = 20330105
SOURCES = ("idm_mobil", "fvdm_ref")
TARGETS = ("vi_ttc", "fvdm_delay05_brake3")
PILOT_PER_MODE = 24
FIELDS = ("ego_collision", "background_collision", "near_miss", "min_ttc",
          "min_distance", "completed")


def _external(sut: str, scenario: CutInScenario, episode_seed: int) -> dict:
    policy = policy_factory(sut, ASSETS)
    env = SingleLaneExternalCutInEnv(scenario, ego_kind=policy.ego_kind)
    env.config["policy_frequency"] = 20
    try:
        env.reset(seed=episode_seed)
        policy.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(policy.act(env))
        return asdict(env.external_result())
    finally:
        env.close()


def _profile(build: str, scenario: CutInScenario, episode_seed: int) -> dict:
    env = SingleLaneLongitudinalEnv(BUILDS[build], scenario)
    try:
        env.reset(seed=episode_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(1)
        result = env.episode_result()
        collision = bool(env.vehicle.crashed)
        return {"ego_collision": collision,
                "background_collision": any(vehicle.crashed for vehicle
                                            in env.road.vehicles
                                            if vehicle is not env.vehicle),
                "near_miss": not collision and (
                    result.min_ttc < 1.5 or result.min_distance < 1.0),
                "min_ttc": result.min_ttc,
                "min_distance": result.min_distance,
                "completed": bool(not terminated and
                                  env.time >= env.config["duration"])}
    finally:
        env.close()


def _proposal():
    sparse.configure_proposal("v8_source_safe")
    anchors, modes, controls, regimes = sparse.sparse_scenarios(SEED)
    if len(modes) != 320 or len(np.unique(modes)) != 5:
        raise RuntimeError("expected five modes x 64 source-blind proposals")
    return anchors, modes, controls, regimes


def _scenario(anchors, modes, controls, index: int) -> CutInScenario:
    return CutInScenario(float(anchors[index, 0]),
                        float(anchors[index, 1]), str(modes[index]),
                        float(controls[index, 0]),
                        float(controls[index, 1]))


def build_sources() -> dict:
    anchors, modes, controls, regimes = _proposal()
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "source_bank.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as bank:
            if (not np.array_equal(bank["anchors"], anchors)
                    or not np.array_equal(bank["modes"], modes)
                    or not np.array_equal(bank["controls"], controls)
                    or tuple(bank["source_names"].astype(str)) != SOURCES):
                raise RuntimeError("existing pilot source bank differs")
    else:
        values = {field: np.empty((len(SOURCES), len(modes)))
                  for field in FIELDS}
        for source_index, source in enumerate(SOURCES):
            for index in range(len(modes)):
                scenario = _scenario(anchors, modes, controls, index)
                outcome = (_external(source, scenario, SEED + index)
                           if source == "idm_mobil" else
                           _profile(source, scenario, SEED + index))
                for field in FIELDS:
                    values[field][source_index, index] = outcome[field]
            print(f"executed source {source}: {len(modes)}", flush=True)
        np.savez_compressed(
            path, anchors=anchors, modes=modes, controls=controls,
            regimes=regimes, source_names=np.asarray(SOURCES),
            source_ego_control_hz=20, source_physics_hz=20,
            lane_count_by_mode=np.asarray([2, 2, 1, 1, 1]),
            **values)
    with np.load(path, allow_pickle=False) as bank:
        events = bank["ego_collision"].astype(bool) | bank[
            "near_miss"].astype(bool)
        eligible = ~events.any(axis=0) & bank["completed"].all(axis=0)
    counts = {str(mode): int(np.sum(eligible & (modes == mode)))
              for mode in np.unique(modes)}
    gate = {"seed": SEED, "sources": SOURCES,
            "candidate_pool": len(modes), "eligible_total": int(eligible.sum()),
            "eligible_by_mode": counts,
            "source_only_gate_passed": bool(eligible.sum() >= 50
                                            and all(counts.values())),
            "target_pilot_per_mode": PILOT_PER_MODE,
            "target_selection_rule": "first eligible proposal indices per mode",
            "source_ego_control_hz": 20, "source_physics_hz": 20,
            "target_ego_control_hz": 20,
            "mode_lane_counts": {"fast_intrusion": 2,
                                 "cutin_braking": 2,
                                 "lead_braking": 1,
                                 "stop_and_go": 1,
                                 "slow_lead_following": 1}}
    (ROOT / "source_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate), flush=True)
    return gate


def run_targets() -> dict:
    gate = build_sources()
    if not gate["source_only_gate_passed"]:
        raise RuntimeError("source-only pilot gate failed")
    anchors, modes, controls, _ = _proposal()
    with np.load(ROOT / "source_bank.npz", allow_pickle=False) as bank:
        eligible = ~((bank["ego_collision"].astype(bool)
                      | bank["near_miss"].astype(bool)).any(axis=0))
        eligible &= bank["completed"].all(axis=0)
    selected = []
    for mode in np.unique(modes):
        indices = np.flatnonzero(eligible & (modes == mode))
        if len(indices) < PILOT_PER_MODE:
            raise RuntimeError(f"insufficient pilot cases for {mode}")
        selected.extend(indices[:PILOT_PER_MODE].tolist())
    rows = []
    for target in TARGETS:
        for index in selected:
            scenario = _scenario(anchors, modes, controls, index)
            outcome = (_external(target, scenario, SEED + index)
                       if target == "vi_ttc" else
                       _profile(target, scenario, SEED + index))
            rows.append({"seed": SEED, "target": target, "index": index,
                         "mode": str(modes[index]), "gap": float(anchors[index, 0]),
                         "relative_speed": float(anchors[index, 1]),
                         "timing": float(controls[index, 0]),
                         "intensity": float(controls[index, 1]),
                         **{field: outcome[field] for field in FIELDS}})
        print(f"executed target pilot {target}: {len(selected)}", flush=True)
    path = ROOT / "target_pilot.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {target: {mode: {
        "queries": PILOT_PER_MODE,
        "ego_collisions": sum(bool(row["ego_collision"]) for row in rows
                              if row["target"] == target and row["mode"] == mode),
        "near_misses": sum(bool(row["near_miss"]) for row in rows
                           if row["target"] == target and row["mode"] == mode),
    } for mode in np.unique(modes)} for target in TARGETS}
    output = {"development_only": True, "seed": SEED,
              "source_gate": gate, "targets": summary,
              "target_pilot_episodes": len(rows)}
    (ROOT / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "targets", "all"),
                        default="sources")
    args = parser.parse_args()
    if args.stage == "sources":
        build_sources()
    else:
        run_targets()


if __name__ == "__main__":
    main()
