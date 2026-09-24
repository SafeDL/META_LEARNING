"""Executable AdaTE A0 response-mixture and A1 DenseRL protocols."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from highway_env_benchmark.data.generate_anchor_bank import generate_anchor_bank
from highway_env_benchmark.data.response_bank import ResponseBank, build_response_bank
from sut_algorithms.highway_env.idm_profiles import get_profile

from .dense_protocol import _scenario_pool, _scenario_probabilities
from .dense_protocol import run_dense_batch, run_dense_protocol
from .dense_env import DenseCutInEnv
from .mixture_selector import MixtureSelector


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _manifest(output: Path, config_path: Path, stage: str, seed: int, costs: dict) -> None:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    payload = {
        "stage": stage,
        "seed": seed,
        "config": str(config_path),
        "git_revision": revision,
        "environment": "highway-env==1.9.1",
        "cost_ledger": costs,
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mixture_bank(config: dict, output: Path,
                  bank_path: Path | None) -> tuple[ResponseBank, int, int]:
    """Build a response bank while keeping target roles explicit when requested."""
    if bank_path is not None:
        return ResponseBank.load(bank_path), 0, 0
    seed = int(config["seed"])
    source_names = [str(name) for name in config.get("source_profiles", [])]
    target_names = [str(name) for name in config.get("target_profiles", [])]
    profile_names = source_names + target_names if source_names or target_names else []
    profiles = tuple(get_profile(name) for name in profile_names) if profile_names else None
    anchors = generate_anchor_bank(int(config["num_anchors"]), seed)
    modes = config.get("modes")
    if modes is not None:
        modes = np.asarray([str(modes[index % len(modes)]) for index in range(len(anchors))])
    bank = build_response_bank(anchors, seed, modes=modes, profiles=profiles)
    bank.save(output / "source_response_bank.npz")
    source_episodes = len(source_names) * len(anchors) if source_names else bank.vulnerability.size
    target_oracle_episodes = len(target_names) * len(anchors)
    return bank, source_episodes, target_oracle_episodes


def run_mixture(config: dict, output: Path, bank_path: Path | None) -> None:
    """Run target-hidden A0 on a saved real highway-env response bank."""
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    seed, budget = int(config["seed"]), int(config["budget"])
    bank, source_episodes, target_oracle_episodes = _mixture_bank(config, output, bank_path)
    if budget > len(bank.anchors):
        raise ValueError("mixture budget exceeds bank size")
    static_ks = [int(value) for value in config.get("static_ks", [config.get("static_k", 4)])]
    variants = [("Uniform-Mixture-H", "uniform", 0)]
    variants.extend(
        (f"AdaTE-Mixture-H-staticK{static_k}", "static", static_k) for static_k in static_ks)
    variants.append(("AdaTE-Mixture-H-sequential", "sequential", 0))
    responses = [
        str(value) for value in config.get("responses", [config.get("response", "vulnerability")])
    ]
    trace_rows: list[dict] = []
    alpha_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    summary_rows: list[dict] = []
    agreement_rows: list[dict] = []
    source_names = [str(name) for name in config.get("source_profiles", [])]
    target_names = [str(name) for name in config.get("target_profiles", [])]
    if bool(source_names) != bool(target_names):
        raise ValueError("source_profiles and target_profiles must be declared together")
    if set(source_names) & set(target_names):
        raise ValueError("A0 targets must remain held out from source profiles")
    if source_names:
        missing = set(source_names + target_names) - set(bank.sut_names)
        if missing:
            raise ValueError(f"response bank lacks configured profiles: {sorted(missing)}")
        source_indices = [bank.index_of(name) for name in source_names]
        target_indices = [bank.index_of(name) for name in target_names]
    else:
        source_indices = []
        target_indices = list(range(len(bank.sut_names)))
    for target_index in target_indices:
        target_name = bank.sut_names[target_index]
        critical = bank.collisions[target_index] | bank.near_misses[target_index]
        for response_name in responses:
            if response_name == "vulnerability":
                response = bank.vulnerability
            elif response_name == "collision":
                response = bank.collisions.astype(float)
            else:
                raise ValueError(f"unsupported A0 response: {response_name}")
            sources = response[source_indices] if source_indices else np.delete(
                response, target_index, axis=0)
            target = response[target_index]
            if source_names:
                for source_index, source_name in zip(source_indices, source_names, strict=True):
                    difference = response[source_index] - target
                    agreement_rows.append({
                        "target_sut": target_name,
                        "source_profile": source_name,
                        "response": response_name,
                        "candidate_count": len(target),
                        "mse": float(np.mean(difference**2)),
                        "max_absolute_error": float(np.max(np.abs(difference))),
                    })
            for display, variant, static_k in variants:
                selector = MixtureSelector(sources, budget, variant, static_k)
                cumulative_critical = 0
                for step in range(1, budget + 1):
                    index = selector.propose()
                    assert index is not None
                    score = float(selector.alpha @ sources[:, index])
                    diagnostic = selector.observe(index, float(target[index]))
                    cumulative_critical += int(critical[index])
                    trace_rows.append({
                        "step": step,
                        "scenario_id": f"a{index:03d}",
                        "anchor_index": index,
                        "target_sut": target_name,
                        "method_variant": display,
                        "response": response_name,
                        "task_kind": "failure_discovery",
                        "predicted_score": score,
                        "score_semantics": f"source_{response_name}_convex_mixture",
                        "visible_target_count": step,
                        "history_split_id": (
                            "configured_source_profiles" if source_names else
                            "source_profiles_excluding_target"
                        ),
                        "selected_reason": "stable_argmax_revealed_only",
                        "observed_event": bool(critical[index]),
                        "collision": bool(bank.collisions[target_index, index]),
                        "near_miss": bool(bank.near_misses[target_index, index]),
                        "valid": True,
                        "cumulative_cost": step,
                        "cumulative_critical": cumulative_critical,
                    })
                    alpha_rows.append({
                        "step": step,
                        "target_sut": target_name,
                        "method_variant": display,
                        "response": response_name,
                        **{f"alpha_{j}": float(value)
                           for j, value in enumerate(selector.alpha)}
                    })
                    if diagnostic:
                        diagnostic_rows.append({
                            "step": step,
                            "target_sut": target_name,
                            "method_variant": display,
                            "status": diagnostic.status,
                            "residual_norm": diagnostic.residual_norm,
                            "iterations": diagnostic.iterations,
                            "simplex_violation": diagnostic.simplex_violation,
                            "fallback": diagnostic.fallback,
                        })
                held_out = np.setdiff1d(np.arange(len(target)), selector.selected)
                prediction = selector.alpha @ sources[:, held_out]
                summary_rows.append({
                    "target_sut":
                    target_name,
                    "method_variant":
                    display,
                    "response":
                    response_name,
                    "critical_count_at_budget":
                    cumulative_critical,
                    "collision_count_at_budget":
                    int(bank.collisions[target_index, selector.selected].sum()),
                    "held_out_response_mse":
                    float(np.mean((prediction - target[held_out])**2)),
                    "final_alpha":
                    json.dumps(selector.alpha.tolist()),
                    "budget":
                    budget,
                })
    _write_rows(output / "target_query_trace.csv", trace_rows)
    _write_rows(output / "mixture_coefficients_trace.csv", alpha_rows)
    _write_rows(output / "qp_diagnostics.csv", diagnostic_rows)
    _write_rows(output / "method_summary.csv", summary_rows)
    if agreement_rows:
        _write_rows(output / "source_target_response_agreement.csv", agreement_rows)
    costs = {
        "source_bank_episodes": source_episodes,
        "oracle_target_bank_episodes": target_oracle_episodes,
        "target_reveals": len(trace_rows),
        "selector_seconds": time.perf_counter() - start
    }
    (output / "cost_ledger.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")
    _manifest(output, Path(config["_path"]), "mixture", seed, costs)


def run_dense(config: dict, output: Path) -> None:
    callback = lambda destination, costs: _manifest(
        destination,
        Path(config["_path"]),
        "dense",
        int(costs.get("run_seed", config["seed"])),
        costs,
    )
    if "batch_seeds" in config or "target_profiles" in config:
        run_dense_batch(config, output, callback)
    else:
        run_dense_protocol(config, output, callback)


def run_calibration(config: dict, output: Path) -> None:
    """Measure the declared natural event rate before adaptive testing."""
    output.mkdir(parents=True, exist_ok=True)
    scenarios = _scenario_pool(config)
    scenario_probabilities = _scenario_probabilities(config, scenarios)
    phi = np.asarray(config["natural_policy"], dtype=float)
    if not np.isclose(phi.sum(), 1.0):
        raise ValueError("natural_policy must sum to one")
    targets = [str(value) for value in config.get("target_profiles", [])]
    draws = int(config.get("calibration_draws", config.get("evaluation_draws", 256)))
    horizon = int(config["horizon"])
    seed = int(config["seed"])
    rows: list[dict] = []
    summary: list[dict] = []
    stress_rows: list[dict] = []
    for target_index, target in enumerate(targets):
        events = []
        critical_events = []
        for draw in range(draws):
            execution_seed = seed + 100000 * target_index + draw
            rng = np.random.default_rng(execution_seed)
            scenario_index = int(rng.choice(len(scenarios), p=scenario_probabilities))
            scenario = scenarios[scenario_index]
            env = DenseCutInEnv(get_profile(target), scenario)
            try:
                env.reset(seed=execution_seed)
                for _ in range(horizon):
                    action = int(rng.choice(phi.size, p=phi))
                    _, _, terminal, truncated, _ = env.step(action)
                    if terminal or truncated:
                        break
                result = env.episode_result()
            finally:
                env.close()
            events.append(result.collision)
            critical_events.append(result.collision or result.near_miss)
            rows.append({
                "target_profile": target,
                "draw": draw,
                "scenario_id": (
                    f"{scenario.mode}_g{scenario.initial_gap:g}"
                    f"_dv{scenario.relative_speed:g}"
                ).replace("-", "m"),
                "collision": result.collision,
                "near_miss": result.near_miss,
                "min_ttc": result.min_ttc,
                "min_distance": result.min_distance,
            })
        collision_rate = float(np.mean(events))
        critical_rate = float(np.mean(critical_events))
        collision_se = float(np.sqrt(collision_rate * (1.0 - collision_rate) / draws))
        summary.append({
            "target_profile": target,
            "draws": draws,
            "collisions": int(np.sum(events)),
            "collision_rate": collision_rate,
            "collision_ci95_half_width": 1.96 * collision_se,
            "critical_events": int(np.sum(critical_events)),
            "critical_event_rate": critical_rate,
        })
        if bool(config.get("stress_sweep", True)):
            for scenario_index, scenario in enumerate(scenarios):
                for action in range(phi.size):
                    execution_seed = seed + 200000 * target_index + 100 * scenario_index + action
                    env = DenseCutInEnv(get_profile(target), scenario)
                    try:
                        env.reset(seed=execution_seed)
                        for _ in range(horizon):
                            _, _, terminal, truncated, _ = env.step(action)
                            if terminal or truncated:
                                break
                        result = env.episode_result()
                    finally:
                        env.close()
                    stress_rows.append({
                        "target_profile": target,
                        "initial_gap": scenario.initial_gap,
                        "relative_speed": scenario.relative_speed,
                        "action": action,
                        "collision": result.collision,
                        "near_miss": result.near_miss,
                        "min_ttc": result.min_ttc,
                        "min_distance": result.min_distance,
                    })
    _write_rows(output / "natural_draws.csv", rows)
    _write_rows(output / "natural_event_rates.csv", summary)
    _write_rows(output / "stress_sweep.csv", stress_rows)
    costs = {
        "targets": targets,
        "draws_per_target": draws,
        "natural_episodes": len(rows),
        "stress_episodes": len(stress_rows),
        "total_episodes": len(rows) + len(stress_rows),
    }
    (output / "cost_ledger.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")
    _manifest(output, Path(config["_path"]), "calibration", seed, costs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mixture", "dense", "calibration"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    args = parser.parse_args()
    config = _load_config(args.config)
    config["_path"] = str(args.config)
    if args.stage == "mixture":
        run_mixture(config, args.output, args.bank)
    elif args.stage == "dense":
        run_dense(config, args.output)
    else:
        run_calibration(config, args.output)


if __name__ == "__main__":
    main()
