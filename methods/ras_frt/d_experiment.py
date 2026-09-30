"""Compare historical transfer selectors on the frozen S01 D bank."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from highway_sim_env.s01_parameters import coordinates, valid_label
from highway_sim_env.build_spec import BuildSpec
from methods.ras_frt.coverage_selector import (
    TargetOracle, history_rank, select_sequence, similarities,
)
from methods.ras_frt.protocol import (
    BUDGET, COUNT, REPEAT_SEEDS, ROOT as HISTORY_ROOT, SOURCES, digest,
    read_jsonl, write_json, write_jsonl,
)
from methods.ras_frt.response_encoder import ResponseEncoder, predict_response
from methods.ras_frt.training import historical_arrays
from methods.ras_frt.transfer_uncertainty import (
    KERNEL_LENGTH_SCALE, MISSED_FAILURE_WEIGHT, OBSERVATION_NOISE,
    RISK_WEIGHT, INFORMATION_WEIGHT, physical_kernel, select_transfer_sequence,
)
from sut_algorithms.highway_env.idm_profiles import SUTProfile


ROOT = Path(
    "results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation"
)
TARGET = "fvdm_safety_speed_23_mps"
METHODS = ("Random", "HistoryRank-adapted", "RAS-FRT", "Transfer-UQ")
CHECKPOINTS = (10, 30, 50, 100)
FAILURE_GRID_BINS_PER_AXIS = 4
REPLAY_PATH = ROOT / "method_replay.json"
PROTOCOL_PATH = ROOT / "method_protocol.json"
EVALUATION_PATH = ROOT / "method_evaluation.json"
FAILURES_PATH = ROOT / "all_fvdm_dangerous_scenarios.jsonl"
METHOD_CODE = {
    "response_encoder": Path("methods/ras_frt/response_encoder.py"),
    "coverage_selector": Path("methods/ras_frt/coverage_selector.py"),
    "transfer_uncertainty": Path("methods/ras_frt/transfer_uncertainty.py"),
    "d_experiment": Path("methods/ras_frt/d_experiment.py"),
}
FROZEN_EXPERIMENT = ROOT / "source_snapshot" / "d_experiment.py"
TARGET_PROFILE = SUTProfile(
    "fvdm_safety_speed_23_mps", "FVDM", max_brake=8.0,
    desired_gap=8.0, target_speed=23.0, fvdm_sensitivity=0.6,
    fvdm_velocity_gain=1.0, fvdm_transition_gap=8.0,
)
TARGET_SPEC = BuildSpec(
    TARGET, "profiled_fvdm", None, "legacy_profile", "Profiled-FVDM",
    20.0, profile=TARGET_PROFILE.__dict__.copy(),
)


def _truth_and_rows(scenarios: list[dict]) -> tuple[np.ndarray, list[dict]]:
    rows = read_jsonl(ROOT / f"{TARGET}.jsonl")
    if len(rows) != COUNT or len(scenarios) != COUNT or any(
        row["scenario_id"] != scene["scenario_id"] or
        row["build_id"] != TARGET or
        row["build_fingerprint"] != TARGET_SPEC.fingerprint
        for row, scene in zip(rows, scenarios)
    ):
        raise ValueError("D target records do not match the frozen scene bank")
    labels = np.asarray([
        -1 if valid_label(row) is None else valid_label(row) for row in rows
    ], dtype=np.int8)
    if np.any(labels < 0):
        raise ValueError("D target bank contains an inconclusive response")
    return labels, rows


def _model(seed: int, frozen: dict) -> ResponseEncoder:
    entry = frozen["models"][str(seed)]
    path = Path(entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError("historical response model changed")
    model = ResponseEncoder(len(SOURCES))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def freeze_protocol() -> None:
    scenarios = read_jsonl(ROOT / "candidate_manifest.jsonl")
    if len(scenarios) != COUNT:
        raise ValueError("D candidate manifest is incomplete")
    historical_scenarios = read_jsonl(HISTORY_ROOT / "history_manifest.jsonl")
    history_coordinates = {tuple(map(float, point))
                          for point in coordinates(historical_scenarios)}
    target_coordinates = {tuple(map(float, point))
                          for point in coordinates(scenarios)}
    if len(historical_scenarios) != COUNT or \
            len(history_coordinates & target_coordinates) != 0:
        raise ValueError("A and D must be complete independent coordinate sets")
    target_rows = read_jsonl(ROOT / f"{TARGET}.jsonl")
    if len(target_rows) != COUNT or any(
        row["build_fingerprint"] != TARGET_SPEC.fingerprint
        for row in target_rows
    ):
        raise ValueError("D target build differs from the frozen profile")
    historical = json.loads((HISTORY_ROOT / "frozen_training.json").read_text(
        encoding="utf-8"))
    for entry in historical["models"].values():
        if digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("historical model checkpoint changed")
    write_json(PROTOCOL_PATH, {
        "purpose": "budgeted discovery and diversity comparison on D",
        "story": "use historical IDM responses from A to find as many diverse FVDM failures on D as possible under a fixed query budget",
        "evaluation_mode": "retrospective offline oracle replay on a fully executed D benchmark",
        "label_isolation": "full D labels are used only by the evaluator; selectors receive one queried label at a time",
        "selection_settings_source": "RAS-FRT settings frozen on A; Transfer-UQ constants frozen in its method definition; no method weights tuned on D",
        "history_root": str(HISTORY_ROOT),
        "target_root": str(ROOT),
        "target_build": TARGET,
        "target_profile": TARGET_SPEC.profile,
        "candidate_count": COUNT,
        "budget_per_run": BUDGET,
        "repeat_seeds": list(REPEAT_SEEDS),
        "methods": list(METHODS),
        "checkpoints": list(CHECKPOINTS),
        "target_bank_sha256": digest(ROOT / f"{TARGET}.jsonl"),
        "candidate_manifest_sha256": digest(ROOT / "candidate_manifest.jsonl"),
        "target_generation_protocol_sha256": digest(ROOT / "protocol.json"),
        "history_manifest_sha256": digest(HISTORY_ROOT / "history_manifest.jsonl"),
        "history_training_sha256": digest(HISTORY_ROOT / "frozen_training.json"),
        "history_bank_sha256": {
            build: digest(HISTORY_ROOT / "banks" / f"{build}.jsonl")
            for build in SOURCES
        },
        "method_code_sha256": {
            name: digest(FROZEN_EXPERIMENT if name == "d_experiment" else path)
            for name, path in METHOD_CODE.items()
        },
        "A_D_exact_coordinate_overlap": 0,
        "history_sources": list(SOURCES),
        "history_labels_on_D_used_for_selection": False,
        "selector_visibility": "target outcomes revealed only through one-query oracle",
        "diversity_metric": {
            "kind": "occupied cells among discovered true failures",
            "bins_per_axis": FAILURE_GRID_BINS_PER_AXIS,
            "axes": "four normalized S01 scenario parameters",
        },
        "transfer_uq": {
            "kernel_length_scale": KERNEL_LENGTH_SCALE,
            "observation_noise_variance": OBSERVATION_NOISE,
            "risk_weight": RISK_WEIGHT,
            "missed_failure_weight": MISSED_FAILURE_WEIGHT,
            "information_weight": INFORMATION_WEIGHT,
        },
    })


def _verify_frozen_protocol(protocol: dict) -> None:
    if protocol["target_bank_sha256"] != digest(ROOT / f"{TARGET}.jsonl") or \
            protocol["candidate_manifest_sha256"] != digest(
                ROOT / "candidate_manifest.jsonl") or \
            protocol["target_generation_protocol_sha256"] != digest(
                ROOT / "protocol.json") or \
            protocol["history_manifest_sha256"] != digest(
                HISTORY_ROOT / "history_manifest.jsonl") or \
            protocol["history_training_sha256"] != digest(
                HISTORY_ROOT / "frozen_training.json"):
        raise ValueError("A or D data changed after method protocol freeze")
    for build, expected in protocol["history_bank_sha256"].items():
        if digest(HISTORY_ROOT / "banks" / f"{build}.jsonl") != expected:
            raise ValueError(f"historical bank changed after freeze: {build}")
    for name, expected in protocol["method_code_sha256"].items():
        # The recorded runner predates the shared-module relocation.
        path = FROZEN_EXPERIMENT if name == "d_experiment" else METHOD_CODE[name]
        if digest(path) != expected:
            raise ValueError(f"method implementation changed after freeze: {name}")


def _replay_one(method: str, seed: int, truth: np.ndarray,
                scenario_ids: list[str], prior: np.ndarray | None,
                similarity: np.ndarray | None, kernel: np.ndarray | None,
                frozen: dict) -> tuple[dict, np.ndarray | None]:
    oracle = TargetOracle(truth)
    if method == "Transfer-UQ":
        selected, observed, score = select_transfer_sequence(
            prior, kernel, oracle, BUDGET,
        )
    else:
        selected, observed, score = select_sequence(
            method, oracle, COUNT, prior, similarity, BUDGET,
            regularizer=frozen["chosen_similarity"]["regularizer"],
            coverage_weight=frozen["coverage_weight"],
            random_seed=seed,
        )
    if len(selected) != BUDGET or len(set(selected)) != BUDGET or \
            oracle.queried != set(selected):
        raise ValueError("query budget or target-label isolation failed")
    discovery = np.cumsum([label == 1 for label in observed]).tolist()
    run = {
        "method": method,
        "seed": seed,
        "selected_indices": selected,
        "selected_scenario_ids": [scenario_ids[index] for index in selected],
        "observed_labels": observed,
        "discovery_curve": discovery,
        "checkpoints": {str(k): discovery[k - 1] for k in CHECKPOINTS},
    }
    return run, score


def replay() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _verify_frozen_protocol(protocol)
    scenarios = read_jsonl(ROOT / "candidate_manifest.jsonl")
    target_path = ROOT / f"{TARGET}.jsonl"
    if digest(target_path) != protocol["target_bank_sha256"] or \
            digest(ROOT / "candidate_manifest.jsonl") != protocol[
                "candidate_manifest_sha256"]:
        raise ValueError("D bank changed after method protocol freeze")
    truth, _ = _truth_and_rows(scenarios)
    frozen = json.loads((HISTORY_ROOT / "frozen_training.json").read_text(
        encoding="utf-8"))
    x_target = coordinates(scenarios)
    x_history, _, valid_history, historical_risk = historical_arrays()
    rank_score = history_rank(
        x_history, historical_risk, valid_history, x_target,
        frozen["chosen_history_rank"]["neighbors"],
    )
    scenario_ids = [scene["scenario_id"] for scene in scenarios]
    runs = []
    scores = {}
    run, score = _replay_one(
        "HistoryRank-adapted", REPEAT_SEEDS[0], truth, scenario_ids,
        rank_score, None, None, frozen,
    )
    runs.append(run)
    scores["HistoryRank-adapted"] = score
    kernel = physical_kernel(x_target)
    for seed in REPEAT_SEEDS:
        run, _ = _replay_one(
            "Random", seed, truth, scenario_ids, None, None, None, frozen,
        )
        runs.append(run)
        response = predict_response(_model(seed, frozen), x_target)
        prior = response.mean(axis=1)
        similarity_config = frozen["chosen_similarity"]
        similarity = similarities(
            x_target, response, similarity_config["sigma_x"],
            similarity_config["sigma_r"],
        )
        run, score = _replay_one(
            "RAS-FRT", seed, truth, scenario_ids, prior,
            similarity, None, frozen,
        )
        runs.append(run)
        scores[f"RAS-FRT_{seed}"] = score
        run, score = _replay_one(
            "Transfer-UQ", seed, truth, scenario_ids, prior,
            None, kernel, frozen,
        )
        runs.append(run)
        runs[-1]["historical_model_seed"] = seed
        scores[f"Transfer-UQ_{seed}"] = score
        print("replayed", seed, flush=True)
    prediction_root = ROOT / "method_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    for name, score in scores.items():
        np.save(prediction_root / f"{name.replace('-', '_')}.npy", score)
    write_json(REPLAY_PATH, {
        "protocol_sha256": digest(PROTOCOL_PATH),
        "target_bank_sha256": digest(target_path),
        "candidate_manifest_sha256": digest(ROOT / "candidate_manifest.jsonl"),
        "target_failure_count": int(np.sum(truth == 1)),
        "valid_count": int(np.sum(truth >= 0)),
        "runs": runs,
    })


def _cell(coordinate: np.ndarray) -> tuple[int, ...]:
    return tuple(min(int(value * FAILURE_GRID_BINS_PER_AXIS),
                     FAILURE_GRID_BINS_PER_AXIS - 1)
                 for value in coordinate)


def evaluate() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _verify_frozen_protocol(protocol)
    replay_data = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    target_path = ROOT / f"{TARGET}.jsonl"
    if replay_data["protocol_sha256"] != digest(PROTOCOL_PATH) or \
            replay_data["target_bank_sha256"] != digest(target_path) or \
            replay_data["candidate_manifest_sha256"] != digest(
                ROOT / "candidate_manifest.jsonl"):
        raise ValueError("replay does not match the frozen D protocol")
    scenarios = read_jsonl(ROOT / "candidate_manifest.jsonl")
    truth, target_rows = _truth_and_rows(scenarios)
    coords = coordinates(scenarios)
    write_jsonl(FAILURES_PATH, [
        {"scenario": scenarios[index], "fvdm_response": target_rows[index]}
        for index in np.flatnonzero(truth == 1)
    ])
    dangerous_cells = {
        _cell(coords[index]) for index in np.flatnonzero(truth == 1)
    }
    expected_runs = {"HistoryRank-adapted": 1, "Random": 5,
                     "RAS-FRT": 5, "Transfer-UQ": 5}
    actual = defaultdict(int)
    for run in replay_data["runs"]:
        actual[run["method"]] += 1
    if dict(actual) != expected_runs:
        raise ValueError("replay is missing a planned method or seed")

    grouped = defaultdict(list)
    for run in replay_data["runs"]:
        selected = run["selected_indices"]
        queried = [int(truth[index]) for index in selected]
        if queried != run["observed_labels"] or len(selected) != BUDGET or \
                len(set(selected)) != BUDGET:
            raise ValueError("recorded query feedback differs from oracle labels")
        discovered_cells = set()
        unique_cell_curve = []
        failures = 0
        for position, index in enumerate(selected, 1):
            if truth[index] == 1:
                failures += 1
                discovered_cells.add(_cell(coords[index]))
            if position in CHECKPOINTS:
                unique_cell_curve.append({
                    "budget": position,
                    "failures_found": failures,
                    "unique_danger_cells_found": len(discovered_cells),
                    "danger_cell_recall": len(discovered_cells) /
                    max(1, len(dangerous_cells)),
                })
        run["unique_danger_cells_curve"] = unique_cell_curve
        method = run["method"]
        if method != "Random":
            key = (f"{method}_{run['historical_model_seed']}"
                   if method == "Transfer-UQ" else
                   f"{method}_{run['seed']}" if method == "RAS-FRT" else method)
            score_path = ROOT / "method_predictions" / f"{key.replace('-', '_')}.npy"
            score = np.load(score_path)
            if score.shape != (COUNT,) or not np.all(np.isfinite(score)):
                raise ValueError("invalid full-D risk score")
            run["average_precision"] = float(average_precision_score(truth, score))
        grouped[method].append(run)

    summary = {}
    for method, runs in grouped.items():
        summary[method] = {
            "failures_at_checkpoints": {
                str(k): {
                    "mean": float(np.mean([r["discovery_curve"][k - 1]
                                            for r in runs])),
                    "std": float(np.std([r["discovery_curve"][k - 1]
                                         for r in runs], ddof=1))
                    if len(runs) > 1 else 0.0,
                } for k in CHECKPOINTS
            },
            "unique_danger_cells_at_checkpoints": {
                str(k): {
                    "mean": float(np.mean([
                        r["unique_danger_cells_curve"][CHECKPOINTS.index(k)][
                            "unique_danger_cells_found"] for r in runs])),
                    "std": float(np.std([
                        r["unique_danger_cells_curve"][CHECKPOINTS.index(k)][
                            "unique_danger_cells_found"] for r in runs], ddof=1))
                    if len(runs) > 1 else 0.0,
                } for k in CHECKPOINTS
            },
            "average_precision": {
                "mean": float(np.mean([r["average_precision"] for r in runs
                                        if "average_precision" in r])),
                "std": float(np.std([r["average_precision"] for r in runs
                                      if "average_precision" in r], ddof=1))
                if len([r for r in runs if "average_precision" in r]) > 1 else 0.0,
            } if method != "Random" else None,
            "values_at_100": [r["discovery_curve"][-1] for r in runs],
            "first_failure_query": [next(
                (i for i, label in enumerate(r["observed_labels"], 1)
                 if label == 1), None,
            ) for r in runs],
        }
    result = {
        "candidate_count": COUNT,
        "valid_target_count": int(np.sum(truth >= 0)),
        "all_fvdm_dangerous_scenarios": int(np.sum(truth == 1)),
        "fvdm_failure_rate": float(np.mean(truth == 1)),
        "all_danger_parameter_cells": len(dangerous_cells),
        "grid_bins_per_axis": FAILURE_GRID_BINS_PER_AXIS,
        "methods": summary,
    }
    write_json(EVALUATION_PATH, result)
    return result


def main() -> None:
    if not PROTOCOL_PATH.exists():
        freeze_protocol()
    if not REPLAY_PATH.exists():
        replay()
    result = evaluate()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
