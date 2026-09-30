"""Evaluate uncertainty-enhanced RAS-FRT on the retained S01 D bank."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from highway_sim_env.s01_parameters import coordinates
from methods.ras_frt.coverage_selector import TargetOracle, similarities
from methods.ras_frt.d_experiment import (
    FAILURE_GRID_BINS_PER_AXIS, PROTOCOL_PATH as BASE_PROTOCOL_PATH,
    ROOT, _model, _truth_and_rows, _verify_frozen_protocol,
)
from methods.ras_frt.fusion_selector import (
    COVERAGE_WEIGHT, INFORMATION_WEIGHT, NEW_CELL_BONUS,
    select_fusion_sequence,
)
from methods.ras_frt.protocol import (
    BUDGET, REPEAT_SEEDS, ROOT as HISTORY_ROOT, digest, read_jsonl,
    write_json,
)
from methods.ras_frt.response_encoder import predict_response
from methods.ras_frt.transfer_uncertainty import (
    KERNEL_LENGTH_SCALE, OBSERVATION_NOISE, physical_kernel,
)


NAME = "RAS-FRT-UQ"
PROTOCOL_PATH = ROOT / "fusion_protocol.json"
REPLAY_PATH = ROOT / "fusion_replay.json"
EVALUATION_PATH = ROOT / "fusion_evaluation.json"
PREDICTION_ROOT = ROOT / "fusion_predictions"
CODE_PATHS = (
    Path("methods/ras_frt/fusion_selector.py"),
    Path("methods/ras_frt/fusion_experiment.py"),
)
FROZEN_EXPERIMENT = ROOT / "source_snapshot" / "fusion_experiment.py"
CHECKPOINTS = (10, 30, 50, 100)


def _cells(x: np.ndarray, bins: int) -> np.ndarray:
    columns = np.minimum((x * bins).astype(int), bins - 1).T
    result = np.zeros(len(x), dtype=np.int32)
    for column in columns:
        result = result * bins + column
    return result


def _source_protocol() -> dict:
    protocol = json.loads(BASE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    _verify_frozen_protocol(protocol)
    return protocol


def _fusion_protocol() -> dict:
    return {
        "method": NAME,
        "task": "S01 A-to-D budgeted FVDM collision discovery",
        "evaluation_mode": "retrospective offline oracle replay",
        "development_status": "fusion mechanism and weights explored on the already labeled D bank; this is not an independent blind confirmation",
        "selector_visibility": "one queried FVDM outcome at a time",
        "budget": BUDGET,
        "seeds": list(REPEAT_SEEDS),
        "checkpoints": list(CHECKPOINTS),
        "base_protocol_sha256": digest(BASE_PROTOCOL_PATH),
        "target_bank_sha256": _source_protocol()["target_bank_sha256"],
        "history_training_sha256": digest(HISTORY_ROOT / "frozen_training.json"),
        "selector_code_sha256": {
            str(path): digest(FROZEN_EXPERIMENT if path.name == "fusion_experiment.py" else path)
            for path in CODE_PATHS
        },
        "settings": {
            "risk_blend": "posterior variance times local RAS-FRT risk plus its complement times Gaussian residual risk",
            "residual_kernel": "physical RBF",
            "kernel_length_scale": KERNEL_LENGTH_SCALE,
            "observation_noise": OBSERVATION_NOISE,
            "local_similarity": "A-frozen physical plus historical-response similarity",
            "local_regularizer": "A-frozen chosen_similarity.regularizer",
            "coverage_weight": COVERAGE_WEIGHT,
            "new_cell_bonus": NEW_CELL_BONUS,
            "information_weight": INFORMATION_WEIGHT,
            "cell_bins_per_axis": FAILURE_GRID_BINS_PER_AXIS,
            "final_risk_score": "Gaussian residual posterior risk after 100 observations",
        },
    }


def _verify_protocol() -> None:
    expected = _fusion_protocol()
    actual = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("fusion protocol, code, or input data changed")


def replay() -> None:
    _source_protocol()
    if not PROTOCOL_PATH.exists():
        write_json(PROTOCOL_PATH, _fusion_protocol())
    _verify_protocol()
    scenarios = read_jsonl(ROOT / "candidate_manifest.jsonl")
    truth, _ = _truth_and_rows(scenarios)
    x = coordinates(scenarios)
    cells = _cells(x, FAILURE_GRID_BINS_PER_AXIS)
    kernel = physical_kernel(x)
    frozen = json.loads((HISTORY_ROOT / "frozen_training.json")
                        .read_text(encoding="utf-8"))
    settings = frozen["chosen_similarity"]
    scenario_ids = [scene["scenario_id"] for scene in scenarios]
    runs = []
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)

    for seed in REPEAT_SEEDS:
        response = predict_response(_model(seed, frozen), x)
        prior = response.mean(axis=1)
        similarity = similarities(x, response, settings["sigma_x"],
                                  settings["sigma_r"])
        oracle = TargetOracle(truth)
        selected, observed, score = select_fusion_sequence(
            prior, similarity, kernel, cells, oracle, BUDGET,
            settings["regularizer"],
        )
        if len(selected) != BUDGET or len(set(selected)) != BUDGET or \
                oracle.queried != set(selected) or \
                observed != [int(truth[index]) for index in selected]:
            raise ValueError("fusion query budget or label isolation failed")
        if score.shape != truth.shape or not np.all(np.isfinite(score)):
            raise ValueError("fusion risk scores are invalid")
        np.save(PREDICTION_ROOT / f"response_seed_{seed}.npy", score)
        discovery = np.cumsum(np.asarray(observed) == 1).tolist()
        runs.append({
            "method": NAME,
            "seed": seed,
            "selected_indices": selected,
            "selected_scenario_ids": [scenario_ids[i] for i in selected],
            "observed_labels": observed,
            "discovery_curve": discovery,
            "checkpoints": {str(k): discovery[k - 1] for k in CHECKPOINTS},
        })
        print("replayed fusion seed", seed, flush=True)

    write_json(REPLAY_PATH, {
        "protocol_sha256": digest(PROTOCOL_PATH),
        "candidate_manifest_sha256": digest(ROOT / "candidate_manifest.jsonl"),
        "target_bank_sha256": _source_protocol()["target_bank_sha256"],
        "runs": runs,
    })


def evaluate() -> dict:
    _verify_protocol()
    replay_data = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    if replay_data["protocol_sha256"] != digest(PROTOCOL_PATH) or \
            replay_data["candidate_manifest_sha256"] != digest(
                ROOT / "candidate_manifest.jsonl") or \
            replay_data["target_bank_sha256"] != _source_protocol()[
                "target_bank_sha256"]:
        raise ValueError("fusion replay does not match the frozen protocol")
    scenarios = read_jsonl(ROOT / "candidate_manifest.jsonl")
    truth, _ = _truth_and_rows(scenarios)
    x = coordinates(scenarios)
    runs = replay_data["runs"]
    if len(runs) != len(REPEAT_SEEDS) or \
            {run["seed"] for run in runs} != set(REPEAT_SEEDS):
        raise ValueError("fusion replay is missing a seed")

    cells = _cells(x, FAILURE_GRID_BINS_PER_AXIS)
    all_danger_cells = len(set(cells[truth == 1]))
    for run in runs:
        selected = run["selected_indices"]
        if run["observed_labels"] != [int(truth[i]) for i in selected]:
            raise ValueError("fusion query trace differs from D truth")
        run["cells_at_checkpoints"] = {}
        for checkpoint in CHECKPOINTS:
            found = [i for i in selected[:checkpoint] if truth[i] == 1]
            run["cells_at_checkpoints"][str(checkpoint)] = len(
                set(cells[found]))
        score = np.load(PREDICTION_ROOT /
                        f"response_seed_{run['seed']}.npy")
        unseen = np.ones(len(truth), dtype=bool)
        unseen[selected] = False
        run["full_d_ap"] = float(average_precision_score(truth, score))
        run["unqueried_ap"] = float(average_precision_score(
            truth[unseen], score[unseen]))

    def summarize(values: list[float]) -> dict:
        return {"mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)), "values": values}

    result = {
        "method": NAME,
        "candidate_count": len(truth),
        "all_fvdm_dangerous_scenarios": int(np.sum(truth == 1)),
        "all_danger_parameter_cells": all_danger_cells,
        "grid_bins_per_axis": FAILURE_GRID_BINS_PER_AXIS,
        "failures_at_checkpoints": {
            str(k): summarize([r["discovery_curve"][k - 1] for r in runs])
            for k in CHECKPOINTS
        },
        "danger_cells_at_checkpoints": {
            str(k): summarize([r["cells_at_checkpoints"][str(k)]
                               for r in runs]) for k in CHECKPOINTS
        },
        "full_d_ap": summarize([r["full_d_ap"] for r in runs]),
        "unqueried_ap": summarize([r["unqueried_ap"] for r in runs]),
        "grid_sensitivity": {},
    }
    for bins in (3, 4, 5):
        comparison_cells = _cells(x, bins)
        result["grid_sensitivity"][str(bins)] = {
            "all_danger_cells": len(set(comparison_cells[truth == 1])),
            "found_cells_at_100": summarize([
                len(set(comparison_cells[
                    [i for i in run["selected_indices"] if truth[i] == 1]]))
                for run in runs
            ]),
        }
    baseline = json.loads((ROOT / "method_evaluation.json")
                          .read_text(encoding="utf-8"))["methods"]
    result["baseline_comparison"] = {
        name: {
            "found_100": values["failures_at_checkpoints"]["100"]["mean"],
            "danger_cells_100": values[
                "unique_danger_cells_at_checkpoints"]["100"]["mean"],
            "full_d_ap": values["average_precision"]["mean"]
            if values["average_precision"] else None,
        }
        for name, values in baseline.items()
    }
    result["best_mean_at_all_reported_endpoints"] = all(
        result["failures_at_checkpoints"][str(k)]["mean"] > max(
            method["failures_at_checkpoints"][str(k)]["mean"]
            for method in baseline.values())
        for k in CHECKPOINTS
    ) and result["danger_cells_at_checkpoints"]["100"]["mean"] > max(
        method["unique_danger_cells_at_checkpoints"]["100"]["mean"]
        for method in baseline.values()
    ) and result["full_d_ap"]["mean"] > max(
        method["average_precision"]["mean"]
        for method in baseline.values() if method["average_precision"]
    )
    write_json(EVALUATION_PATH, result)
    return result


def main() -> None:
    replay()
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
