"""Compare nine selectors with 200 visible queries on the retained D bank."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from highway_sim_env.s01_parameters import coordinates
from methods.ras_frt_uq.comparison_selectors import (
    GP_INITIAL_QUERIES, GP_UCB_BETA, farthest_first, target_gp_ucb,
)
from methods.ras_frt_uq.coverage_selector import (
    TargetOracle, history_rank, select_sequence, similarities,
)
from methods.ras_frt_uq.data import ROOT, TARGET, target_labels
from methods.ras_frt_uq.fusion_selector import select_fusion_sequence
from methods.ras_frt_uq.protocol import (
    COUNT, REPEAT_SEEDS, ROOT as HISTORY_ROOT, SOURCES, read_jsonl,
    write_json,
)
from methods.ras_frt_uq.response_encoder import ResponseEncoder, predict_response
from methods.ras_frt_uq.training import historical_arrays
from methods.ras_frt_uq.transfer_uncertainty import (
    physical_kernel, select_transfer_sequence,
)


BUDGET = 200
CHECKPOINTS = (10, 30, 50, 100, 150, 200)
GRID_BINS = 4
METHODS = (
    "Random", "Farthest-First", "Target-GP-UCB", "HistoryRank-adapted",
    "History-only", "Residual risk-only", "RAS-FRT", "Transfer-UQ",
    "RAS-FRT-UQ",
)


def _target_scenes() -> list[dict]:
    scenes = read_jsonl(ROOT / "candidate_manifest.jsonl")
    if len(scenes) != COUNT:
        raise ValueError("incomplete D manifest")
    historical = read_jsonl(HISTORY_ROOT / "history_manifest.jsonl")
    new_points = {tuple(point) for point in coordinates(scenes)}
    if new_points & {tuple(point) for point in coordinates(historical)}:
        raise ValueError("D overlaps the historical A bank")
    return scenes


def _prior(seed: int, frozen: dict, target_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = HISTORY_ROOT / "models" / Path(frozen["models"][str(seed)]["path"]).name
    model = ResponseEncoder(len(SOURCES))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    responses = predict_response(model, target_x)
    return responses.mean(axis=1), responses


def _cells(x: np.ndarray, bins: int) -> np.ndarray:
    digits = np.minimum((x * bins).astype(int), bins - 1)
    return np.ravel_multi_index(digits.T, (bins,) * x.shape[1])


def _record(
    method: str, seed: int | None, selected: list[int],
    observed: list[int | None], scores: np.ndarray | None,
    cells: np.ndarray, truth: np.ndarray, scenario_ids: list[str],
) -> dict:
    found = np.cumsum(np.asarray(observed) == 1)
    discovered_cells: set[int] = set()
    cell_curve = []
    for index, label in zip(selected, observed):
        if label == 1:
            discovered_cells.add(int(cells[index]))
        cell_curve.append(len(discovered_cells))
    run = {
        "method": method, "seed": seed,
        "selected_indices": selected,
        "selected_scenario_ids": [scenario_ids[index] for index in selected],
        "observed_labels": observed,
        "failures_at_checkpoints": {str(k): int(found[k - 1]) for k in CHECKPOINTS},
        "danger_cells_at_checkpoints": {str(k): cell_curve[k - 1] for k in CHECKPOINTS},
    }
    if scores is not None:
        run["full_d_ap"] = float(average_precision_score(truth, scores))
    return run


def _summary(runs: list[dict], total_failures: int,
             total_danger_cells: int) -> dict:
    values = defaultdict(list)
    for run in runs:
        values[run["method"]].append(run)

    def stats(numbers: list[float]) -> dict:
        return {
            "mean": float(np.mean(numbers)),
            "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else 0.0,
            "values": numbers,
        }

    return {
        method: {
            "runs": len(group),
            "failures": {str(k): stats([r["failures_at_checkpoints"][str(k)]
                                        for r in group]) for k in CHECKPOINTS},
            "query_collision_rate": {
                str(k): stats([r["failures_at_checkpoints"][str(k)] / k
                               for r in group]) for k in CHECKPOINTS
            },
            "collision_recall": {
                str(k): stats([r["failures_at_checkpoints"][str(k)] / total_failures
                               for r in group]) for k in CHECKPOINTS
            },
            "danger_cells": {str(k): stats([r["danger_cells_at_checkpoints"][str(k)]
                                            for r in group]) for k in CHECKPOINTS},
            "danger_cell_coverage_rate": {
                str(k): stats([
                    r["danger_cells_at_checkpoints"][str(k)] / total_danger_cells
                    for r in group
                ]) for k in CHECKPOINTS
            },
            "full_d_ap": stats([r["full_d_ap"] for r in group])
            if "full_d_ap" in group[0] else None,
        }
        for method, group in values.items()
    }


def _grid_sensitivity(runs: list[dict], x: np.ndarray,
                      truth: np.ndarray) -> dict:
    result = {}
    for bins in (3, 4, 5):
        cells = _cells(x, bins)
        grouped = defaultdict(list)
        for run in runs:
            for checkpoint in (100, 200):
                found = [index for index in run["selected_indices"][:checkpoint]
                         if truth[index] == 1]
                grouped[run["method"], checkpoint].append(len(set(cells[found])))
        result[str(bins)] = {
            "all_danger_cells": len(set(cells[truth == 1])),
            "found_cells": {
                method: {str(k): float(np.mean(grouped[method, k]))
                         for k in (100, 200)}
                for method in METHODS
            },
        }
    return result


def main() -> None:
    scenes = _target_scenes()
    truth = target_labels(scenes)
    target_x = coordinates(scenes)
    cells = _cells(target_x, GRID_BINS)
    scenario_ids = [scene["scenario_id"] for scene in scenes]
    frozen = json.loads((HISTORY_ROOT / "frozen_training.json").read_text(
        encoding="utf-8"))
    chosen = frozen["chosen_similarity"]
    history_x, _, history_valid, history_risk = historical_arrays()
    rank = history_rank(
        history_x, history_risk, history_valid, target_x,
        frozen["chosen_history_rank"]["neighbors"],
    )
    kernel = physical_kernel(target_x)
    runs = []

    def add(method: str, seed: int | None, selector) -> None:
        oracle = TargetOracle(truth)
        selected, observed, scores = selector(oracle)
        runs.append(_record(method, seed, selected, observed, scores,
                            cells, truth, scenario_ids))
        print("selected", method, seed, runs[-1]["failures_at_checkpoints"]["200"],
              flush=True)

    add("HistoryRank-adapted", None, lambda oracle: select_sequence(
        "HistoryRank-adapted", oracle, COUNT, rank, None, BUDGET,
    ))
    for seed in REPEAT_SEEDS:
        add("Random", seed, lambda oracle: select_sequence(
            "Random", oracle, COUNT, None, None, BUDGET, random_seed=seed,
        ))
        add("Farthest-First", seed, lambda oracle: farthest_first(
            target_x, oracle, BUDGET, seed,
        ))
        add("Target-GP-UCB", seed, lambda oracle: target_gp_ucb(
            kernel, oracle, BUDGET, seed,
        ))
        prior, responses = _prior(seed, frozen, target_x)
        similarity = similarities(target_x, responses, chosen["sigma_x"],
                                  chosen["sigma_r"])
        for method in ("History-only", "Residual risk-only", "RAS-FRT"):
            add(method, seed, lambda oracle, name=method: select_sequence(
                name, oracle, COUNT, prior, similarity, BUDGET,
                regularizer=chosen["regularizer"],
                coverage_weight=frozen["coverage_weight"],
            ))
        add("Transfer-UQ", seed, lambda oracle: select_transfer_sequence(
            prior, kernel, oracle, BUDGET,
        ))
        add("RAS-FRT-UQ", seed, lambda oracle: select_fusion_sequence(
            prior, similarity, kernel, cells, oracle, BUDGET,
            chosen["regularizer"],
        ))

    protocol_path = ROOT / "budget_200_protocol.json"
    protocol = {
        "task": "A-to-D S01 FVDM collision discovery with 200 visible queries",
        "evaluation_mode": "retrospective replay on the existing D bank",
        "development_status": "fusion settings were explored after viewing D results",
        "candidate_count": COUNT,
        "target_build_id": TARGET,
        "budget_per_run": BUDGET,
        "checkpoints": CHECKPOINTS,
        "repeat_seeds": REPEAT_SEEDS,
        "methods": METHODS,
        "grid_bins_per_axis": GRID_BINS,
        "target_gp_ucb": {
            "initial_random_queries": GP_INITIAL_QUERIES,
            "exploration_weight": GP_UCB_BETA,
            "score": "unclipped GP posterior mean plus weight times posterior standard deviation",
        },
        "historical_settings": "frozen on A with a 100-query pseudo-target budget",
        "fusion_settings": "fixed after earlier 100-query development on this same D bank",
    }
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    write_json(ROOT / "budget_200_method_replay.json", {"runs": runs})
    write_json(ROOT / "budget_200_method_evaluation.json", {
        "candidate_count": COUNT,
        "target_failures": int(truth.sum()),
        "target_collision_rate": float(truth.mean()),
        "danger_cells": len(set(cells[truth == 1])),
        "methods": _summary(runs, int(truth.sum()),
                            len(set(cells[truth == 1]))),
        "grid_sensitivity": _grid_sensitivity(runs, target_x, truth),
    })


if __name__ == "__main__":
    main()
