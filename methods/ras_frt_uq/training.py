"""Five historical pseudo-target folds and frozen full-history models."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from highway_sim_env.s01_parameters import coordinates
from methods.ras_frt_uq.banks import labels
from methods.ras_frt_uq.coverage_selector import (
    TargetOracle, history_rank, select_sequence, similarities,
)
from methods.ras_frt_uq.protocol import (
    BUDGET, REPEAT_SEEDS, ROOT, SOURCES, historical_manifest, write_json,
)
from methods.ras_frt_uq.response_encoder import (
    fit_full_history, fit_response, predict_response,
)


SIGMA_X = (0.15, 0.3, 0.6)
SIGMA_R = (None, 0.1, 0.25, 0.5)
REGULARIZERS = (0.1, 1.0, 5.0)
NEIGHBORS = (5, 10, 20)
SPLIT_SEED = 20260929


def historical_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scenarios = historical_manifest()
    x = coordinates(scenarios)
    retrieved = [labels(build, scenarios) for build in SOURCES]
    columns = [item[0] for item in retrieved]
    y = np.asarray([[-1 if value is None else value for value in column]
                    for column in columns], dtype=np.int8).T
    valid = y >= 0
    risk_columns = []
    for _, rows in retrieved:
        values = []
        for row in rows:
            ttc = row["min_ttc"]
            clearance = row["min_clearance"]
            values.append(float(row["ego_collision"] is True) +
                          (0.49 / (1 + max(0.0, float(ttc))) if ttc is not None else 0.0) +
                          (0.01 / (1 + max(0.0, float(clearance)))
                           if clearance is not None else 0.0))
        risk_columns.append(values)
    return (x, np.maximum(y, 0).astype(np.float32), valid,
            np.asarray(risk_columns, dtype=np.float32).T)


def _discovery(observed: list[int | None]) -> int:
    return sum(label == 1 for label in observed)


def _threshold(predictions: list[np.ndarray], truths: list[np.ndarray]) -> float:
    scores = np.concatenate(predictions)
    labels = np.concatenate(truths)
    valid = labels >= 0
    scores, labels = scores[valid], labels[valid]
    if not np.any(labels == 1):
        return 1.0
    best_score, best_threshold = -1.0, 0.5
    for threshold in np.linspace(0, 1, 101):
        predicted = scores >= threshold
        true_positive = np.sum(predicted & (labels == 1))
        f1 = 2 * true_positive / (np.sum(predicted) + np.sum(labels == 1))
        if f1 > best_score:
            best_score, best_threshold = f1, float(threshold)
    return best_threshold


def freeze_training() -> None:
    if (ROOT / "frozen_training.json").exists():
        raise FileExistsError("historical training for A is already frozen")
    x, y, valid, historical_risk = historical_arrays()
    permutation = np.random.default_rng(SPLIT_SEED).permutation(len(x))
    train_indices, val_indices = permutation[:1638], permutation[1638:]
    folds = []
    for held_out in range(len(SOURCES)):
        heads = [index for index in range(len(SOURCES)) if index != held_out]
        model, fit = fit_response(x, y[:, heads], valid[:, heads],
                                  train_indices, val_indices, seed=REPEAT_SEEDS[0])
        predicted = predict_response(model, x[val_indices])
        pseudo_labels = np.where(valid[val_indices, held_out],
                                 y[val_indices, held_out], -1).astype(np.int8)
        history_risk = history_rank(
            x[train_indices], historical_risk[train_indices][:, heads],
            valid[train_indices][:, heads], x[val_indices], 5,
        )
        folds.append({
            "held_out": SOURCES[held_out], "x": x[val_indices],
            "response": predicted, "prior": predicted.mean(axis=1),
            "labels": pseudo_labels, "fit": fit, "history_risk": history_risk,
            "train_risk": historical_risk[train_indices][:, heads],
            "train_valid": valid[train_indices][:, heads],
            "train_x": x[train_indices],
        })
    candidates = []
    for sigma_x, sigma_r in product(SIGMA_X, SIGMA_R):
        matrices = [similarities(fold["x"], fold["response"], sigma_x, sigma_r)
                    for fold in folds]
        for regularizer in REGULARIZERS:
            discoveries, mid_discoveries = [], []
            for fold, similarity in zip(folds, matrices):
                _, observed, _ = select_sequence(
                    "RAS-FRT", TargetOracle(fold["labels"]), len(fold["labels"]),
                    fold["prior"], similarity, BUDGET,
                    regularizer=regularizer,
                )
                discoveries.append(_discovery(observed))
                mid_discoveries.append(_discovery(observed[:50]))
            candidates.append({
                "sigma_x": sigma_x, "sigma_r": sigma_r,
                "regularizer": regularizer, "discoveries": discoveries,
                "mean_discovery": float(np.mean(discoveries)),
                "mean_discovery_at_50": float(np.mean(mid_discoveries)),
            })
    chosen = max(candidates, key=lambda item: (
        item["mean_discovery"], item["mean_discovery_at_50"]))
    rank_candidates = []
    for k in NEIGHBORS:
        discoveries, mid_discoveries = [], []
        for fold in folds:
            score = history_rank(fold["train_x"], fold["train_risk"],
                                 fold["train_valid"], fold["x"], k)
            _, observed, _ = select_sequence(
                "HistoryRank-adapted", TargetOracle(fold["labels"]),
                len(fold["labels"]), score, None, BUDGET,
            )
            discoveries.append(_discovery(observed))
            mid_discoveries.append(_discovery(observed[:50]))
        rank_candidates.append({"neighbors": k, "discoveries": discoveries,
                                "mean_discovery": float(np.mean(discoveries)),
                                "mean_discovery_at_50": float(np.mean(mid_discoveries))})
    chosen_rank = max(rank_candidates, key=lambda item: (
        item["mean_discovery"], item["mean_discovery_at_50"]))
    validation_predictions = {method: [] for method in (
        "HistoryRank-adapted", "History-only", "Residual risk-only", "RAS-FRT")}
    validation_truths = []
    fold_diagnostics = []
    for fold in folds:
        similarity = similarities(fold["x"], fold["response"],
                                  chosen["sigma_x"], chosen["sigma_r"])
        rank_score = history_rank(fold["train_x"], fold["train_risk"],
                                  fold["train_valid"], fold["x"],
                                  chosen_rank["neighbors"])
        validation_truths.append(fold["labels"])
        discovered = {}
        for method in validation_predictions:
            score = rank_score if method == "HistoryRank-adapted" else fold["prior"]
            _, observed, risk = select_sequence(
                method, TargetOracle(fold["labels"]), len(fold["labels"]),
                score, similarity, BUDGET,
                regularizer=chosen["regularizer"],
            )
            validation_predictions[method].append(risk)
            discovered[method] = _discovery(observed)
        true = fold["labels"]
        fold_diagnostics.append({
            "pseudo_target": fold["held_out"],
            "valid_count": int(np.sum(true >= 0)),
            "failure_count": int(np.sum(true == 1)),
            "fit": fold["fit"], "discovery_at_100": discovered,
            "prior_ap": float(average_precision_score(true[true >= 0],
                                                       fold["prior"][true >= 0]))
            if np.any(true == 1) else None,
        })
    thresholds = {method: _threshold(scores, validation_truths)
                  for method, scores in validation_predictions.items()}
    epochs = int(np.median([fold["fit"]["best_epoch"] for fold in folds]))
    models = {}
    for seed in REPEAT_SEEDS:
        model = fit_full_history(x, y, valid, seed, epochs)
        path = ROOT / "models" / f"response_seed_{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.cpu().state_dict(), path)
        models[str(seed)] = {"path": str(path)}
    disagreement = {}
    for i in range(len(SOURCES)):
        for j in range(i + 1, len(SOURCES)):
            both = valid[:, i] & valid[:, j]
            disagreement[f"{SOURCES[i]}__{SOURCES[j]}"] = float(
                np.mean(y[both, i] != y[both, j])) if np.any(both) else None
    write_json(ROOT / "frozen_training.json", {
        "split_seed": SPLIT_SEED, "train_count": len(train_indices),
        "validation_count": len(val_indices), "pseudo_target_folds": fold_diagnostics,
        "source_failure_counts": {build: int(np.sum(y[:, j] * valid[:, j]))
                                  for j, build in enumerate(SOURCES)},
        "source_disagreement": disagreement,
        "similarity_candidates": candidates,
        "chosen_similarity": chosen,
        "history_rank_candidates": rank_candidates,
        "chosen_history_rank": chosen_rank,
        "thresholds": thresholds,
        "coverage_weight": 0.2,
        "final_training_epochs": epochs,
        "models": models,
        "target_labels_used": False,
    })
