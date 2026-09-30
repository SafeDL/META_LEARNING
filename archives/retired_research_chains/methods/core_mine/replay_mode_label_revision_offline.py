"""Retrospective online-policy replay on completed controller-revision banks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from methods.core_mine.acquisition import choose
from methods.core_mine.config import SUPPORT_BUDGET
from methods.core_mine.data import CachedTask, _features, response_value
from methods.core_mine.metrics import record_metrics
from methods.core_mine.multimode20_experiment import _quantile


RESULT_ROOT = Path("results/method_chains/core_mine/studies/mode_label_revision_offline")
FAMILIES = {
    "idm": {
        "root": Path("results/method_chains/core_mine/studies/idm_revision_validation"),
        "seeds": (20300109, 20300123, 20300206),
        "targets": ("idm_delay07", "idm_brake3", "idm_delay07_brake3"),
    },
    "fvdm": {
        "root": Path("results/method_chains/core_mine/studies/fvdm_revision_validation"),
        "seeds": (20301205, 20301219, 20310109),
        "targets": ("fvdm_delay05", "fvdm_brake3", "fvdm_delay05_brake3"),
    },
}
METHODS = (
    "SourceMeanRaw-Static",
    "ModeQuantile-Static",
    "ModeLabelShift-Risk",
    "ModeUCB1",
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20330925


def _load_task(family: str, seed: int, target: str) -> tuple[CachedTask, np.ndarray]:
    root = FAMILIES[family]["root"] / str(seed)
    with np.load(root / "candidate_pool.npz", allow_pickle=False) as source, \
            np.load(root / "target_bank.npz", allow_pickle=False) as bank:
        target_index = list(bank["target_names"].astype(str)).index(target)
        anchors, modes, controls = (source[key].copy() for key in
                                    ("anchors", "modes", "controls"))
        source_collision = source["source_ego_collision"].copy()
        source_event = source_collision | source["source_near_miss"]
        source_y = response_value(
            source["source_min_ttc"], source_event, source_collision)[None, :]
        target_collision = bank["ego_collision"][target_index].copy()
        target_event = (target_collision
                        | bank["near_miss"][target_index].copy())
        target_ttc = bank["min_ttc"][target_index].copy()

    features, dimensions = _features(anchors, modes, controls)
    task = CachedTask(
        seed=seed,
        target_name=f"Target-{target}-{family}-revision",
        coverage="source_safe_revision",
        heterogeneity=target,
        anchors=anchors,
        modes=modes,
        controls=controls,
        features=features,
        active_dimensions=dimensions,
        source_y=source_y,
        source_event=source_event[None, :],
        source_collision=source_collision[None, :],
        target_y=response_value(target_ttc, target_event, target_collision),
        target_event=target_event,
        target_collision=target_collision,
        target_ttc=target_ttc,
        source_safe_target_failure=target_event.copy(),
    )
    eligible = np.arange(len(modes), dtype=int)
    return task, eligible


def _label_scores(task: CachedTask, selected: list[int],
                  labels: list[float]) -> np.ndarray:
    source = task.source_y.mean(axis=0)
    scores = source.copy()
    if not selected:
        return scores
    observed = np.asarray(selected, dtype=int)
    residual = np.asarray(labels, dtype=float) - source[observed]
    for mode in np.unique(task.modes[observed]):
        mask = task.modes[observed] == mode
        scores[task.modes == mode] += residual[mask].mean()
    return scores


def _ucb_mode(task: CachedTask, selected: list[int], labels: list[float],
              eligible: np.ndarray) -> str:
    mode_order = list(dict.fromkeys(task.modes.astype(str).tolist()))
    counts = {mode: 0 for mode in mode_order}
    rewards = {mode: 0.0 for mode in mode_order}
    for index, label in zip(selected, labels, strict=True):
        mode = str(task.modes[index])
        counts[mode] += 1
        rewards[mode] += label

    remaining = eligible[~np.isin(eligible, np.asarray(selected, dtype=int))]
    available_modes = [mode for mode in mode_order
                       if np.any(task.modes[remaining] == mode)]
    untried = [mode for mode in available_modes if counts[mode] == 0]
    if untried:
        return untried[0]
    total = len(selected)
    return max(available_modes, key=lambda mode: (
        rewards[mode] / counts[mode]
        + math.sqrt(2.0 * math.log(total) / counts[mode]),
        -mode_order.index(mode),
    ))


def _choose(method: str, task: CachedTask, eligible: np.ndarray,
            selected: list[int], labels: list[float]) -> int:
    source = task.source_y.mean(axis=0)
    if method == "SourceMeanRaw-Static":
        return choose(source, selected, task.modes, SUPPORT_BUDGET,
                      allowed_indices=eligible)
    if method == "ModeQuantile-Static":
        return choose(_quantile(task, eligible), selected, task.modes,
                      SUPPORT_BUDGET, allowed_indices=eligible)
    if method == "ModeLabelShift-Risk":
        return choose(_label_scores(task, selected, labels), selected,
                      task.modes, SUPPORT_BUDGET, allowed_indices=eligible)
    if method == "ModeUCB1":
        if len(selected) < SUPPORT_BUDGET:
            return choose(source, selected, task.modes, SUPPORT_BUDGET,
                          allowed_indices=eligible)
        mode = _ucb_mode(task, selected, labels, eligible)
        available = eligible[
            (task.modes[eligible] == mode)
            & ~np.isin(eligible, np.asarray(selected, dtype=int))]
        return int(available[np.lexsort((available, -source[available]))[0]])
    raise ValueError(method)


def _replay(method: str, task: CachedTask, eligible: np.ndarray) -> dict:
    selected: list[int] = []
    labels: list[float] = []
    for _ in range(50):
        index = _choose(method, task, eligible, selected, labels)
        event = bool(task.target_event[index])
        collision = bool(task.target_collision[index])
        selected.append(index)
        labels.append(0.5 * float(event) + 0.5 * float(collision))
    if len(set(selected)) != 50:
        raise RuntimeError("offline replay did not select 50 distinct cases")
    metrics = record_metrics(task, selected, 50)
    events = task.target_event[np.asarray(selected, dtype=int)]
    metrics["EarlyNovelAUC"] = float(np.cumsum(events).sum() / 1275.0)
    metrics["selected_indices"] = selected
    metrics["pool_failures"] = int(task.target_event.sum())
    metrics["pool_collisions"] = int(task.target_collision.sum())
    return metrics


def _paired_bootstrap(units: list[dict], method: str,
                      comparator: str) -> dict:
    values = {(unit["family"], unit["seed"], unit["target"], unit["method"]): unit
              for unit in units}
    differences = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for family in FAMILIES:
        family_seeds = list(FAMILIES[family]["seeds"])
        family_targets = list(FAMILIES[family]["targets"])
        matrix = np.asarray([
            [values[(family, seed, target, method)]["NewHistoricalFailureCount"]
             - values[(family, seed, target, comparator)]["NewHistoricalFailureCount"]
             for target in family_targets]
            for seed in family_seeds], dtype=float)
        samples = np.empty(BOOTSTRAP_DRAWS)
        for draw in range(BOOTSTRAP_DRAWS):
            sampled_seeds = rng.integers(0, len(family_seeds), len(family_seeds))
            sampled_targets = rng.integers(
                0, len(family_targets), (len(family_seeds), len(family_targets)))
            samples[draw] = matrix[sampled_seeds[:, None], sampled_targets].mean()
        differences[family] = {
            "mean_new_failures_difference": float(matrix.mean()),
            "seed_cluster_bootstrap_95": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975))],
            "paired_seed_target_differences": matrix.tolist(),
        }
    return differences


def run() -> dict:
    units = []
    for family, config in FAMILIES.items():
        for seed in config["seeds"]:
            for target in config["targets"]:
                task, eligible = _load_task(family, seed, target)
                for method in METHODS:
                    record = _replay(method, task, eligible)
                    units.append({
                        **record,
                        "family": family,
                        "seed": seed,
                        "target": target,
                        "method": method,
                    })

    summary = {
        family: {
            method: {
                metric: float(np.mean([
                    unit[metric] for unit in units
                    if unit["family"] == family and unit["method"] == method]))
                for metric in ("NewHistoricalFailureCount", "CollisionCount",
                               "CriticalCount", "EarlyNovelAUC", "CVS",
                               "pool_failures")
            }
            for method in METHODS
        }
        for family in FAMILIES
    }
    overall = {
        method: {
            metric: float(np.mean([
                unit[metric] for unit in units if unit["method"] == method]))
            for metric in ("NewHistoricalFailureCount", "CollisionCount",
                           "CriticalCount", "EarlyNovelAUC", "CVS", "pool_failures")
        }
        for method in METHODS
    }
    paired = {
        method: _paired_bootstrap(units, "ModeLabelShift-Risk", method)
        for method in METHODS if method != "ModeLabelShift-Risk"
    }
    output = {
        "schema": "mode_label_revision_offline_replay_v1",
        "development_only": True,
        "retrospective_full_outcome_banks": True,
        "new_physical_episodes": 0,
        "budget_per_seed_target_method": 50,
        "families": {
            family: {"seeds": config["seeds"], "targets": config["targets"]}
            for family, config in FAMILIES.items()
        },
        "methods": METHODS,
        "summary_by_family": summary,
        "summary_overall": overall,
        "paired_label_minus_comparators": paired,
        "unit_rows": units,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "analysis50.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    result = run()
    print(json.dumps({
        "summary_by_family": result["summary_by_family"],
        "summary_overall": result["summary_overall"],
        "paired_label_minus_comparators": result["paired_label_minus_comparators"],
    }, indent=2))


if __name__ == "__main__":
    main()
