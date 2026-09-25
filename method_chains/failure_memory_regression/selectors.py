"""Same-candidate, same-budget regression test selectors."""

from __future__ import annotations

import math

import numpy as np

from highway_env_benchmark.envs.fbrt_scenarios import FBRTScenario
from method_chains.failure_memory_regression.boundary_memory import (
    Patch, coordinates, nearest_patch,
)
from method_chains.failure_memory_regression.boundary_shift import BoundaryShift


class TargetOracle:
    def __init__(self, bank: dict[str, dict]) -> None:
        self._bank = bank
        self.queried: set[str] = set()

    def query(self, scenario_id: str) -> dict:
        if scenario_id in self.queried:
            raise ValueError(f"Repeated query: {scenario_id}")
        self.queried.add(scenario_id)
        return self._bank[scenario_id]


def margin_key(row: dict) -> tuple[float, float]:
    ttc = row["min_ttc"]
    return (float(ttc) if math.isfinite(ttc) else 1e6,
            float(row["min_clearance"]))


def art_choice(available: list[FBRTScenario], selected: list[FBRTScenario]) -> FBRTScenario:
    if not selected:
        return min(available, key=lambda scene: scene.scenario_id)
    def distance(scene: FBRTScenario) -> float:
        neighbors = [np.linalg.norm(coordinates(scene) - coordinates(other))
                     for other in selected if other.template_id == scene.template_id]
        return float(min(neighbors)) if neighbors else 2.0
    return max(available, key=lambda scene: (distance(scene), scene.scenario_id))


def run_selector(method: str, candidates: list[FBRTScenario],
                 historical_scenarios: list[FBRTScenario], source: dict[str, dict],
                 patches: list[Patch], oracle: TargetOracle, budget: int,
                 random_seed: int = 0) -> list[dict]:
    available = {scene.scenario_id: scene for scene in candidates}
    selected: list[FBRTScenario] = []
    results: list[dict] = []
    shift = BoundaryShift(patches)
    random_order = list(available)
    np.random.default_rng(random_seed).shuffle(random_order)
    ucb_counts = {template: 0 for template in {scene.template_id for scene in candidates}}
    ucb_rewards = {template: 0 for template in ucb_counts}
    failed_points = {}
    for scene in historical_scenarios:
        if source[scene.scenario_id]["ego_collision"]:
            failed_points.setdefault(scene.template_id, []).append(coordinates(scene))
    for rank in range(1, min(budget, len(available)) + 1):
        choices = list(available.values())
        patch = None
        u = 0.0
        if method in ("FBRT-Static", "FBRT-Adaptive") and rank in (10, 20):
            scene = art_choice(choices, selected)
            reason = "global_art"
        elif method == "Random":
            scene = available[next(sid for sid in random_order if sid in available)]
            reason = "random"
        elif method == "ART-Maximin":
            scene = art_choice(choices, selected)
            reason = "global_art"
        elif method == "HistoryMargin":
            scene = min(choices, key=lambda item: (margin_key(source[item.scenario_id]),
                                                    item.scenario_id))
            reason = "fallback_history"
        elif method == "FailureDistance":
            def failed_distance(item: FBRTScenario) -> float:
                points = failed_points.get(item.template_id, [])
                return min((float(np.linalg.norm(coordinates(item) - point))
                            for point in points), default=2.0)
            scene = min(choices, key=lambda item: (failed_distance(item),
                                                    item.scenario_id))
            reason = "failure_distance"
        elif method in ("HistoryRank-UCB", "FBRT-RegionBandit"):
            def ucb(item: FBRTScenario) -> float:
                template = item.template_id
                count = ucb_counts[template]
                return (float("inf") if count == 0 else
                        ucb_rewards[template] / count + math.sqrt(2 * math.log(rank) / count))
            template = max(ucb_counts, key=lambda name: (max(
                ucb(scene) for scene in choices if scene.template_id == name)
                if any(scene.template_id == name for scene in choices) else -1, name))
            local = [item for item in choices if item.template_id == template]
            if method == "HistoryRank-UCB":
                scene = min(local, key=lambda item: (margin_key(source[item.scenario_id]),
                                                     item.scenario_id))
                reason = "history_ucb"
            else:
                margin_order = sorted(local, key=lambda item: (
                    margin_key(source[item.scenario_id]), item.scenario_id))
                margin_rank = {item.scenario_id: index for index, item in enumerate(margin_order)}
                boundary = {item.scenario_id: nearest_patch(item, patches) for item in local}
                if any(boundary[item.scenario_id][0] is not None for item in local):
                    boundary_order = sorted(local, key=lambda item: (
                        -boundary[item.scenario_id][1], item.scenario_id))
                    boundary_rank = {item.scenario_id: index for index, item in enumerate(boundary_order)}
                    scene = min(local, key=lambda item: (
                        margin_rank[item.scenario_id] + boundary_rank[item.scenario_id],
                        margin_rank[item.scenario_id], item.scenario_id))
                    patch, _, u, _ = boundary[scene.scenario_id]
                    reason = "region_bandit"
                else:
                    scene = margin_order[0]
                    reason = "region_bandit_no_patch"
        elif method in ("FBRT-Static", "FBRT-Adaptive"):
            scored = []
            for item in choices:
                candidate_patch, static_score, candidate_u, v = nearest_patch(item, patches)
                if candidate_patch is None:
                    score = -1.0
                elif method == "FBRT-Adaptive":
                    score = shift.score(candidate_patch, candidate_u, v)
                else:
                    score = static_score
                scored.append((score, item.scenario_id, item, candidate_patch, candidate_u))
            score, _, scene, patch, u = max(scored, key=lambda item: (item[0], item[1]))
            if score < 0:
                scene = min(choices, key=lambda item: (margin_key(source[item.scenario_id]),
                                                       item.scenario_id))
                reason = "fallback_history"
            else:
                reason = "adaptive_boundary" if method == "FBRT-Adaptive" else "boundary"
        else:
            raise ValueError(f"Unknown selector: {method}")
        outcome = oracle.query(scene.scenario_id)
        regression = bool(source[scene.scenario_id]["completed"] and
                          not source[scene.scenario_id]["ego_collision"] and
                          outcome["ego_collision"])
        if (method == "FBRT-Adaptive" and patch is not None and
                reason == "adaptive_boundary" and outcome["semantic_valid"]):
            shift.update(patch, u, outcome["ego_collision"])
        if method in ("HistoryRank-UCB", "FBRT-RegionBandit"):
            ucb_counts[scene.template_id] += 1
            ucb_rewards[scene.template_id] += int(regression)
        results.append({"method": method, "rank": rank, "scenario_id": scene.scenario_id,
                        "template_id": scene.template_id, "selection_reason": reason,
                        "patch_id": patch.patch_id if patch else "",
                        "ego_collision": bool(outcome["ego_collision"]),
                        "regression": regression})
        selected.append(scene)
        del available[scene.scenario_id]
    return results
