"""Budgeted selectors with isolated oracles and auditable memory contributions."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from method_chains.failure_memory_regression.bayes_model import (
    build_source_prior, extend_prior, posterior_failure_probabilities, source_fits,
    target_posterior,
)
from method_chains.failure_memory_regression.pattern_memory import (
    active_values, build_dictionaries, build_pattern_cards, new_failure_card,
)
from method_chains.failure_memory_regression.schema_v2 import stable_hash


METHODS = ("Random", "HistoryRank-UCB-v2", "FailureDistance-v2",
           "FBRT-NoMemory", "FBRT-Memory")
CHECKPOINTS = (1, 5, 10, 20)


class TargetOracle:
    """The only interface allowed to reveal evaluator rows."""

    def __init__(self, bank: dict[str, dict]):
        self.__bank = bank
        self.queried: set[str] = set()

    def query(self, scenario_id: str) -> dict:
        if scenario_id in self.queried:
            raise ValueError(f"repeated target query: {scenario_id}")
        if scenario_id not in self.__bank:
            raise KeyError(f"candidate is absent from evaluator bank: {scenario_id}")
        self.queried.add(scenario_id)
        return dict(self.__bank[scenario_id])


def _scenario(item: dict) -> dict:
    value = item.get("scenario", item)
    return value if isinstance(value, dict) else {}


def _failure(row: dict) -> bool:
    return row.get("ego_collision") is True and not row.get("inconclusive", False)


def _parent_pass(item: dict) -> bool:
    if "parent_pass" in item:
        return bool(item["parent_pass"])
    return item.get("completed") is True and item.get("ego_collision") is False


def _art_choice(available: list[dict], selected: list[dict], rng: np.random.Generator) -> dict:
    if not selected:
        return available[int(rng.integers(len(available)))]
    scores = []
    for item in available:
        z = np.asarray(active_values(_scenario(item)))
        same = [np.asarray(active_values(_scenario(other))) for other in selected
                if other.get("template_id") == item.get("template_id")]
        scores.append(min((float(np.linalg.norm(z - point)) for point in same), default=2.0))
    best = max(scores)
    ties = [i for i, value in enumerate(scores) if np.isclose(value, best)]
    return available[ties[int(rng.integers(len(ties)))]]


def _source_risk(item: dict, history: list[dict]) -> float:
    exact_id = item.get("scenario_id")
    same = [row for row in history if row.get("template_id") == item.get("template_id")
            and not row.get("inconclusive")]
    exact = [row for row in same if row.get("scenario_id") == exact_id]
    pool = exact or sorted(same, key=lambda row: float(np.linalg.norm(
        np.asarray(active_values(_scenario(item))) -
        np.asarray(active_values(_scenario(row))))))[:5]
    if not pool:
        return 0.0
    risks = []
    for row in pool:
        collision = 1.0 if row.get("ego_collision") is True else 0.0
        ttc = row.get("min_ttc")
        clearance = row.get("min_clearance")
        risks.append(collision + 0.49 * (1 / (1 + max(0.0, float(ttc or 1e6)))) +
                     0.01 * (1 / (1 + max(0.0, float(clearance or 1e6)))))
    return float(np.mean(risks))


def _history_rank(item: dict, history: list[dict]) -> float:
    # Within-source percentile first, then equal-weight across source families.
    by_build: dict[str, list[dict]] = defaultdict(list)
    for row in history:
        by_build[row.get("build_id", "unknown")].append(row)
    percentiles = []
    for rows in by_build.values():
        same_template = [row for row in rows if row.get("template_id") == item.get("template_id")]
        if not same_template:
            continue
        candidate_risk = _source_risk(item, same_template)
        local_risks = [_source_risk({**item, "scenario_id": row.get("scenario_id"),
                                     "scenario": row.get("scenario")}, same_template)
                       for row in same_template]
        percentiles.append(sum(r <= candidate_risk for r in local_risks) / len(local_risks))
    return float(np.mean(percentiles)) if percentiles else 0.0


def _choose_tie(indices: list[int], rng: np.random.Generator) -> int:
    return indices[int(rng.integers(len(indices)))]


def run_selector_v2(method: str, candidates: list[dict], history: list[dict],
                    oracle: TargetOracle, budget: int, random_seed: int,
                    target_build_id: str, parent_build_id: str | None = None,
                    mode: str = "cross_agent", session_id: str = "session",
                    family_by_build: dict[str, str] | None = None,
                    initial_cards=None) -> tuple[list[dict], list[dict], list[dict]]:
    if method not in METHODS:
        raise ValueError(f"unknown v2 method {method}")
    rng = np.random.default_rng(random_seed)
    candidate_map = {row["scenario_id"]: row for row in candidates}
    available_ids = set(candidate_map)
    candidate_order = list(candidate_map)
    selected: list[dict] = []
    queries: list[dict] = []
    observations: list[dict] = []
    updates: list[dict] = []
    if method == "FBRT-NoMemory":
        cards = []
    else:
        cards = list(initial_cards) if initial_cards is not None else build_pattern_cards(history)
    dictionaries = build_dictionaries(history if method != "FBRT-NoMemory" else [], cards,
                                       candidates, seed=7319)
    fits_by_template = {template: source_fits(history if method != "FBRT-NoMemory" else [], dictionary)
                        for template, dictionary in dictionaries.items()}
    families = family_by_build or {row["build_id"]: row.get("family", row["build_id"])
                                   for row in history}
    priors = {}
    source_ids = {}
    for template, dictionary in dictionaries.items():
        fits = fits_by_template[template]
        if method == "FBRT-NoMemory":
            dim = 1 + dictionary.feature_dim + len(dictionary.centers) + len(dictionary.coverage_centers)
            priors[template] = (np.zeros(dim), np.full(dim, 4.0))
            source_ids[template] = []
        else:
            mean, variance, used = build_source_prior(
                fits, families, parent_build_id=parent_build_id,
                regression=(mode == "regression"))
            mean, variance = extend_prior(mean, variance,
                1 + dictionary.feature_dim + len(dictionary.centers) + len(dictionary.coverage_centers))
            priors[template] = (mean, variance)
            source_ids[template] = used

    ucb_count: dict[str, int] = defaultdict(int)
    ucb_reward: dict[str, float] = defaultdict(float)
    templates = sorted({item["template_id"] for item in candidates})
    untried_order = list(templates)
    rng.shuffle(untried_order)
    parent_by_id = {item["scenario_id"]: item for item in candidates}

    horizon = min(int(budget), len(candidates))
    for rank in range(1, horizon + 1):
        pool = [candidate_map[key] for key in candidate_order if key in available_ids]
        reason = ""
        contributing: list[str] = []
        if method == "Random":
            index = int(rng.integers(len(pool)))
            scene = pool[index]
            reason = "seeded_random_without_replacement"
        elif method == "HistoryRank-UCB-v2":
            untried = [template for template in untried_order
                       if ucb_count[template] == 0 and any(
                           item["template_id"] == template for item in pool)]
            if untried:
                chosen_template = untried[0]
            else:
                options = [template for template in templates if any(
                    item["template_id"] == template for item in pool)]
                values = [ucb_reward[t] / ucb_count[t] +
                          math.sqrt(2 * math.log(max(rank, 2)) / ucb_count[t])
                          for t in options]
                best = max(values)
                chosen_template = options[_choose_tie(
                    [i for i, value in enumerate(values) if np.isclose(value, best)], rng)]
            local = [item for item in pool if item["template_id"] == chosen_template]
            ranks_by_history = [_history_rank(item, history) for item in local]
            maximum = max(ranks_by_history)
            scene = local[_choose_tie([i for i, value in enumerate(ranks_by_history)
                                       if np.isclose(value, maximum)], rng)]
            reason = "history_risk_percentile_ucb"
        elif method == "FailureDistance-v2":
            failed_by_template: dict[str, list[np.ndarray]] = defaultdict(list)
            for row in history:
                if _failure(row):
                    failed_by_template[row["template_id"]].append(
                        np.asarray(active_values(_scenario(row))))
            for row in observations:
                if _failure(row):
                    failed_by_template[row["template_id"]].append(
                        np.asarray(active_values(_scenario(row))))
            if rank in (10, 20):
                scene = _art_choice(pool, selected, rng)
                reason = "global_art_maximin"
            else:
                distances = [min((float(np.linalg.norm(np.asarray(active_values(_scenario(item))) -
                                                         point))
                                  for point in failed_by_template[item["template_id"]]), default=2.0)
                             for item in pool]
                minimum = min(distances)
                scene = pool[_choose_tie([i for i, value in enumerate(distances)
                                           if np.isclose(value, minimum)], rng)]
                reason = "nearest_observed_failure"
        else:
            template = None
            probabilities = np.zeros(len(pool), dtype=float)
            if rank in (10, 20):
                scene = _art_choice(pool, selected, rng)
                reason = "global_art_maximin"
            else:
                for template_name in {item["template_id"] for item in pool}:
                    dictionary = dictionaries[template_name]
                    prior_mean, prior_variance = priors[template_name]
                    fit = target_posterior(dictionary, observations, prior_mean, prior_variance)
                    subset_indices = [i for i, item in enumerate(pool)
                                      if item["template_id"] == template_name]
                    subset = [pool[i] for i in subset_indices]
                    scores_local = posterior_failure_probabilities(
                        dictionary, subset, fit, seed=random_seed + rank +
                        sum(template_name.encode("utf-8")), samples=32)
                    probabilities[subset_indices] = scores_local
                best = float(np.max(probabilities))
                ties = [i for i, value in enumerate(probabilities) if np.isclose(value, best)]
                selected_index = _choose_tie(ties, rng)
                scene = pool[selected_index]
                reason = "posterior_mean_failure_probability"
                template = scene["template_id"]
                dictionary = dictionaries[template]
                z = np.asarray(active_values(_scenario(scene)))
                contributing = [item["center_id"] for item in dictionary.centers
                                if np.exp(-np.sum((z - np.asarray(item["center"])) ** 2) /
                                          (2 * float(item["bandwidth"]) ** 2)) >= 0.1]
        outcome = oracle.query(scene["scenario_id"])
        parent = parent_by_id.get(scene["scenario_id"], {})
        collision = outcome.get("ego_collision") is True and not outcome.get("inconclusive", False)
        regression = bool(_parent_pass(parent) and collision)
        execution_id = outcome.get("execution_id") or "target-" + stable_hash({
            "build_id": target_build_id, "scenario_id": scene["scenario_id"],
            "seed": outcome.get("simulator_seed", outcome.get("seed")),
            "contract": outcome.get("execution_contract_version", "legacy_replay_v2"),
        })[:24]
        observed = {
            "execution_id": execution_id, "scenario_id": scene["scenario_id"],
            "build_id": target_build_id,
            "template_id": scene["template_id"], "scenario": _scenario(scene),
            "context_id": scene.get("context_id") or _scenario(scene).get("context_id", "legacy_unspecified"),
            "completed": outcome.get("completed"), "ego_collision": outcome.get("ego_collision"),
            "inconclusive": bool(outcome.get("inconclusive", False)),
            "min_ttc": outcome.get("min_ttc"), "min_clearance": outcome.get("min_clearance"),
            "visibility": "historical", "episode_cost": int(outcome.get("episode_cost", 0)),
        }
        if not observed["inconclusive"] and observed["ego_collision"] in (True, False):
            observations.append(observed)
        created_card_id = None
        feature_added = False
        if collision and method in ("FBRT-Memory", "FBRT-NoMemory"):
            card, feature_added = new_failure_card(observed, dictionaries, session_id)
            created_card_id = card.pattern_id
            # A prior local pass is recorded as a contrast only when actually observed.
            same = [row for row in observations[:-1]
                    if row["template_id"] == scene["template_id"]
                    and row.get("ego_collision") is False]
            if same:
                nearest = min(same, key=lambda row: float(np.linalg.norm(
                    np.asarray(active_values(_scenario(row))) - np.asarray(card.center))))
                card.pass_contrast_record_ids.append(nearest["execution_id"])
                card.boundary_edges.append((execution_id, nearest["execution_id"]))
                card.evidence_status = "contrast_available"
            cards.append(card)
            updates.append({"rank": rank, "event": "new_failure_pattern",
                            "pattern_id": card.pattern_id, "execution_id": execution_id,
                            "feature_added": feature_added,
                            "nearby_observed_pass_ids": card.pass_contrast_record_ids})
        if method == "HistoryRank-UCB-v2":
            ucb_count[scene["template_id"]] += 1
            ucb_reward[scene["template_id"]] += int(regression)
        if method in ("FBRT-Memory", "FBRT-NoMemory") and not outcome.get("inconclusive", False):
            updates.append({"rank": rank, "event": "posterior_refit",
                            "template_id": scene["template_id"],
                            "observations_in_template": sum(
                                row["template_id"] == scene["template_id"] for row in observations),
                            "feature_count": 1 + dictionaries[scene["template_id"]].feature_dim +
                            len(dictionaries[scene["template_id"]].centers) +
                            len(dictionaries[scene["template_id"]].coverage_centers),
                            "contributing_pattern_ids": contributing if method == "FBRT-Memory" and
                            rank not in (10, 20) else [],
                            "source_build_ids": source_ids[scene["template_id"]]})
        query = {
            "method": method, "rank": rank, "scenario_id": scene["scenario_id"],
            "template_id": scene["template_id"], "selection_reason": reason,
            "selected_build_id": target_build_id, "session_id": session_id,
            "mode": mode, "ego_collision": outcome.get("ego_collision"),
            "inconclusive": bool(outcome.get("inconclusive", False)),
            "regression": regression, "execution_id": execution_id,
            "episode_cost": observed["episode_cost"],
            "contributing_pattern_ids": (contributing if method == "FBRT-Memory" and
                                          rank not in (10, 20) else []),
            "new_pattern_id": created_card_id or "",
            "target_observations_before_query": len(observations) -
            int(not observed["inconclusive"] and observed["ego_collision"] in (True, False)),
            "history_source_count": len(source_ids.get(scene["template_id"], [])),
        }
        selected.append(scene)
        queries.append(query)
        available_ids.remove(scene["scenario_id"])
    return queries, observations, cards, updates
