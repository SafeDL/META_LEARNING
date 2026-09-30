"""Target-isolated FM2 selector and budgeted global/local router."""

from __future__ import annotations

import numpy as np
import torch

from methods.failure_memory_regression.fm2_dfe import DiverseFailureExpansion
from methods.failure_memory_regression.fm2_model import FM2Model, support_row
from methods.failure_memory_regression.fm2_schema import valid_label


class DiscountedRouter:
    def __init__(self, seed: int, gamma: float = 0.95):
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.success = np.zeros(2, dtype=float)
        self.failure = np.zeros(2, dtype=float)

    def select(self, local_available: bool) -> tuple[int, list[float], dict]:
        before = {"alpha": (1 + self.success).tolist(),
                  "beta": (1 + self.failure).tolist()}
        theta = [float(self.rng.beta(1 + self.success[i], 1 + self.failure[i]))
                 for i in range(2)] if local_available else [1.0, 0.0]
        return (int(theta[1] > theta[0]) if local_available else 0), theta, before

    def update(self, arm: int, reward: int) -> dict:
        self.success *= self.gamma
        self.failure *= self.gamma
        self.success[arm] += reward
        self.failure[arm] += 1 - reward
        return {"alpha": (1 + self.success).tolist(),
                "beta": (1 + self.failure).tolist()}


def run_fm2(model: FM2Model, scenarios: list[dict], coords: np.ndarray,
            memory: dict[str, np.ndarray], cards: list, oracle, *, budget: int = 50,
            seed: int = 0, dfe_enabled: bool = False, dfe_config: dict | None = None,
            router_discount: float = 0.95):
    if len(scenarios) != len(coords) or len({row["scenario_id"] for row in scenarios}) != len(scenarios):
        raise ValueError("candidate pool mismatch or duplicate IDs")
    ids = [row["scenario_id"] for row in scenarios]
    model.eval()
    with torch.no_grad():
        history_probability = torch.sigmoid(model(coords, memory, [])).cpu().numpy()
    support, queried, queries, retrieval, router_audit = [], set(), [], [], []
    dfe = DiverseFailureExpansion(coords, **(dfe_config or {})) if dfe_enabled else None
    router = DiscountedRouter(seed, gamma=router_discount)
    for rank in range(1, min(budget, len(ids)) + 1):
        with torch.no_grad():
            logit, audit = model(coords, memory, support, audit=True)
            probability = torch.sigmoid(logit).cpu().numpy()
        available = [i for i in range(len(ids)) if i not in queried]
        global_index = max(available, key=lambda i: (float(probability[i]), -i))
        local = dfe.propose(probability, queried) if dfe else None
        local_index, origin = local if local is not None else (None, None)
        arm, theta, before = router.select(local_index is not None)
        selected = local_index if arm == 1 else global_index
        assert selected is not None and selected not in queried
        # Snapshot retrieval before the only permitted target query.
        weights = audit["attention"][selected]
        gate = audit["gate"][selected]
        retrieval.append({"rank": rank, "scenario_id": ids[selected],
                          "memory_ids": [card.pattern_id for card in cards] + ["NULL_RETRIEVAL"],
                          "weights": weights.tolist(), "gates": gate.tolist(),
                          "fixed_probe_scenario_id": ids[0],
                          "fixed_probe_weights": audit["attention"][0].tolist(),
                          "support_size_before": len(support)})
        outcome = oracle.query(ids[selected])
        queried.add(selected)
        label = valid_label(outcome)
        reward = int(label == 1)
        after = router.update(arm, reward) if dfe else None
        item = support_row(coords[selected], outcome, float(history_probability[selected]))
        if item is not None:
            support.append(item)
        if dfe:
            dfe.observe(selected, label, origin if arm == 1 else None)
            router_audit.append({"rank": rank, "selected_arm": "local" if arm else "global",
                                 "theta_global": theta[0], "theta_local": theta[1],
                                 "candidate_global": ids[global_index],
                                 "candidate_local": ids[local_index] if local_index is not None else None,
                                 "actual_candidate": ids[selected], "outcome": label,
                                 "posterior_before": before, "posterior_after": after})
        queries.append({"method": "FM2-FBT" if dfe else "FM2-NoDFE",
                        "rank": rank, "scenario_id": ids[selected],
                        "failure_probability_before_query": float(probability[selected]),
                        "p_history_0": float(history_probability[selected]),
                        "history_target_residual": item["history_target_residual"] if item else None,
                        "target_observations_before_query": len(support) - int(item is not None),
                        "selected_arm": "local" if arm == 1 else "global",
                        "ego_collision": outcome.get("ego_collision"),
                        "completed": outcome.get("completed"),
                        "inconclusive": outcome.get("inconclusive"),
                        "valid_collision": bool(reward), "selection_reward": reward,
                        "episode_cost": 1, "execution_id": outcome.get("execution_id")})
    return queries, retrieval, router_audit
