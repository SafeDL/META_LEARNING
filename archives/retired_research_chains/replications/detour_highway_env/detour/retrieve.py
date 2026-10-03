"""Faithful DETOUR tree retrieval with an explicit no-failure fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from replications.detour_highway_env.detour.tree import DetourTree, TreeNode


@dataclass(frozen=True)
class RetrievalTrace:
    selected_index: int
    selected_reason: str
    leaf_node_id: int
    path_node_ids: tuple[int, ...]
    branch_probabilities: tuple[dict[str, float], ...]
    node_counts: tuple[dict[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class Retriever:
    def __init__(self, tree: DetourTree, seed: int) -> None:
        self.tree, self.rng = tree, np.random.default_rng(seed)

    def retrieve(self) -> RetrievalTrace | None:
        if not self.tree.active_candidates: return None
        if not self.tree.failure_indices:
            selected = int(self.rng.choice(sorted(self.tree.active_candidates)))
            self.tree.remove_candidate(selected)
            return RetrievalTrace(selected, "fallback_no_failure",
                                  self.tree.history_count + selected, (self.tree.root.node_id, ),
                                  (), (self._counts(self.tree.root), ))
        return self._retrieve(self.tree.root, [self.tree.root.node_id], [],
                              [self._counts(self.tree.root)])

    def _counts(self, node: TreeNode) -> dict[str, int]:
        return {
            "node_id": node.node_id,
            "executed_count": node.executed_count,
            "fail_count": self.tree.node_fail_count(node),
            "selectable_count": len(self.tree.active_in(node))
        }

    def _retrieve(self, node: TreeNode, path: list[int], probabilities: list[dict[str, float]],
                  counts: list[dict[str, int]]) -> RetrievalTrace:
        if node.left is not None:
            eligible = [
                child for child in (node.left, node.right)
                if self.tree.active_in(child) and self.tree.node_fail_count(child) > 0
            ]
            if eligible:
                weights = np.asarray([
                    self.tree.node_fail_count(child) / child.executed_count for child in eligible
                ])
                weights /= weights.sum()
                child = eligible[int(self.rng.choice(len(eligible), p=weights))]
                probabilities.append({
                    str(item.node_id): float(weight)
                    for item, weight in zip(eligible, weights, strict=True)
                })
                return self._retrieve(child, path + [child.node_id], probabilities,
                                      counts + [self._counts(child)])
        return self._choose_nearest(node, path, probabilities, counts)

    def _choose_nearest(self, node: TreeNode, path: list[int], probabilities: list[dict[str,
                                                                                        float]],
                        counts: list[dict[str, int]]) -> RetrievalTrace:
        candidates, failures = self.tree.active_in(node), [
            index for index in node.history_indices if index in self.tree.failure_indices
        ]
        reason = "nearest_failure"
        if not candidates or not failures:
            candidates, failures, reason = sorted(self.tree.active_candidates), sorted(
                self.tree.failure_indices), "nearest_failure_ancestor_fallback"
        _distance, selected, _failure = min(
            (self.tree.candidate_history_distance(candidate, failure), candidate, failure)
            for candidate in candidates for failure in failures)
        self.tree.remove_candidate(selected)
        return RetrievalTrace(selected, reason, self.tree.history_count + selected, tuple(path),
                              tuple(probabilities), tuple(counts))
