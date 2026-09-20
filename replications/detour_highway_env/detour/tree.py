"""Ward hierarchy with immutable executed/failure counts and dynamic candidates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


@dataclass
class TreeNode:
    node_id: int
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    history_indices: tuple[int, ...] = ()
    candidate_indices: tuple[int, ...] = ()
    parent: "TreeNode | None" = field(default=None, repr=False)

    @property
    def executed_count(self) -> int:
        return len(self.history_indices)

    @property
    def selectable_count(self) -> int:
        return len(self.candidate_indices)


@dataclass
class DetourTree:
    root: TreeNode
    distances: np.ndarray
    history_count: int
    candidate_count: int
    failure_indices: frozenset[int]
    active_candidates: set[int]
    nodes: dict[int, TreeNode]

    def node_fail_count(self, node: TreeNode) -> int:
        return sum(index in self.failure_indices for index in node.history_indices)

    def active_in(self, node: TreeNode) -> list[int]:
        return [index for index in node.candidate_indices if index in self.active_candidates]

    def remove_candidate(self, candidate_index: int) -> None:
        self.active_candidates.remove(candidate_index)

    def candidate_history_distance(self, candidate_index: int, history_index: int) -> float:
        return float(self.distances[self.history_count + candidate_index, history_index])

    def node_rows(self) -> list[dict[str, object]]:
        return [{
            "node_id": node.node_id,
            "left_id": "" if node.left is None else node.left.node_id,
            "right_id": "" if node.right is None else node.right.node_id,
            "parent_id": "" if node.parent is None else node.parent.node_id,
            "executed_count": node.executed_count,
            "fail_count": self.node_fail_count(node),
            "selectable_count_initial": node.selectable_count,
            "candidate_indices": ";".join(map(str, node.candidate_indices)),
            "history_indices": ";".join(map(str, node.history_indices))
        } for node in (self.nodes[key] for key in sorted(self.nodes))]


def build_tree(history_features: np.ndarray, candidate_features: np.ndarray,
               failed_history: np.ndarray) -> DetourTree:
    """Cluster history and candidates jointly, preserving duplicate coordinates."""
    history, candidates, failed = np.asarray(history_features, dtype=float), np.asarray(
        candidate_features, dtype=float), np.asarray(failed_history, dtype=bool)
    if history.ndim != 2 or candidates.ndim != 2 or history.shape[1] != candidates.shape[1]:
        raise ValueError("history and candidates must be two-dimensional with equal width")
    if len(history) == 0 or len(candidates) == 0 or failed.shape != (len(history), ):
        raise ValueError("non-empty aligned history/candidates are required")
    features, total = np.vstack((history, candidates)), len(history) + len(candidates)
    pairwise = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    linkage_matrix = linkage(squareform(pairwise, checks=False), method="ward")
    nodes = {
        leaf: TreeNode(leaf, history_indices=(leaf, ))
        if leaf < len(history) else TreeNode(leaf, candidate_indices=(leaf - len(history), ))
        for leaf in range(total)
    }
    for offset, (left_id, right_id, _distance, _count) in enumerate(linkage_matrix):
        left, right, node_id = nodes[int(left_id)], nodes[int(right_id)], total + offset
        node = TreeNode(node_id, left, right, left.history_indices + right.history_indices,
                        left.candidate_indices + right.candidate_indices)
        left.parent = right.parent = node
        nodes[node_id] = node
    return DetourTree(nodes[total + len(linkage_matrix) - 1], pairwise, len(history),
                      len(candidates), frozenset(np.flatnonzero(failed).tolist()),
                      set(range(len(candidates))), nodes)
