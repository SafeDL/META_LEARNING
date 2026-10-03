"""Static DETOUR prioritization and selection stopping semantics."""

from __future__ import annotations
from dataclasses import dataclass
from replications.detour_highway_env.detour.retrieve import RetrievalTrace, Retriever


@dataclass(frozen=True)
class StopDecision:
    step: int
    selected_index: int | None
    safe_neighbor: bool | None
    safe_neighbor_streak: int
    reason: str


def nearest_history_is_safe(tree, candidate_index: int, m_neighbors: int) -> bool | None:
    if tree.history_count < m_neighbors: return None
    nearest = sorted((tree.candidate_history_distance(candidate_index, index), index)
                     for index in range(tree.history_count))[:m_neighbors]
    return all(index not in tree.failure_indices for _distance, index in nearest)


def prioritize(retriever: Retriever,
               count: int) -> tuple[list[int], list[RetrievalTrace], list[StopDecision]]:
    order, traces, decisions = [], [], []
    for step in range(count):
        trace = retriever.retrieve()
        if trace is None:
            decisions.append(StopDecision(step, None, None, 0, "candidates_exhausted"))
            break
        order.append(trace.selected_index)
        traces.append(trace)
        decisions.append(StopDecision(step + 1, trace.selected_index, None, 0, "selected"))
    if len(order) == count:
        decisions.append(StopDecision(len(order), None, None, 0, "budget_reached"))
    return order, traces, decisions


def select(retriever: Retriever, min_count: int, max_count: int, m_neighbors: int,
           w_streak: int) -> tuple[list[int], list[RetrievalTrace], list[StopDecision]]:
    if not 0 <= min_count <= max_count or m_neighbors < 1 or w_streak < 1:
        raise ValueError("invalid selection bounds or stopping parameters")
    order, traces, decisions, streak = [], [], [], 0
    while len(order) < max_count:
        trace = retriever.retrieve()
        if trace is None:
            decisions.append(StopDecision(len(order), None, None, streak, "candidates_exhausted"))
            break
        safe = nearest_history_is_safe(retriever.tree, trace.selected_index, m_neighbors)
        streak = streak + 1 if safe is True else 0
        order.append(trace.selected_index)
        traces.append(trace)
        decisions.append(
            StopDecision(len(order), trace.selected_index, safe, streak,
                         "selected" if safe is not None else "insufficient_neighbors"))
        if len(order) >= min_count and streak >= w_streak:
            decisions.append(StopDecision(len(order), None, None, streak, "safe_neighbor_streak"))
            break
    else:
        decisions.append(StopDecision(len(order), None, None, streak, "max_count_reached"))
    return order, traces, decisions
