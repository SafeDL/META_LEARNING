"""Two-stage graph-filtered fuzzing projected onto an unqueried public library."""

import time
from collections import Counter, deque
from dataclasses import asdict, dataclass

import numpy as np
import torch

from experiments.scenarios import DOMAINS, features, validate
from experiments.schema import Scenario
from testing_methods.selection import validate_initial_indices

from .graphs import graph
from .model import batch_graphs


@dataclass(frozen=True)
class Config:
    cycles: int = 3
    mutants: int = 100
    executions: int = 3
    neighbor_steps: int = 5
    filter_attempts: int = 6


DEFAULT = Config()


def mutate(parent, generator, *, local, steps=5):
    domain = DOMAINS[parent.kind]
    while True:
        parameters = {}
        for name, (low, high) in domain.bounds.items():
            precision = domain.quantization[name]
            if local:
                lower = max(-steps, int(np.ceil((low - parent.parameters[name]) / precision)))
                upper = min(steps, int(np.floor((high - parent.parameters[name]) / precision)))
                parameters[name] = (
                    parent.parameters[name] + int(generator.integers(lower, upper + 1)) * precision
                )
            else:
                parameters[name] = (
                    low + int(generator.integers(int((high - low) / precision) + 1)) * precision
                )
        categories = {}
        for name, options in domain.categories.items():
            current = options.index(parent.categories[name])
            choices = options[max(0, current - steps) : current + steps + 1] if local else options
            categories[name] = str(generator.choice(choices))
        proposal = Scenario("public_mutation", parent.kind, parameters, categories)
        try:
            validate(proposal)
        except ValueError:
            continue
        return proposal


def probabilities(classifier, candidates):
    classifier.eval().requires_grad_(False)
    device = next(classifier.parameters()).device
    with torch.no_grad():
        output = [
            classifier(batch_graphs([graph(s) for s in candidates[start : start + 100]]).to(device))
            .sigmoid()
            .cpu()
            .numpy()
            for start in range(0, len(candidates), 100)
        ]
    return np.concatenate(output)


def run(candidates, oracle, *, classifier, initial_indices, budget=128, seed=11, config=DEFAULT):
    initial = list(initial_indices)
    validate_initial_indices(len(candidates), initial, budget)
    if budget >= 1000:
        raise ValueError("This fixed-model adaptation is limited to fewer than 1000 new records")
    torch.set_num_threads(1)
    generator = np.random.default_rng(seed)
    started = time.perf_counter()
    risk = probabilities(classifier, candidates)
    inference_s = time.perf_counter() - started
    x = features(candidates)
    remaining, selected, queries = set(range(len(candidates))), [], []
    queue, frequency, campaign_log = deque(), Counter(), []

    def execute(
        index, *, campaign=0, cycle=0, reason="common_initialization", selection_s=0.0, **details
    ):
        feedback = oracle.query(
            index, purpose="initial" if reason == "common_initialization" else "target"
        )
        if feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError("ScenarioFuzz feedback differs from the paid candidate")
        remaining.remove(index)
        selected.append(index)
        queries.append(
            {
                "query": len(selected),
                "index": index,
                "feedback": feedback.to_dict(),
                "prequery_probability": float(risk[index]),
                "campaign": campaign,
                "cycle": cycle,
                "reason": reason,
                "selection_s": selection_s,
                "adaptation_s": 0.0,
                **details,
            }
        )
        return feedback

    for index in initial:
        execute(index)
    campaign = 0
    while len(selected) < budget:
        if not queue:
            queue.append(int(generator.integers(len(candidates))))
        parent = queue.popleft()
        campaign += 1
        frequency[parent] += 1
        successor = None
        for cycle in range(1, config.cycles + 1):
            started = time.perf_counter()
            local = successor is not None
            base = candidates[successor if local else parent]
            available = sorted(remaining, key=lambda i: candidates[i].scenario_id)
            count = min(config.executions, budget - len(selected), len(available))
            for attempt in range(1, config.filter_attempts + 1):
                proposals = [
                    mutate(base, generator, local=local, steps=config.neighbor_steps)
                    for _ in range(config.mutants)
                ]
                points = features(proposals)
                distance = ((points[:, None] - x[available][None]) ** 2).sum(-1)
                projected = sorted({available[i] for i in distance.argmin(-1)})
                ranked = sorted(projected, key=lambda i: (-risk[i], candidates[i].scenario_id))
                eligible = [i for i in ranked if risk[i] > 0.5]
                if len(eligible) >= count:
                    break
            batch = eligible[:count]
            # Fixed paid budgets require progress even if the trained filter rejects a region.
            fill = [i for i in ranked if i not in batch]
            fill += sorted(
                (i for i in available if i not in projected),
                key=lambda i: (-risk[i], candidates[i].scenario_id),
            )
            batch += fill[: count - len(batch)]
            selection_s = time.perf_counter() - started
            scores, error = [], False
            for number, index in enumerate(batch):
                feedback = execute(
                    index,
                    campaign=campaign,
                    cycle=cycle,
                    reason="random_neighbor" if local else "random_mutation",
                    selection_s=selection_s if number == 0 else 0.0,
                    filter_attempts=attempt,
                    projected_mutants=len(projected),
                    above_threshold=len(eligible),
                    threshold_fill=bool(risk[index] <= 0.5),
                    mutation_parent_index=successor if local else parent,
                )
                if feedback.collision:
                    reinsert_probability = 1 - frequency[parent] / campaign
                    reinserted = bool(generator.random() < reinsert_probability)
                    if reinserted:
                        queue.append(parent)
                    campaign_log.append(
                        {
                            "campaign": campaign,
                            "seed_index": parent,
                            "seed_selections": frequency[parent],
                            "reinsert_probability": reinsert_probability,
                            "reinserted": reinserted,
                            "collision_query": len(selected),
                        }
                    )
                    error = True
                    break
                score = feedback.safety.driving_quality_score
                if score is not None:
                    scores.append((score, index))
            if scores:
                successor = min(
                    scores, key=lambda item: (item[0], candidates[item[1]].scenario_id)
                )[1]
            if error or len(selected) == budget:
                break
    return {
        "method": "scenario_fuzz",
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(queries),
        "maximum_queries": budget,
        "termination": "budget_exhausted",
        "config": asdict(config),
        "model_inference_s": inference_s,
        "campaigns": campaign_log,
        "threshold_fill_queries": sum(q.get("threshold_fill", False) for q in queries),
        "feedback_used": "paid collision and safe-execution driving-quality score",
        "online_classifier_updates": 0,
        "adaptation": "public waypoint graph, finite-library projection and fixed-budget threshold completion",
    }
