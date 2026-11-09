"""Energy-driven BehAVExplor with paid trajectories and public pool projection."""

import time
from dataclasses import asdict, dataclass

import numpy as np

from experiments.costs import measure_resources
from experiments.scenarios import DOMAINS, features, validate
from experiments.schema import Scenario
from testing_methods.selection import validate_initial_indices

from .behavior import BehaviorMiner, motion_trace, temporal_features


@dataclass(frozen=True)
class Config:
    interval_s: float = 0.1
    window: int = 10
    clusters: int = 10
    novelty_threshold: float = 0.4
    energy_threshold: float = 0.5
    mutation_probability: float = 0.5
    offset_sigma_m: float = 12
    speed_sigma_mps: float = 6
    other_sigma_fraction: float = 0.1


DEFAULT = Config()


@dataclass
class Seed:
    index: int
    fitness: float
    energy: float = 1.5
    local_trials: int = 0
    successful_trials: int = 0

    def update(self, *, collision, novel, improvement, distance, local):
        # Published code: useful local mutants include failures and new benign behaviors.
        if local:
            self.local_trials += 1
            useful = collision or novel
            self.successful_trials += int(useful)
            rate = self.successful_trials / self.local_trials
            frequency = rate if useful else -0.1 * (1 - rate)
            change = np.tanh(improvement / (1 - distance + 1e-10))
            self.energy += 0.5 * frequency + 0.5 * change - 0.05
        else:
            self.energy -= 0.1
        self.energy = max(0.0, float(self.energy))


def mutate(parent, generator, *, local, config=DEFAULT):
    domain = DOMAINS[parent.kind]
    while True:
        parameters = {}
        for name, (low, high) in domain.bounds.items():
            value = parent.parameters[name]
            if not local:
                value = generator.uniform(low, high)
            elif generator.random() < config.mutation_probability:
                sigma = (
                    config.speed_sigma_mps
                    if name.endswith("_mps")
                    else config.offset_sigma_m
                    if name.endswith("_m")
                    else config.other_sigma_fraction * (high - low)
                )
                value = np.clip(generator.normal(value, sigma), low, high)
            parameters[name] = float(value)
        categories = {
            name: parent.categories[name] if local else str(generator.choice(options))
            for name, options in domain.categories.items()
        }
        proposal = Scenario("behavexplor_proposal", parent.kind, parameters, categories)
        try:
            validate(proposal)
        except ValueError:
            continue
        return proposal


@measure_resources
def run(candidates, oracle, *, initial_indices, budget=200, seed=11, config=DEFAULT):
    initial = list(initial_indices)
    validate_initial_indices(len(candidates), initial, budget)
    x = features(candidates)
    generator = np.random.default_rng(seed)
    miner = BehaviorMiner(clusters=config.clusters, threshold=config.novelty_threshold, seed=seed)
    remaining = set(range(len(candidates)))
    corpus, initial_features, selected, queries = [], [], [], []
    for step in range(budget):
        started = time.perf_counter()
        parent, local = None, False
        if step < len(initial):
            index, reason = initial[step], "common_initialization"
        elif not corpus:
            index = int(generator.choice(sorted(remaining)))
            reason = "empty_corpus_uniform"
        else:
            weights = np.array([s.energy for s in corpus])
            probabilities = weights / weights.sum() if weights.sum() else None
            parent = corpus[int(generator.choice(len(corpus), p=probabilities))]
            local = parent.energy > config.energy_threshold
            proposal = mutate(candidates[parent.index], generator, local=local, config=config)
            available = sorted(remaining, key=lambda i: candidates[i].scenario_id)
            distance = ((x[available] - features([proposal])[0]) ** 2).sum(1)
            index = available[int(np.argmin(distance))]
            reason = "gaussian_mutation" if local else "uniform_mutation"
        selection_s = time.perf_counter() - started
        parent_energy = None if parent is None else parent.energy
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError("BehAVExplor feedback differs from its paid candidate")
        started = time.perf_counter()
        observed = temporal_features(
            motion_trace(feedback.trajectory, interval=config.interval_s), window=config.window
        )
        fitness = float(np.clip(feedback.safety.min_clearance_m, 0, 1))
        distance, novel, retained = None, False, False
        if parent is None:
            if not feedback.collision:
                corpus.append(Seed(index, fitness))
                retained = True
                if step < len(initial):
                    initial_features.append(observed)
                else:
                    miner.initialize([observed])
            if step == len(initial) - 1 and initial_features:
                miner.initialize(initial_features)
        else:
            novel, distance = miner.feedback(observed)
            improvement = parent.fitness - fitness
            if not feedback.collision and (novel or improvement > 0):
                corpus.append(Seed(index, fitness, max(0, 1 + (improvement + distance) / 2)))
                retained = True
                if novel:
                    miner.add(observed)
            parent.update(
                collision=feedback.collision,
                novel=novel and not feedback.collision,
                improvement=improvement,
                distance=distance,
                local=local,
            )
        remaining.remove(index)
        selected.append(index)
        queries.append(
            {
                "query": step + 1,
                "index": index,
                "feedback": feedback.to_dict(),
                "reason": reason,
                "mutation_parent_index": None if parent is None else parent.index,
                "parent_energy_before": parent_energy,
                "parent_energy_after": None if parent is None else parent.energy,
                "collision_fitness": fitness,
                "behavior_distance": distance,
                "novel_behavior": bool(novel),
                "retained_seed": retained,
                "corpus_size": len(corpus),
                "behavior_rebuilds": miner.rebuilds,
                "behavior_clusters": 0 if miner.model is None else miner.model.n_clusters,
                "selection_s": selection_s,
                "adaptation_s": time.perf_counter() - started,
            }
        )
    return {
        "method": "behavexplor",
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(selected),
        "maximum_queries": budget,
        "termination": "budget_exhausted",
        "configuration": asdict(config),
        "feedback_used": "paid collision, clearance and ego trajectory",
        "corpus_indices": [s.index for s in corpus],
        "corpus_energy": [s.energy for s in corpus],
        "author_code_revision": "7ee9cdde4673cb194eab2da70eecc408cf2a67b7",
        "energy_definition": "author code bounded update and initializer",
        "pool_projection": "nearest unqueried normalized coordinate; scene ID ties",
    }
