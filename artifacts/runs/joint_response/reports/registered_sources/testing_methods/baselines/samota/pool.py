"""SAMOTA's global committee and clustered local RBF search on a finite pool."""

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch import nn

from experiments.scenarios import DOMAINS, features
from testing_methods.selection import validate_initial_indices


@dataclass(frozen=True)
class Config:
    rbf_centers: int = 10
    rbf_epochs: int = 1000
    elite_fraction: float = 0.2
    min_cluster_size: int = 5
    thresholds: tuple[float, ...] = (0.0, 0.0, 0.2)


DEFAULT = Config()


def preference_order(values, active):
    selected = []
    for objective in active:
        best = int(np.argmin(values[:, objective]))
        if best not in selected:
            selected.append(best)
    remaining = set(range(len(values))) - set(selected)
    objectives = values[:, active]
    dominance = np.all(objectives[:, None] <= objectives[None, :], axis=2) & np.any(
        objectives[:, None] < objectives[None, :], axis=2
    )
    while remaining:
        indices = sorted(remaining)
        front = [i for i in indices if not dominance[indices, i].any()]
        crowding = {i: 0.0 for i in front}
        for objective in active:
            ordered = sorted(front, key=lambda i: values[i, objective])
            span = values[ordered[-1], objective] - values[ordered[0], objective]
            crowding[ordered[0]] = crowding[ordered[-1]] = np.inf
            if span > 1e-12:
                for position in range(1, len(ordered) - 1):
                    crowding[ordered[position]] += (
                        values[ordered[position + 1], objective]
                        - values[ordered[position - 1], objective]
                    ) / span
        selected.extend(sorted(front, key=lambda i: (-crowding[i], i)))
        remaining.difference_update(front)
    return selected


class RadialNetwork(nn.Module):
    def __init__(self, x, *, centers=10, seed=11):
        super().__init__()
        selected = np.random.default_rng(seed).integers(len(x), size=centers)
        self.centers = nn.Parameter(torch.as_tensor(x[selected], dtype=torch.float32).clone())
        self.betas = nn.Parameter(torch.full((centers,), 3.0))
        self.output = nn.Linear(centers, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        basis = torch.exp(-self.betas * ((x[:, None] - self.centers) ** 2).sum(-1))
        return self.output(basis).squeeze(-1)


class RadialSurrogate:
    def __init__(self, x, y, *, seed=11, config=DEFAULT):
        torch.manual_seed(seed)
        self.scaler = StandardScaler().fit(x)
        x = self.scaler.transform(x)
        self.model = RadialNetwork(x, centers=config.rbf_centers, seed=seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-7)
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        generator = np.random.default_rng(seed)
        for _ in range(config.rbf_epochs):
            indices = generator.permutation(len(x))
            for start in range(0, len(x), 8):
                batch = indices[start : start + 8]
                optimizer.zero_grad(set_to_none=True)
                loss = (self.model(x[batch]) - y[batch]).abs().mean()
                loss.backward()
                optimizer.step()

    def predict(self, x):
        with torch.no_grad():
            values = self.model(torch.tensor(self.scaler.transform(x), dtype=torch.float32)).numpy()
        return np.clip(values, 0, 1)


class Committee:
    def __init__(self, x, y, *, seed=11, config=DEFAULT):
        order = np.random.default_rng(seed).permutation(len(x))
        split = max(2, min(len(x) - 1, int(0.8 * len(x))))
        train, validation = order[:split], order[split:]
        self.models, self.weights = [], []
        for objective in range(y.shape[1]):
            members = [
                RadialSurrogate(x[train], y[train, objective], seed=seed, config=config),
                make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()).fit(
                    x[train], y[train, objective]
                ),
                make_pipeline(
                    StandardScaler(),
                    GaussianProcessRegressor(kernel=1.0 * RBF(1.0), random_state=0),
                ).fit(x[train], y[train, objective]),
            ]
            predictions = np.clip(np.asarray([m.predict(x[validation]) for m in members]), 0, 1)
            errors = np.abs(predictions - y[validation, objective]).mean(axis=1)
            total = errors.sum()
            weights = (total - errors) / (2 * total) if total > 0 else np.full(3, 1 / 3)
            self.models.append(members)
            self.weights.append(weights)
        self.weights = np.asarray(self.weights)

    def predict(self, x):
        means, spread = [], []
        for members, weights in zip(self.models, self.weights):
            values = np.clip(np.asarray([m.predict(x) for m in members]), 0, 1)
            means.append(weights @ values)
            spread.append(values.max(axis=0) - values.min(axis=0))
        return np.asarray(means).T, np.asarray(spread).T


def mixed_distance(x, kind):
    domain = DOMAINS[kind]
    continuous = len(domain.bounds)
    distance = np.abs(x[:, None, :continuous] - x[None, :, :continuous]).sum(-1)
    offset = continuous
    for options in domain.categories.values():
        end = offset + len(options)
        distance += np.any(x[:, None, offset:end] != x[None, :, offset:end], axis=-1)
        offset = end
    return distance / (continuous + len(domain.categories))


def local_clusters(x, y, objective, kind, *, config=DEFAULT):
    count = min(len(x), max(config.min_cluster_size, int(np.ceil(len(x) * config.elite_fraction))))
    indices = np.argsort(y[:, objective], kind="stable")[:count]
    if len(indices) <= config.min_cluster_size:
        return [indices]
    # sklearn includes the point itself in min_samples; the authors' hdbscan excludes it.
    labels = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_cluster_size + 1,
        metric="precomputed",
    ).fit_predict(mixed_distance(x[indices], kind))
    clusters = [indices[labels == label] for label in sorted(set(labels) - {-1})]
    return clusters if clusters else [indices]


def objectives(feedback):
    return np.array(
        [
            1 - int(feedback.collision),
            min(100, max(0, feedback.safety.min_clearance_m)) / 100,
            min(10, feedback.safety.min_ttc_s) / 10,
        ]
    )


class PoolSearch:
    def __init__(self, candidates, *, seed=11, config=DEFAULT):
        self.candidates = candidates
        self.x = features(candidates)
        self.kind, self.seed, self.config = candidates[0].kind, seed, config
        self.indices, self.values, self.pending = [], [], []
        self.covered, self.restarts = set(), []
        self.phase = "global"

    def observe(self, index, feedback):
        value = objectives(feedback)
        self.indices.append(index)
        self.values.append(value)
        self.covered.update(np.flatnonzero(value <= np.asarray(self.config.thresholds)).tolist())

    def propose(self, available):
        fitting_s = 0.0
        if not self.pending:
            if len(self.covered) == len(self.config.thresholds):
                self.restarts.append(
                    {
                        "after_paid_queries": len(self.indices),
                        "covered_objectives": sorted(self.covered),
                    }
                )
                self.covered.clear()
            active = [j for j in range(3) if j not in self.covered]
            candidates = sorted(available, key=lambda i: self.candidates[i].scenario_id)
            x, y = self.x[self.indices], np.asarray(self.values)
            proposals = []
            if self.phase == "global":
                started = time.perf_counter()
                model = Committee(x, y, seed=self.seed, config=self.config)
                fitting_s += time.perf_counter() - started
                mean, uncertainty = model.predict(self.x[candidates])
                order = preference_order(mean, active)
                best = {int(np.argmin(mean[:, j])) for j in active}
                proposals.extend(candidates[i] for i in order if i in best)
                proposals.extend(candidates[int(np.argmax(uncertainty[:, j]))] for j in active)
            else:
                for objective in active:
                    for cluster in local_clusters(x, y, objective, self.kind, config=self.config):
                        started = time.perf_counter()
                        local = RadialSurrogate(
                            x[cluster], y[cluster, objective], seed=self.seed, config=self.config
                        )
                        fitting_s += time.perf_counter() - started
                        lower, upper = x[cluster].min(0), x[cluster].max(0)
                        # Project a bounded local search to the closest available pool region.
                        points = self.x[candidates]
                        outside = np.maximum(lower - points, 0) + np.maximum(points - upper, 0)
                        distance = (outside**2).sum(-1)
                        region = np.flatnonzero(distance <= distance.min() + 1e-12)
                        values = local.predict(points[region])
                        proposals.append(candidates[int(region[int(np.argmin(values))])])
            self.pending = [(index, self.phase) for index in dict.fromkeys(proposals)]
            self.phase = "local" if self.phase == "global" else "global"
        index, phase = self.pending.pop(0)
        return index, phase, fitting_s


def run(candidates, oracle, *, initial_indices, budget=128, seed=11, config=DEFAULT):
    initial = list(initial_indices)
    validate_initial_indices(len(candidates), initial, budget, minimum=3)
    torch.set_num_threads(1)
    search = PoolSearch(candidates, seed=seed, config=config)
    available, queries, selected = set(range(len(candidates))), [], []
    for step in range(budget):
        started = time.perf_counter()
        index, phase, fitting_s = (
            (initial[step], "common_initialization", 0.0)
            if step < len(initial)
            else search.propose(available)
        )
        selection_s = time.perf_counter() - started - fitting_s
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError("SAMOTA feedback differs from the selected paid coordinate")
        search.observe(index, feedback)
        selected.append(index)
        available.remove(index)
        queries.append(
            {
                "query": step + 1,
                "index": index,
                "feedback": feedback.to_dict(),
                "prequery_probability": None,
                "selection_s": selection_s,
                "adaptation_s": fitting_s,
                "reason": phase,
                "covered_objectives": sorted(search.covered),
                "coverage_restarts": len(search.restarts),
            }
        )
    return {
        "method": "samota",
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(queries),
        "maximum_queries": budget,
        "termination": "budget_exhausted",
        "config": asdict(config),
        "coverage_restarts": search.restarts,
        "feedback_used": ["paid collision Boolean", "paid clearance", "paid minimum TTC"],
        "adaptation": "finite pool exact global/local acquisition; restart fully covered objectives",
    }
