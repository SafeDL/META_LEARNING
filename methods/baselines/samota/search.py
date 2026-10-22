"""SAMOTA's RBF/PR/Kriging committee, MOSA preference and clustered local search."""

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


CONFIG = {"population_size": 32, "generations": 100, "local_generations": 60,
          "elite_fraction": 0.2, "rbf_centers": 10, "thresholds": [0.0, 0.0, 0.2]}


class RadialSurrogate:
    def __init__(self, x, y, *, centers=10, seed=11):
        self.centers = KMeans(n_clusters=min(centers, len(x)), n_init=1,
                             random_state=seed).fit(x).cluster_centers_
        distances = pdist(self.centers)
        self.width = max(0.05, float(np.median(distances)) if len(distances) else 0.3)
        self.regression = Ridge(alpha=1e-3).fit(self.basis(x), y)

    def basis(self, x):
        return np.exp(-cdist(x, self.centers)**2 / (2 * self.width**2))

    def predict(self, x):
        return self.regression.predict(self.basis(x))


class Committee:
    def __init__(self, x, y, generator, *, centers=10, seed=11):
        order = generator.permutation(len(x))
        split = max(2, min(len(x) - 1, int(0.8 * len(x))))
        train, validation = order[:split], order[split:]
        self.models = [RadialSurrogate(x[train], y[train], centers=centers, seed=seed),
                       make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-3)).fit(x[train], y[train]),
                       GaussianProcessRegressor(kernel=RBF(0.3, length_scale_bounds="fixed"),
                                                alpha=1e-5, optimizer=None, normalize_y=True).fit(x[train], y[train])]
        errors = np.array([np.mean(np.abs(m.predict(x[validation]) - y[validation]), axis=0)
                           for m in self.models])
        total = errors.sum(0)
        self.weights = np.divide(0.5 * (total[None] - errors), total[None],
                                 out=np.full_like(errors, 1 / 3), where=total[None] > 1e-12)

    def predict(self, x):
        predictions = np.array([model.predict(x) for model in self.models])
        return (predictions * self.weights[:, None]).sum(0), predictions.max(0) - predictions.min(0)


def preference_order(values, active):
    selected = []
    for objective in active:
        best = int(np.argmin(values[:, objective]))
        if best not in selected:
            selected.append(best)
    remaining = set(range(len(values))) - set(selected)
    objectives = values[:, active]
    dominance = (np.all(objectives[:, None] <= objectives[None, :], axis=2)
                 & np.any(objectives[:, None] < objectives[None, :], axis=2))
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
                    crowding[ordered[position]] += (values[ordered[position + 1], objective]
                                                    - values[ordered[position - 1], objective]) / span
        selected.extend(sorted(front, key=lambda i: (-crowding[i], i)))
        remaining.difference_update(front)
    return selected


class Samota:
    def __init__(self, space, *, config=CONFIG):
        self.space = space
        self.config = config
        self.scenes, self.values = [], []
        self.covered = set()
        self.pending = []
        self.global_phase = True

    @staticmethod
    def objectives(feedback):
        return np.array([1 - int(feedback.collision),
                         min(100, max(0, feedback.safety.min_clearance_m)) / 100,
                         min(10, feedback.safety.min_ttc_s) / 10])

    def observe(self, scenario, feedback):
        value = self.objectives(feedback)
        self.scenes.append(scenario)
        self.values.append(value)
        self.covered.update(np.flatnonzero(value <= np.asarray(self.config["thresholds"])).tolist())
        return {"covered_objectives": sorted(self.covered)}

    def _evolve(self, population, predict, active, generations, *, bounds=None):
        best, uncertain = {}, {}
        size = len(population)
        for _ in range(generations):
            offspring = []
            for _ in range(size):
                a, b = self.space.generator.integers(size, size=2)
                first, second = self.space.vector(population[a]), self.space.vector(population[b])
                cut = int(self.space.generator.integers(1, self.space.dimension))
                vector = np.r_[first[:cut], second[cut:]]
                if bounds is not None:
                    vector = np.clip(vector, bounds[0], bounds[1])
                crossed = self.space.decode(vector)
                if not self.space.valid(crossed):
                    crossed = population[a]
                child = self.space.mutate(crossed)
                if bounds is not None:
                    child = self.space.decode(np.clip(self.space.vector(child), bounds[0], bounds[1]))
                if self.space.valid(child):
                    offspring.append(child)
            combined = population + offspring
            x = np.array([self.space.vector(s) for s in combined])
            means, spread = predict(x)
            for objective in active:
                i, j = int(np.argmin(means[:, objective])), int(np.argmax(spread[:, objective]))
                if objective not in best or means[i, objective] < best[objective][0]:
                    best[objective] = (means[i, objective], combined[i])
                if objective not in uncertain or spread[j, objective] > uncertain[objective][0]:
                    uncertain[objective] = (spread[j, objective], combined[j])
            population = [combined[i] for i in preference_order(means, active)[:size]]
        return [best[j][1] for j in active] + [uncertain[j][1] for j in active]

    def _global(self, active):
        x = np.array([self.space.vector(s) for s in self.scenes])
        committee = Committee(x, np.array(self.values), self.space.generator,
                              centers=self.config["rbf_centers"], seed=self.space.seed)
        population = [self.space.random() for _ in range(self.config["population_size"])]
        return self._evolve(population, committee.predict, active, self.config["generations"])

    def _local(self, active):
        x, y = np.array([self.space.vector(s) for s in self.scenes]), np.array(self.values)
        proposals = []
        for objective in active:
            count = min(len(x), max(6, int(np.ceil(len(x) * self.config["elite_fraction"])) ))
            indices = np.argsort(y[:, objective], kind="stable")[:count]
            labels = HDBSCAN(min_cluster_size=2, min_samples=2).fit_predict(x[indices])
            clusters = [indices[labels == label] for label in sorted(set(labels) - {-1})]
            if not clusters:
                clusters = [indices]
            for cluster in clusters:
                radial = RadialSurrogate(x[cluster], y[cluster, objective],
                                          centers=self.config["rbf_centers"], seed=self.space.seed)
                lower = np.maximum(0, x[cluster].min(0) - 0.05)
                upper = np.minimum(1, x[cluster].max(0) + 0.05)
                population = []
                while len(population) < 16:
                    point = self.space.decode(self.space.generator.uniform(lower, upper))
                    if self.space.valid(point):
                        population.append(point)
                def predict(points):
                    values = radial.predict(points).reshape(-1, 1)
                    return values, np.zeros_like(values)
                proposals.append(self._evolve(population, predict, [0], self.config["local_generations"],
                                               bounds=(lower, upper))[0])
        return proposals

    def propose(self, remaining):
        while not self.pending:
            active = [j for j in range(3) if j not in self.covered] or [0, 1, 2]
            proposals = self._global(active) if self.global_phase else self._local(active)
            phase = "global_committee" if self.global_phase else "clustered_local_rbf"
            self.global_phase = not self.global_phase
            coordinates = set()
            for scene in proposals:
                if self.space.valid(scene) and scene.coordinate() not in coordinates:
                    self.pending.append((scene, phase))
                    coordinates.add(scene.coordinate())
        scene, phase = self.pending.pop(0)
        if not self.space.valid(scene):
            return self.propose(remaining)
        return scene, None, {"search": "samota", "phase": phase,
                              "covered_objectives": sorted(self.covered)}
