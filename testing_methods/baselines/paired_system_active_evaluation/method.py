"""Parashar et al., arXiv:2608.13719v1, Algorithm 1 and Appendices C/D/I."""

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.special import expit, xlogy
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from experiments.scenarios import features
from testing_methods.selection import validate_initial_indices


@dataclass(frozen=True)
class Config:
    batch_size: int = 5
    shortlist_size: int = 64
    training_epochs: int = 1000
    learning_rate: float = 0.001
    mc_draws: int = 32
    local_samples: int = 64
    paired_samples: int = 16
    maximum_clusters: int = 10
    existing_support_probability: float = 0.01
    failure_threshold: float = 0.5


DEFAULT = Config()


def binary_entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    return -xlogy(p, p) - xlogy(1 - p, 1 - p)


def support_information(cluster_probabilities, discovery_probabilities):
    """Eq. (3), using the equivalent H(R) - E_Z H(R|Z)."""
    p = np.asarray(cluster_probabilities)
    q = np.asarray(discovery_probabilities)
    marginal = (p * q).sum(axis=-1)
    return np.maximum(0, binary_entropy(marginal) - (p * binary_entropy(q)).sum(axis=-1))


def control_variate(paired_target, paired_proxy, additional_proxy):
    """Eq. (2), including the k/(k+n) coefficient for independent proxy-only samples."""
    target, proxy, additional = map(np.asarray, (paired_target, paired_proxy, additional_proxy))
    n, k = len(target), len(additional)
    if n < 2 or len(proxy) != n or k < 1:
        raise ValueError(
            "Local control variates need at least two paired and one additional samples"
        )
    variance = float(np.var(np.r_[proxy, additional], ddof=1))
    covariance = float(np.cov(target, proxy, ddof=1)[0, 1])
    beta = k / (k + n) * covariance / variance if variance > 0 else 0.0
    value = float(np.mean(target - beta * proxy) + beta * additional.mean())
    return value, beta


def support_model(x, observed, *, seed=11, config=DEFAULT):
    support = x[observed]
    upper = min(config.maximum_clusters, max(1, len(support) // 2), len(np.unique(support, axis=0)))
    models = [
        KMeans(n_clusters=k, n_init=10, random_state=seed).fit(support) for k in range(1, upper + 1)
    ]
    if upper > 2 and models[0].inertia_ > models[-1].inertia_:
        fraction = np.asarray([m.inertia_ for m in models])
        fraction = (fraction - fraction[-1]) / (fraction[0] - fraction[-1])
        elbow = int(np.argmax(np.linspace(1, 0, upper) - fraction))
    else:
        elbow = 0
    model = models[elbow]
    spread = np.array(
        [
            np.sqrt(np.mean(np.sum((support[model.labels_ == k] - center) ** 2, axis=-1)))
            for k, center in enumerate(model.cluster_centers_)
        ]
    )
    spread = np.maximum(spread, 1e-8)
    distance = cdist(x, model.cluster_centers_)
    weight = np.exp(-0.5 * (distance / spread) ** 2)
    prior = np.column_stack((weight, np.ones(len(x))))
    prior /= prior.sum(axis=1, keepdims=True)
    nearest = cdist(support, support)
    np.fill_diagonal(nearest, np.inf)
    nearest = nearest.min(axis=1)
    rho, softness = float(np.median(nearest)), max(1e-8, float(np.std(nearest)))
    boundary = np.maximum(distance - spread, 0).min(axis=1)
    existing = config.existing_support_probability * (1 - weight)
    new = expit((boundary - rho) / softness)
    likelihood = np.column_stack((existing, new))
    return support_information(prior, likelihood), {
        "clusters": len(spread),
        "novelty_radius": rho,
        "softness": softness,
    }


class TargetSurrogate(nn.Module):
    def __init__(self, dimension, *, config=DEFAULT):
        super().__init__()
        modules = []
        for width in (96, 24, 6):
            modules.extend(
                [nn.Linear(dimension, width), nn.ELU(), nn.LayerNorm(width), nn.Dropout(0.1)]
            )
            dimension = width
        modules.append(nn.Linear(6, 1))
        self.network = nn.Sequential(*modules)
        self.config = config
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def fit(self, x, labels):
        self.train()
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
        for _ in range(self.config.training_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(self(x), y)
            loss.backward()
            self.optimizer.step()

    def statistics(self, x):
        self.train()  # MC dropout remains enabled; LayerNorm has no running statistics.
        inputs = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        predictions = self(inputs.repeat(self.config.mc_draws, 1)).sigmoid()
        predictions = predictions.reshape(self.config.mc_draws, len(x))
        mean, sigma = predictions.mean(0), predictions.std(0, unbiased=False)
        gradient = torch.autograd.grad(mean.sum(), inputs)[0]
        return (
            mean.detach().numpy(),
            sigma.detach().numpy(),
            gradient.detach().norm(dim=-1).numpy(),
            predictions[0].detach().numpy(),
        )


def choose_batch(x, observed, proxy, model, count, generator, *, seed=11, config=DEFAULT):
    available = np.array(sorted(set(range(len(x))) - set(observed)))
    mean, sigma, gradient, target_sample = model.statistics(x)
    information, support = support_model(x, observed, seed=seed, config=config)
    order = np.argsort(-information[available], kind="stable")
    shortlist = available[order[: config.shortlist_size]]
    distance = cdist(x, x, metric="sqeuclidean")
    near = distance.copy()
    np.fill_diagonal(near, np.inf)
    r_min, r_max = float(np.median(near.min(axis=1))), float(distance.max())
    sensitivity = gradient / (sigma + 1e-8)
    normalized = (sensitivity - sensitivity.min()) / (np.ptp(sensitivity) + 1e-8)
    radii = r_min + (r_max - r_min) / (1 + np.exp(normalized))
    scores, coefficients = [], []
    for index in shortlist:
        # A minimum of four public neighbours makes the local sample covariance estimable.
        nearest = np.argsort(distance[index], kind="stable")
        radius = max(radii[index], distance[index, nearest[min(3, len(x) - 1)]])
        region = np.flatnonzero(distance[index] <= radius)
        region = generator.permutation(region)[: config.local_samples]
        n = min(config.paired_samples, len(region) // 2)
        paired, additional = region[:n], region[n:]
        score, beta = control_variate(target_sample[paired], proxy[paired], proxy[additional])
        scores.append(score)
        coefficients.append(beta)
    scores, coefficients = np.asarray(scores), np.asarray(coefficients)
    eligible = np.flatnonzero(scores >= config.failure_threshold)
    filled = max(0, count - len(eligible))
    if filled:
        excluded = np.flatnonzero(scores < config.failure_threshold)
        order = np.argsort(-scores[excluded], kind="stable")
        eligible = np.r_[eligible, excluded[order[:filled]]]
    clusters = min(count, len(np.unique(x[shortlist[eligible]], axis=0)))
    cluster = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(x[shortlist[eligible]])
    chosen = []
    for k, center in enumerate(cluster.cluster_centers_):
        members = eligible[cluster.labels_ == k]
        best = int(members[np.argmin(((x[shortlist[members]] - center) ** 2).sum(-1))])
        chosen.append(best)
    if len(chosen) < count:
        remaining = sorted(
            set(eligible) - set(chosen), key=lambda j: (-scores[j], int(shortlist[j]))
        )
        chosen.extend(remaining[: count - len(chosen)])
    return [
        (
            int(shortlist[j]),
            {
                "prequery_probability": float(mean[shortlist[j]]),
                "control_variate_risk": float(scores[j]),
                "control_variate_beta": float(coefficients[j]),
                "support_information": float(information[shortlist[j]]),
                "batch_threshold_fill": filled,
                "batch_geometry_fill": count - clusters,
                **support,
            },
        )
        for j in chosen
    ]


def run(candidates, oracle, *, proxy, initial_indices, budget=128, seed=11, config=DEFAULT):
    initial = list(initial_indices)
    validate_initial_indices(len(candidates), initial, budget, minimum=2)
    x, proxy = features(candidates), np.asarray(proxy, dtype=float)
    if proxy.shape != (len(candidates),) or not np.isfinite(proxy).all():
        raise ValueError(
            "The predeclared proxy supplies one finite prediction per public candidate"
        )
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = TargetSurrogate(x.shape[1], config=config)
    selected, labels, queries, pending, fits = [], [], [], [], []
    batch_id = 0
    for step in range(budget):
        started = time.perf_counter()
        if step < len(initial):
            index, details = initial[step], {"prequery_probability": None}
        else:
            if not pending:
                batch_id += 1
                pending = choose_batch(
                    x,
                    selected,
                    proxy,
                    model,
                    min(config.batch_size, budget - step),
                    generator,
                    seed=seed,
                    config=config,
                )
            index, details = pending.pop(0)
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError("Paired evaluation feedback differs from its paid coordinate")
        selected.append(index)
        labels.append(feedback.collision)
        started = time.perf_counter()
        if (
            step + 1 == len(initial) or (step >= len(initial) and not pending)
        ) and step + 1 < budget:
            model.fit(x[selected], labels)
            fits.append(len(selected))
        queries.append(
            {
                "query": step + 1,
                "index": index,
                "feedback": feedback.to_dict(),
                **details,
                "batch": batch_id,
                "selection_s": selection_s,
                "adaptation_s": time.perf_counter() - started,
                "reason": "common_initialization" if step < len(initial) else "support_cv_batch",
            }
        )
    return {
        "method": "paired_system_active_evaluation",
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(queries),
        "maximum_queries": budget,
        "termination": "budget_exhausted",
        "config": asdict(config),
        "target_fit_sample_counts": fits,
        "feedback_used": ["fixed historical proxy predictions", "paid collision Boolean"],
        "adaptation": "Boolean risk; fixed historical proxy; finite-pool threshold batch completion",
    }
