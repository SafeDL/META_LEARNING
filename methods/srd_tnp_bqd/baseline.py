"""Eight named baseline selectors on the unchanged S01 candidate pool."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import ndtr, owens_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from .benchmark import BUDGET, CONFIG, GRID_BINS
from .qd import RiskArchive


METHODS = {
    "uniform_random": "Uniform random",
    "farthest_first": "Farthest-first",
    "knn_history": "kNN historical-risk ranking",
    "gp_ucb": "GP-UCB",
    "gp_ei": "GP-EI",
    "bop_elites": "BOP-Elites",
    "bas": "BAS",
    "rf_bo": "RF surrogate BO",
}
SETTINGS = CONFIG["baselines"]


def expected_improvement(mean, variance, incumbent):
    sd = np.sqrt(np.maximum(variance, 0))
    delta = mean - incumbent
    standardized = np.divide(delta, sd, out=np.zeros_like(delta), where=sd > 0)
    ei = delta * ndtr(standardized) + sd * np.exp(-0.5 * standardized ** 2) / np.sqrt(2 * np.pi)
    return np.where(sd > 0, np.maximum(ei, 0), np.maximum(delta, 0))


def expected_bernoulli_variance(standardized_threshold, explained_variance_fraction):
    """Sinha et al., Appendix A.2: Phi2(a,-a; rho=-r²).

    The symmetric bivariate-normal CDF equals 2*T(a,sqrt((1-r²)/(1+r²))).
    Owen's T evaluates the paper's expression without stochastic CDF integration.
    """
    fraction = np.clip(explained_variance_fraction, 0., 1.)
    return 2 * owens_t(standardized_threshold, np.sqrt((1 - fraction) / (1 + fraction)))


def bas_scores(gp, threshold):
    """One-step reduction of integrated Bernoulli variance over the full fixed pool."""
    variance = gp.variance()
    sd = np.sqrt(variance)
    standardized = np.divide(threshold - gp.mean, sd,
                             out=np.full_like(sd, np.inf), where=sd > 0)
    baseline = ndtr(standardized) * ndtr(-standardized)
    denominator = variance[:, None] * (variance[None, :] + gp.settings["noise_variance"] + gp.settings["jitter"])
    fraction = np.divide(gp.covariance ** 2, denominator,
                         out=np.zeros_like(denominator), where=denominator > 0)
    expected = expected_bernoulli_variance(standardized[:, None], fraction)
    return (baseline[:, None] - expected).mean(axis=0)


def jackknife_variance(full_prediction, fold_predictions):
    """Huang et al., equations (9)-(11), using delete-one-fold pseudo-values."""
    count = len(fold_predictions)
    if count < 2:
        raise ValueError("Jackknife needs at least two folds")
    pseudo = count * full_prediction - (count - 1) * fold_predictions
    return ((pseudo - pseudo.mean(axis=0)) ** 2).sum(axis=0) / (count * (count - 1))


def rf_posterior(x, indices, responses, settings, seed):
    """Fit only disclosed responses; fixed RF parameters and ten-fold Jackknife."""
    inputs, values = x[indices], np.asarray(responses)
    params = {"n_estimators": settings["rf_trees"], "max_depth": settings["rf_max_depth"],
              "min_samples_split": settings["rf_min_samples_split"], "random_state": seed, "n_jobs": 1}
    prediction = RandomForestRegressor(**params).fit(inputs, values).predict(x)
    folds = KFold(n_splits=min(settings["rf_jackknife_folds"], len(indices)),
                  shuffle=True, random_state=seed)
    deleted = np.asarray([RandomForestRegressor(**params).fit(inputs[train], values[train]).predict(x)
                          for train, _ in folds.split(inputs)])
    return prediction, jackknife_variance(prediction, deleted)


class GaussianRiskGP:
    """Exact fixed Matérn-5/2 GP on raw measured R, not collision labels."""
    def __init__(self, x, settings=None):
        self.settings = SETTINGS if settings is None else settings
        distance = np.sqrt(5) * cdist(x, x) / self.settings["lengthscale"]
        self.covariance = self.settings["variance"] * (
            1 + distance + distance ** 2 / 3
        ) * np.exp(-distance)
        self.mean = np.full(len(x), self.settings["mean"], dtype=float)

    def observe(self, index, risk):
        cross = self.covariance[:, index].copy()
        denominator = cross[index] + self.settings["noise_variance"] + self.settings["jitter"]
        self.mean += cross * ((risk - self.mean[index]) / denominator)
        self.covariance -= np.outer(cross, cross) / denominator

    def variance(self):
        variance = np.diag(self.covariance)
        if variance.min() < -1e-8:
            raise FloatingPointError("negative GP latent variance")
        return np.maximum(variance, 0)


def select(method, x, cells, oracle, seed, budget=None, settings=None,
           history_scores=None, cell_count=256):
    if method not in METHODS:
        raise ValueError(f"unknown baseline: {method}")
    if method == "knn_history" and history_scores is None:
        raise ValueError("kNN requires scores from the allowed historical sources")
    budget = BUDGET if budget is None else budget
    settings = SETTINGS if settings is None else settings
    rng = np.random.default_rng(seed)
    remaining = np.ones(len(x), bool)
    nearest_distance = np.full(len(x), np.inf)
    rank = history_scores if method == "knn_history" else None
    gp = GaussianRiskGP(x, settings) if method in ("gp_ucb", "gp_ei", "bop_elites", "bas") else None
    initial = rng.choice(len(x), size=settings["initial_queries"], replace=False) if gp or method == "rf_bo" else []
    archive = RiskArchive(cell_count=cell_count)
    best_risk = -np.inf
    selected, records, observed_risks = [], [], []
    for step in range(budget):
        if method == "uniform_random":
            index = int(rng.choice(np.flatnonzero(remaining)))
        elif method == "farthest_first":
            index = int(rng.integers(len(x))) if step == 0 else int(np.argmax(nearest_distance))
        elif method == "knn_history":
            score = rank.copy()
            score[~remaining] = -np.inf
            index = int(np.argmax(score))
        elif step < len(initial):
            index = int(initial[step])
        elif method == "rf_bo":
            mean, variance = rf_posterior(x, selected, observed_risks, settings, seed)
            score = expected_improvement(mean, variance, best_risk)
            score[~remaining] = -np.inf
            index = int(np.argmax(score))
        else:
            variance = gp.variance()
            if method == "gp_ucb":
                beta = 2 * np.log(len(x) * (step + 1) ** 2 * np.pi ** 2 / (6 * settings["ucb_delta"]))
                score = gp.mean + np.sqrt(beta * variance)
            elif method == "gp_ei":
                score = expected_improvement(gp.mean, variance, best_risk)
            elif method == "bas":
                score = bas_scores(gp, settings["bas_threshold"])
            else:
                # Known descriptor: region membership probability is exactly one.
                bound = archive.threshold + archive.elite_quality[cells]
                score = expected_improvement(gp.mean, variance, bound)
            score[~remaining] = -np.inf
            index = int(np.argmax(score))
        observation = oracle.query(index)
        remaining[index] = False
        selected.append(index)
        if not observation.valid_risk:
            raise ValueError("fixed benchmark contains an invalid measured risk")
        if gp is not None:
            gp.observe(index, observation.risk)
        observed_risks.append(observation.risk)
        best_risk = max(best_risk, observation.risk)
        archive.observe(int(cells[index]), observation.risk)
        if method == "farthest_first":
            distance = np.sum((x - x[index]) ** 2, axis=1)
            nearest_distance = np.minimum(nearest_distance, distance)
            nearest_distance[~remaining] = -np.inf
        records.append({
            "query_number": step + 1, "index": index, "scenario_id": observation.scenario_id,
            "risk": observation.risk, "label": observation.collision, "valid_risk": observation.valid_risk,
            "cell": int(cells[index]), "risk_archive": archive.metrics(),
        })
    return {"method": METHODS[method], "method_id": method, "seed": seed,
            "selected_indices": selected, "queries": records,
            "feedback": "continuous risk; collision label is recorded only for evaluation",
            "protocol": {"budget": budget, "grid_bins": GRID_BINS, **settings}}
