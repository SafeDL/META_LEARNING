"""Huang et al. (T-ITS 2025), RF, grouped Jackknife, and expected improvement."""

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import optuna
from scipy.special import ndtr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from experiments.scenarios import features
from testing_methods.selection import validate_initial_indices


@dataclass(frozen=True)
class Config:
    jackknife_folds: int = 10
    tune_every: int = 50
    tuning_trials: int = 20
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    ttc_horizon_s: float = 10.0


DEFAULT = Config()


def risk(feedback, *, ttc_horizon_s=10.0):
    if feedback.collision:
        delta_v = feedback.safety.impact_delta_v_mps
        if delta_v is None:
            raise ValueError("RF-EI requires measured impact delta-V on a paid collision")
        return delta_v
    # The finite simulation window censors TTC beyond its observable horizon.
    return -min(feedback.safety.min_ttc_s, ttc_horizon_s)


def jackknife_variance(full_prediction, fold_predictions):
    """Eqs. (9)-(11), including the pseudo-value scaling."""
    predictions = np.asarray(fold_predictions, dtype=float)
    k = len(predictions)
    if k < 2:
        raise ValueError("Grouped Jackknife requires at least two fitted folds")
    pseudo = k * np.asarray(full_prediction) - (k - 1) * predictions
    return ((pseudo - pseudo.mean(axis=0)) ** 2).sum(axis=0) / (k * (k - 1))


def expected_improvement(mean, variance, best):
    mean, variance = np.broadcast_arrays(
        np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)
    )
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError("EI requires finite means and nonnegative variances")
    sigma = np.sqrt(variance)
    improvement = mean - best
    z = np.divide(improvement, sigma, out=np.zeros_like(mean), where=sigma > 0)
    ei = improvement * ndtr(z) + sigma * np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    return np.maximum(0, np.where(sigma > 0, ei, np.maximum(improvement, 0)))


class RandomForestEI:
    def __init__(self, *, seed=11, config=DEFAULT):
        self.seed = seed
        self.config = config
        self.parameters = {
            name: getattr(config, name)
            for name in ("n_estimators", "max_depth", "min_samples_split")
        }
        self.tuning = []

    def _forest(self, parameters):
        return RandomForestRegressor(**parameters, random_state=self.seed, n_jobs=1)

    def tune(self, x, y):
        folds = list(KFold(n_splits=10, shuffle=True, random_state=self.seed).split(x))

        def objective(trial):
            parameters = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            errors = []
            for train, validation in folds:
                model = self._forest(parameters).fit(x[train], y[train])
                errors.append(
                    float(np.sqrt(np.mean((model.predict(x[validation]) - y[validation]) ** 2)))
                )
            return float(np.mean(errors))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(
            seed=self.seed + len(y), n_startup_trials=10, multivariate=False, constant_liar=False
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.enqueue_trial(self.parameters)
        study.optimize(objective, n_trials=self.config.tuning_trials)
        self.parameters = dict(study.best_params)
        self.tuning.append(
            {
                "paid_observations": len(y),
                "parameters": self.parameters.copy(),
                "mean_fold_rmse": study.best_value,
                "trials": len(study.trials),
            }
        )

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(y) < 2 or len(x) != len(y) or not np.isfinite(y).all():
            raise ValueError("RF-EI requires at least two finite paid observations")
        if len(y) % self.config.tune_every == 0:
            self.tune(x, y)
        self.model = self._forest(self.parameters).fit(x, y)
        k = min(self.config.jackknife_folds, len(y))
        folds = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        self.fold_models = [
            self._forest(self.parameters).fit(x[train], y[train]) for train, _ in folds.split(x)
        ]
        self.best = float(y.max())
        return self

    def predict(self, x):
        mean = self.model.predict(x)
        folds = np.asarray([model.predict(x) for model in self.fold_models])
        variance = jackknife_variance(mean, folds)
        return mean, variance, expected_improvement(mean, variance, self.best)


def run(candidates, oracle, *, initial_indices, budget=128, seed=11, config=DEFAULT):
    initial = list(initial_indices)
    size = len(candidates)
    validate_initial_indices(len(candidates), initial, budget, minimum=2)
    x = features(candidates)
    model = RandomForestEI(seed=seed, config=config)
    available = set(range(size))
    selected, values, queries = [], [], []
    for step in range(budget):
        started = time.perf_counter()
        prediction = {}
        if step < len(initial):
            index = initial[step]
        else:
            indices = sorted(available, key=lambda i: candidates[i].scenario_id)
            mean, variance, ei = model.predict(x[indices])
            position = int(np.argmax(ei))
            index = indices[position]
            prediction = {
                "prequery_risk_mean": float(mean[position]),
                "prequery_risk_variance": float(variance[position]),
                "expected_improvement": float(ei[position]),
            }
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError("RF-EI received feedback from a different paid scenario")
        selected.append(index)
        available.remove(index)
        values.append(risk(feedback, ttc_horizon_s=config.ttc_horizon_s))
        started = time.perf_counter()
        if step + 1 >= len(initial) and step + 1 < budget:
            model.fit(x[selected], values)
        adaptation_s = time.perf_counter() - started
        queries.append(
            {
                "query": step + 1,
                "index": index,
                "feedback": feedback.to_dict(),
                "prequery_probability": None,
                **prediction,
                "risk": values[-1],
                "selection_s": selection_s,
                "adaptation_s": adaptation_s,
                "reason": "common_initialization" if step < len(initial) else "rf_ei_pool_maximum",
            }
        )
    return {
        "method": "av_bo_rf_ei",
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(queries),
        "maximum_queries": budget,
        "termination": "budget_exhausted",
        "config": asdict(config),
        "tuning": model.tuning,
        "feedback_used": ["paid collision Boolean", "paid impact delta-V", "paid minimum TTC"],
        "adaptation": "finite candidate pool; exact acquisition maximization",
    }
