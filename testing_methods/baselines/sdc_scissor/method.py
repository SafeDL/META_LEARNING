"""Adaptive SDC-Scissor with risk-ranked queries on a fixed traffic suite."""

import time

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from testing_methods.baselines.detour.method import parameter_features


SETTINGS = {
    "features": "registered_numeric_ranges_and_binary_side",
    "label": "native_collision",
    "penalty": "l2", "C": 1.0, "solver": "lbfgs", "tol": 1e-4, "max_iter": 100,
    "oversampling_seed": 0, "oversampling": "minority_with_replacement_per_fit",
    "history": "all_admitted_executions", "update": "refit_after_each_paid_query",
    "tie_break": "lowest_unqueried_index", "sklearn_version": sklearn.__version__,
}


def fit_classifier(features, labels):
    groups = [np.flatnonzero(labels == label) for label in (False, True)]
    minority, majority = sorted(groups, key=len)
    if not len(minority):
        return None
    extra = np.random.default_rng(SETTINGS["oversampling_seed"]).choice(
        minority, size=len(majority) - len(minority), replace=True)
    indices = np.concatenate((np.arange(len(labels)), extra))
    model = LogisticRegression(**{k: SETTINGS[k] for k in ("penalty", "C", "solver", "tol", "max_iter")})
    return model.fit(features[indices], labels[indices])


def select(candidates, oracle, *, history, initial_indices):
    started = time.perf_counter()
    candidates_x = np.stack([parameter_features(scene) for scene in candidates])
    n_history = len(history)
    capacity = n_history + len(candidates)
    train_x = np.empty((capacity, candidates_x.shape[1]), dtype=float)
    train_y = np.empty(capacity, dtype=bool)
    for i, (scene, collision) in enumerate(history):
        train_x[i], train_y[i] = parameter_features(scene), collision
    available = np.ones(len(candidates), dtype=bool)
    selected, queries = [], []
    ranking_s = time.perf_counter() - started

    def observe(index, purpose):
        feedback = oracle.query(index, purpose=purpose)
        pos = n_history + len(selected)
        train_x[pos], train_y[pos] = candidates_x[index], feedback.collision
        available[index] = False
        selected.append(index)
        row = {"index": index, "query": len(selected), "feedback": feedback.to_dict()}
        queries.append(row)
        return row

    for index in initial_indices:
        observe(index, "initial")
    while not oracle.stopped and available.any():
        n_train = n_history + len(selected)
        started = time.perf_counter()
        model = fit_classifier(train_x[:n_train], train_y[:n_train])
        adaptation_s = time.perf_counter() - started
        started = time.perf_counter()
        remaining = np.flatnonzero(available)
        if model is None:
            index, probability = int(remaining[0]), float(train_y[n_train - 1])
        else:
            probabilities = model.predict_proba(candidates_x[remaining])[:, 1]
            best = int(np.argmax(probabilities))
            index, probability = int(remaining[best]), float(probabilities[best])
        ranking_s += time.perf_counter() - started
        row = observe(index, "target")
        row.update(prequery_probability=probability, adaptation_s=adaptation_s)
    return {"method": "sdc_scissor", "selected_indices": selected, "queries": queries,
            "ranking_s": ranking_s, "cpu_threads": 1}
