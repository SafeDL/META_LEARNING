"""Directed risk regression and contextual event-residual inference."""
import math

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import log_ndtr, ndtr, ndtri, stdtr, stdtrit
from sklearn.isotonic import IsotonicRegression

from methods.history_guided_testing.config import ROOT as HISTORY_ROOT
from methods.history_guided_testing.gp import RiskGP
from methods.history_guided_testing.history import split_indices
from methods.history_guided_testing.io import write_json

from .config import PROFILES


def integrated_event_probability(mean, deviation, residual_mean, residual_variance, readout, degrees_of_freedom=None):
    """Integrate calibration segments using the risk predictive CDF and quantile."""
    x, p = readout.X_thresholds_, readout.y_thresholds_
    if degrees_of_freedom is None:
        cdf_function, quantile_function = ndtr, ndtri
    else:
        cdf_function = lambda value: stdtr(degrees_of_freedom, value)
        quantile_function = lambda value: stdtrit(degrees_of_freedom, value)
    cdf = cdf_function((x[None, :] - mean[:, None]) / deviation[:, None])
    event_sd = np.sqrt(1 + residual_variance)

    def event_chance(probability):
        return ndtr((math.sqrt(2) * ndtri(np.clip(probability, .01, .99)) + residual_mean) / event_sd)

    result = event_chance(p[0]) * cdf[:, 0]
    result += event_chance(p[-1]) * cdf_function((mean - x[-1]) / deviation)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    for i in range(len(x) - 1):
        mass = np.maximum(cdf[:, i + 1] - cdf[:, i], 0)
        if p[i + 1] == p[i]:
            result += mass * event_chance(p[i])
            continue
        probability_nodes = cdf[:, i, None] + .5 * mass[:, None] * (1 + nodes)
        risk_nodes = mean[:, None] + deviation[:, None] * quantile_function(np.clip(probability_nodes, 1e-14, 1 - 1e-14))
        mapped = p[i] + (p[i + 1] - p[i]) * (risk_nodes - x[i]) / (x[i + 1] - x[i])
        mapped = np.clip(mapped, p[i], p[i + 1])
        event = ndtr((math.sqrt(2) * ndtri(np.clip(mapped, .01, .99)) + residual_mean[:, None]) / event_sd[:, None])
        result += .5 * mass * (event @ weights)
    return np.clip(result, .01, .99)


def fit_event_readout(folder):
    models, report = {}, {}
    with np.load(HISTORY_ROOT / "history/responses.npz") as bank:
        train, _ = split_indices(bank["x"])
        for group in (0, 1):
            indices = train[bank["x"][train, 4] == group]
            risk = np.concatenate([bank[p.name][indices] for p in PROFILES])
            event = np.concatenate([bank[p.name + "_collision"][indices] for p in PROFILES])
            model = IsotonicRegression(y_min=.01, y_max=.99, increasing=True, out_of_bounds="clip")
            model.fit(risk, event.astype(float))
            models[group] = model
            report[str(group)] = {"risk_knots": model.X_thresholds_.tolist(),
                                  "event_probabilities": model.y_thresholds_.tolist(),
                                  "historical_samples": len(risk)}
    write_json(folder / "event_readout.json", {
        "families": report, "source_names": [p.name for p in PROFILES],
        "scope": "original six measured sources, frozen spatial training indices; no target outcomes",
        "regularization": "monotone historical mean only; target contextual residual can reverse it"})
    return models


class RiskEventResponse:
    def __init__(self, reference, family, readout):
        self.reference = reference
        original = reference.session.gp
        self.gp = RiskGP(original.covariance, original.mean, noise=original.noise)
        self.family = np.asarray(family, dtype=int)
        self.readout = readout
        deviation = np.sqrt(np.diag(original.covariance))
        local = original.covariance / (deviation[:, None] * deviation[None, :])
        shared = self.family[:, None] == self.family[None, :]
        self.kernel = .5 * (local + shared)
        self.indices, self.risks, self.labels = [], [], []
        self.alpha = np.empty(0)
        self.records = []
        self.selection_checks = []
        self.dirty = True
        self.factor = None
        self.root_precision = None

    def mean_from_risk(self, risk, family):
        probability = np.empty_like(risk, dtype=float)
        for group in (0, 1):
            mask = family == group
            if not mask.any():
                continue
            probability[mask] = self.readout[group].predict(np.clip(risk[mask], 0, 1).reshape(-1)).reshape(risk[mask].shape)
        return math.sqrt(2) * ndtri(np.clip(probability, .01, .99))

    def fit_residual(self):
        if not self.indices or not self.dirty:
            return
        indices = np.asarray(self.indices)
        kernel = self.kernel[np.ix_(indices, indices)]
        mean = self.mean_from_risk(np.asarray(self.risks), self.family[indices])
        sign = 2 * np.asarray(self.labels, dtype=float) - 1
        alpha = np.pad(self.alpha, (0, len(indices) - len(self.alpha)))

        def objective(a):
            residual = kernel @ a
            return float(log_ndtr(sign * (mean + residual)).sum() - .5 * a @ residual)

        for _ in range(100):
            residual = kernel @ alpha
            value = mean + residual
            z = sign * value
            ratio = np.exp(-.5 * z ** 2 - .5 * math.log(2 * math.pi) - log_ndtr(z))
            gradient = sign * ratio
            precision = np.clip(ratio * (ratio + z), 0, 1)
            root = np.sqrt(precision)
            factor = np.linalg.cholesky(np.eye(len(indices)) + root[:, None] * kernel * root[None, :])
            b = precision * residual + gradient
            proposed = b - root * cho_solve((factor, True), root * (kernel @ b))
            old_objective = objective(alpha)
            step = 1.
            for _ in range(25):
                candidate = alpha + step * (proposed - alpha)
                if objective(candidate) >= old_objective - 1e-10:
                    break
                step *= .5
            else:
                raise RuntimeError("Event-residual Laplace update did not improve its objective")
            change = np.max(np.abs(kernel @ (candidate - alpha)))
            alpha = candidate
            if change < 1e-7:
                break
        else:
            raise RuntimeError("Event-residual Laplace mode did not converge")
        z = sign * (mean + kernel @ alpha)
        ratio = np.exp(-.5 * z ** 2 - .5 * math.log(2 * math.pi) - log_ndtr(z))
        self.root_precision = np.sqrt(np.clip(ratio * (ratio + z), 0, 1))
        self.factor = np.linalg.cholesky(np.eye(len(indices)) + self.root_precision[:, None]
                                        * kernel * self.root_precision[None, :])
        self.alpha = alpha
        self.dirty = False

    def probabilities(self):
        self.fit_residual()
        residual_mean = np.zeros(len(self.family))
        residual_variance = np.ones(len(self.family))
        if self.indices:
            cross = self.kernel[:, self.indices]
            residual_mean = cross @ self.alpha
            solved = solve_triangular(self.factor, self.root_precision[:, None] * cross.T, lower=True)
            residual_variance -= (solved ** 2).sum(axis=0)
            if residual_variance.min() < -1e-8:
                raise FloatingPointError("Invalid event-residual predictive variance")
            residual_variance = np.maximum(residual_variance, 0)
        probability = np.empty(len(self.family))
        deviation = np.sqrt(self.gp.variance + self.gp.noise)
        degrees_of_freedom = getattr(self.gp, "predictive_df", None)
        if degrees_of_freedom is not None:
            deviation *= np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)
        for group in (0, 1):
            mask = self.family == group
            probability[mask] = integrated_event_probability(
                self.gp.mean[mask], deviation[mask], residual_mean[mask], residual_variance[mask], self.readout[group], degrees_of_freedom)
        return probability

    def rank(self, remaining, *, tie_break):
        head = self.reference.head()
        indices = np.flatnonzero(remaining & (self.family == self.family[head]))
        probability = self.probabilities()
        order = np.lexsort((tie_break[indices], -self.gp.mean[indices], -np.round(probability[indices], 12)))
        proposed = int(indices[order[0]])
        chosen = proposed if probability[proposed] > probability[head] + 1e-12 else int(head)
        self.selection_checks.append({"query": len(self.indices) + 1, "chosen_index": chosen,
                                      "reference_index": int(head), "family": int(self.family[head]),
                                      "chosen_probability": float(probability[chosen]),
                                      "reference_probability": float(probability[head]),
                                      "scope": "reference functional-regime anchor; strict model probability gain within regime"})
        return np.asarray([chosen])

    def observe(self, index, risk, collision):
        self.gp.observe(index, risk)
        self.indices.append(int(index))
        self.risks.append(float(risk))
        self.labels.append(bool(collision))
        self.dirty = True
        self.records.append({"index": int(index), "risk": float(risk), "collision": bool(collision),
                             "risk_model_feedback": "risk only", "event_model_feedback": "exact queried risk and event"})

    def diagnostics(self):
        return {"event_residual_records": self.records, "event_residual_observations": len(self.indices),
                "regime_selection_checks": self.selection_checks,
                "risk_scale_records": getattr(self.gp, "scale_records", []),
                "event_inference": "Laplace approximation with probit likelihood; exact constant calibration segments and sixteen-node probability-space quadrature on linear segments",
                "information_flow": "event feedback cannot update the risk GP"}
