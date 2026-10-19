"""Paid-feedback local correction and whole-controller stopping calibration."""

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit

from .config import ALPHA, SEGMENT
from .model import corrected_log_pmf, log_pmf, snapshot, tilted_log_pmf


def fit_tilt(logits, labels, penalty):
    logits, labels = np.asarray(logits), np.asarray(labels)

    def gradient(u):
        return float((expit(logits + u) - labels).sum() + penalty * u)

    # The positive quadratic penalty bounds the unique root by n / lambda.
    limit = (len(labels) + 1) / penalty
    return float(brentq(gradient, -limit, limit, xtol=1e-10))


def threshold(scores, mode="empirical", alpha=ALPHA):
    scores = sorted(map(float, scores))
    if not scores:
        return 1.0
    if mode == "empirical":
        return scores[-1]
    if mode != "strict":
        raise ValueError(f"Unknown calibration mode: {mode}")
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return scores[rank - 1] if rank <= len(scores) else 1.0


def block_scores(forecasts):
    scores = {}
    for task in forecasts:
        full = task["full_found"]
        score = max(
            (r["score"] for r in task["forecast"] if 20 * r["found"] < 19 * full),
            default=0.0,
        )
        key = task.get("sut") or task["source"]
        scores[key] = max(scores.get(key, 0.0), score)
    return scores


class StopController:
    def __init__(
        self,
        network,
        gamma,
        penalty=1.0,
        *,
        correction=True,
        direct_tilt=False,
        contract=None,
    ):
        self.network = network
        self.gamma = gamma
        self.penalty = penalty
        self.correction = correction
        self.direct_tilt = direct_tilt
        self.contract = contract
        self.u = 0.0
        self.logits, self.labels = [], []
        self.pending = None
        self.state = None
        self.segment_prediction = None

    def validate(self, prior, budget, initial_indices, policy):
        if self.contract is None:
            return
        actual = dict(
            pool_size=len(prior.x),
            budget=budget,
            initial=len(initial_indices),
            policy=policy,
            segment=SEGMENT,
        )
        if "family" in self.contract:
            actual["family"] = {0: "cutin", 1: "brake"}[int(prior.x[0, 4])]
        for key, value in actual.items():
            if self.contract[key] != value:
                raise ValueError(f"Stopping calibration mismatch: {key}")

    @torch.no_grad()
    def before_state(self, state):
        if state["step"] % SEGMENT:
            return
        length = min(SEGMENT, state["budget"] - state["step"])
        if (
            self.segment_prediction is not None
            and self.segment_prediction[0] == state["step"]
        ):
            parameters = self.segment_prediction[1]
        else:
            parameters = self.network.predict(state, [length])[0]
        self.segment_prediction = None
        base = log_pmf(parameters, length)
        k = torch.arange(len(base), device=base.device, dtype=base.dtype)
        rate = float((base.exp() * k).sum().cpu()) / length
        self.pending = dict(
            logit=float(
                np.log(np.clip(rate, 1e-12, 1 - 1e-12) / np.clip(1 - rate, 1e-12, 1))
            ),
            length=length,
            found=0,
            observed=0,
        )

    @torch.no_grad()
    def before_query(self, index, **context):
        if len(context["queries"]) % SEGMENT:
            return
        state = (
            self.state if self.state is not None else snapshot(index=index, **context)
        )
        state["next_index"] = int(index)
        self.state = None
        self.before_state(state)

    def observe(self, risk, count):
        if self.pending is None:
            raise RuntimeError("A paid segment needs its saved prequery prediction")
        self.pending["found"] += int(risk > 0.5)
        self.pending["observed"] += 1
        if self.pending["observed"] == self.pending["length"]:
            self.logits.append(self.pending["logit"])
            self.labels.append(self.pending["found"] / self.pending["length"])
            self.pending = None
            if self.correction:
                self.u = fit_tilt(self.logits, self.labels, self.penalty)

    @torch.no_grad()
    def evaluate_state(self, state):
        remaining = state["budget"] - state["step"]
        length = min(SEGMENT, remaining)
        lengths = sorted({length, remaining})
        parameters = self.network.predict(state, lengths)
        self.segment_prediction = (state["step"], parameters[0])
        base = log_pmf(parameters[-1], remaining)
        transform = tilted_log_pmf if self.direct_tilt else corrected_log_pmf
        corrected = transform(base, self.u)
        p = corrected.exp()
        limit = state["found"] // 19
        score = float(p[: limit + 1].sum().clamp(0, 1).cpu())
        mean = float((p * torch.arange(len(p), device=p.device)).sum().cpu())
        deterministic = remaining <= limit
        stop = bool(deterministic or score > self.gamma)
        return {
            "query": state["step"],
            "found": state["found"],
            "score": score,
            "remaining_mean": mean,
            "base_remaining_mean": float(
                (base.exp() * torch.arange(len(base), device=base.device)).sum().cpu()
            ),
            "tilt": self.u,
            "threshold": self.gamma,
            "allowed_misses": limit,
            "deterministic": deterministic,
            "stop": stop,
            "remaining_log_probability": corrected.cpu().tolist(),
        }

    def evaluate(self, model, remaining, queries, evidence, budget, initial_indices):
        # Full-horizon summaries do not use the unknown next active selection.
        available = np.flatnonzero(remaining)
        index = int(available[0])
        if budget - len(queries) == 1:
            index = int(available[np.argmax(model.probabilities()[available])])
        self.state = snapshot(
            model, remaining, queries, evidence, budget, initial_indices, index
        )
        return self.evaluate_state(self.state)
