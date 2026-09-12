"""Fold-local formal-event calibration for the analytic Stage 1-v2 teacher."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from .state import TeacherState, TeacherWorld


FORMAL_CLASS_SCORES = np.asarray((0.0, 0.5, 1.0), dtype=np.float64)


def formal_class_indexes(scores: np.ndarray) -> np.ndarray:
    """Map the frozen formal scale ``{0, .5, 1}`` to class indexes."""
    values = np.asarray(scores, dtype=np.float64)
    result = np.full(values.shape, -1, dtype=np.int64)
    for index, value in enumerate(FORMAL_CLASS_SCORES):
        result[np.isclose(values, value)] = index
    if np.any(result < 0):
        raise ValueError("formal scores must use the frozen three-class scale")
    return result


def _expanded_features(state: TeacherState) -> np.ndarray:
    """Build transferable scenario/belief features without SUT identity."""
    mean, variance = state.predicted_vulnerability()
    std = np.sqrt(np.maximum(variance, 1e-12))
    scenario = state.world.features
    return np.column_stack(
        (
            scenario,
            mean,
            std,
            np.square(mean),
            np.power(mean, 3),
            scenario * mean[:, None],
        )
    )


@dataclass(frozen=True)
class FormalEventCalibrator:
    """Deterministic class-weighted multinomial calibration model."""

    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    training_source_indexes: tuple[int, ...]
    class_counts: tuple[int, int, int]
    l2: float

    @classmethod
    def fit(
        cls,
        world: TeacherWorld,
        training_source_indexes: Iterable[int],
        *,
        context_sizes: tuple[int, ...] = (0, 1, 2, 4),
        l2: float = 1e-3,
        max_iterations: int = 500,
    ) -> "FormalEventCalibrator":
        indexes = tuple(int(index) for index in training_source_indexes)
        if len(indexes) < 2 or len(set(indexes)) != len(indexes):
            raise ValueError("formal calibration requires distinct training sources")
        if any(index < 0 or index >= world.formal_scores.shape[0] for index in indexes):
            raise IndexError("formal calibration source is outside the source bank")
        if not context_sizes or context_sizes[0] != 0 or tuple(sorted(set(context_sizes))) != context_sizes:
            raise ValueError("formal calibration context sizes must be sorted and start at zero")
        if context_sizes[-1] >= len(world.pool):
            raise ValueError("formal calibration context exhausts the anchor pool")
        if l2 < 0.0 or max_iterations < 1:
            raise ValueError("invalid formal calibration optimizer settings")

        feature_rows: list[np.ndarray] = []
        label_rows: list[np.ndarray] = []
        for source_index in indexes:
            state = TeacherState(world)
            for context_size in context_sizes:
                while len(state.history_indexes) < context_size:
                    selected = state.select_index("diagnostic")
                    state.observe(
                        selected,
                        float(world.responses[source_index, selected]),
                        float(world.formal_scores[source_index, selected]),
                    )
                available = np.flatnonzero(state.available_mask())
                feature_rows.append(_expanded_features(state)[available])
                label_rows.append(formal_class_indexes(world.formal_scores[source_index, available]))

        features = np.concatenate(feature_rows, axis=0)
        labels = np.concatenate(label_rows, axis=0)
        feature_mean = features.mean(axis=0)
        feature_scale = features.std(axis=0)
        feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
        standardized = (features - feature_mean) / feature_scale
        design = np.column_stack((np.ones(len(standardized)), standardized))
        counts = np.bincount(labels, minlength=3)
        if np.any(counts == 0):
            raise ValueError("training fold must contain all three formal event classes")
        class_weights = np.clip(len(labels) / (3.0 * counts), 0.25, 10.0)
        example_weights = class_weights[labels]
        normalizer = float(example_weights.sum())

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            weights = flat.reshape(design.shape[1], 3)
            logits = design @ weights
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            chosen = np.maximum(probabilities[np.arange(len(labels)), labels], 1e-15)
            loss = -float(np.dot(example_weights, np.log(chosen))) / normalizer
            loss += 0.5 * l2 * float(np.square(weights[1:]).sum())
            residual = probabilities.copy()
            residual[np.arange(len(labels)), labels] -= 1.0
            residual *= example_weights[:, None] / normalizer
            gradient = design.T @ residual
            gradient[1:] += l2 * weights[1:]
            return loss, gradient.ravel()

        initial = np.zeros((design.shape[1], 3), dtype=np.float64)
        result = minimize(
            objective,
            initial.ravel(),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(max_iterations), "ftol": 1e-12, "gtol": 1e-8},
        )
        # Hitting the deterministic iteration cap is an acceptable converged
        # approximation; non-finite objectives or line-search failures are not.
        if not np.isfinite(result.fun) or int(result.status) not in (0, 1):
            raise RuntimeError(f"formal event calibration failed: {result.message}")
        return cls(
            weights=np.asarray(result.x, dtype=np.float64).reshape(design.shape[1], 3),
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            training_source_indexes=indexes,
            class_counts=tuple(int(value) for value in counts),
            l2=float(l2),
        )

    def predict_proba(self, state: TeacherState) -> np.ndarray:
        features = (_expanded_features(state) - self.feature_mean) / self.feature_scale
        design = np.column_stack((np.ones(len(features)), features))
        logits = design @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("formal event calibrator produced non-finite probabilities")
        return probabilities

    def expected_utility(self, state: TeacherState) -> np.ndarray:
        return self.predict_proba(state) @ FORMAL_CLASS_SCORES
