"""Joint Gaussian-RSS/Bernoulli inference using observed and cross blocks only."""

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from testing_methods.common.classification import torch_moments
from testing_methods.common.gp_laplace import laplace

from .model import MARGIN_NOISE, JointResponseKernel

TARGET_PARAMETERS = (
    "mean",
    "margin_mean",
    "log_lengthscale",
    "log_scale",
    "log_margin_scale",
    "raw_correlation",
)


class JointPosterior:
    """Condition RSS analytically, then solve the Bernoulli mode once."""

    def __init__(self, mean, covariance, labels, margins):
        self.count = len(mean)
        self.r_factor = None
        self.state = None
        self.r_nll = mean.new_zeros(())
        if not self.count:
            self.info = {"iterations": 0, "gradient_max": 0.0, "backtracks": 0}
            return
        collision_mean = mean[:, 0]
        collision_covariance = covariance[::2, ::2]
        if margins is not None:
            values = torch.as_tensor(margins, dtype=mean.dtype, device=mean.device)
            if values.shape != (self.count,) or not torch.isfinite(values).all():
                raise ValueError("Observed RSS must align with paid collision labels")
            rr = covariance[1::2, 1::2]
            self.r_factor = torch.linalg.cholesky(
                rr + MARGIN_NOISE**2 * torch.eye(self.count, dtype=mean.dtype, device=mean.device)
            )
            difference = values - mean[:, 1]
            self.r_alpha = torch.cholesky_solve(difference[:, None], self.r_factor)[:, 0]
            self.r_to_f = torch.cholesky_solve(covariance[1::2, ::2], self.r_factor)
            collision_mean = collision_mean + covariance[::2, 1::2] @ self.r_alpha
            collision_covariance = collision_covariance - covariance[::2, 1::2] @ self.r_to_f
            self.r_nll = (
                0.5 * (difference * self.r_alpha).sum()
                + self.r_factor.diagonal().log().sum()
                + 0.5 * self.count * math.log(2 * math.pi)
            )
        self.state = laplace(collision_mean, collision_covariance, labels)
        self.info = {
            "iterations": self.state.iterations,
            "gradient_max": self.state.gradient_max,
            "backtracks": self.state.backtracks,
        }

    @property
    def nll(self):
        return self.r_nll + self.state.nll

    def predict(self, mean, cross, blocks):
        if not self.count:
            return mean, blocks
        flat_mean = mean.reshape(-1)
        cross_f = cross[:, ::2]
        result = blocks
        if self.r_factor is not None:
            cross_r = cross[:, 1::2]
            flat_mean = flat_mean + cross_r @ self.r_alpha
            solved_r = torch.linalg.solve_triangular(self.r_factor, cross_r.T, upper=False)
            solved_r = solved_r.T.reshape(len(mean), 2, self.count)
            result = result - solved_r @ solved_r.transpose(-1, -2)
            cross_f = cross_f - cross_r @ self.r_to_f
        flat_mean = flat_mean + cross_f @ self.state.alpha
        solved = torch.linalg.solve_triangular(
            self.state.factor, self.state.root[:, None] * cross_f.T, upper=False
        ).T.reshape(len(mean), 2, self.count)
        result = result - solved @ solved.transpose(-1, -2)
        return flat_mean.reshape(-1, 2), 0.5 * (result + result.transpose(-1, -2))


def joint_evidence(model, x, labels, margins, *, encoded=None):
    covariance = model.kernel(x) if encoded is None else model.kernel_from_features(encoded)
    state = JointPosterior(model.point_mean(len(x)), covariance, labels, margins)
    return state.nll / len(labels)


def fit_joint(model, x, labels, margins, *, steps=10, learning_rate=0.003, encoded=None):
    if not len(labels):
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    if encoded is None and not any(p.requires_grad for p in model.embedding.parameters()):
        encoded = model.features(x).detach()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = joint_evidence(model, x, labels, margins, encoded=encoded)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite joint response evidence")
        loss.backward()
        for group in model.adaptation_groups():
            torch.nn.utils.clip_grad_norm_(group, 5)
        optimizer.step()


def adapted_model(base, x, labels, margins, *, learning_rate=0.003, encoded=None):
    # Prefix states own six scalars and reuse the same frozen feature network.
    memo = {}
    if not any(p.requires_grad for p in base.embedding.parameters()):
        memo[id(base.embedding)] = base.embedding
    if hasattr(base, "construction_specification"):
        memo[id(base.construction_specification)] = base.construction_specification
    model = copy.deepcopy(base, memo).double()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in TARGET_PARAMETERS)
    with torch.enable_grad():
        fit_joint(model, x, labels, margins, learning_rate=learning_rate, encoded=encoded)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def historical_state(model, archive, *, encoded=None, use_rss=True):
    if archive.role != "history" and not archive.provenance.get("completed_task"):
        raise ValueError("Only historical or completed admitted tasks can supply a source")
    return JointPosterior(
        model.point_mean(len(archive.x)),
        model.kernel(archive.x) if encoded is None else model.kernel_from_features(encoded),
        archive.collision,
        archive.continuous_margin if use_rss else None,
    )


def historical_prior(model, archive, query_x, *, batch_size=128, encoded_query=None, use_rss=True):
    """Export pointwise joint blocks; never construct K(query_x, query_x)."""
    encoded_history = model.features(archive.x)
    encoded_query = model.features(query_x) if encoded_query is None else encoded_query
    state = historical_state(model, archive, encoded=encoded_history, use_rss=use_rss)
    means, blocks = [], []
    for start in range(0, len(query_x), batch_size):
        z = encoded_query[start : start + batch_size]
        mean, block = state.predict(
            model.point_mean(len(z)),
            model.kernel_from_features(z, encoded_history),
            model.point_covariance(len(z)),
        )
        means.append(mean)
        blocks.append(block)
    return torch.cat(means), torch.cat(blocks)


@dataclass
class JointPrior:
    model: JointResponseKernel
    x: np.ndarray
    reference: torch.Tensor
    source: dict | None = None

    def mean(self, indices):
        mean = self.model.point_mean(len(indices))
        if self.source is not None:
            mean = mean + self.source["means"][indices] - self.reference
        return mean

    def covariance(self, left, right, *, base_covariance=None):
        result = (
            self.model.kernel(self.x[left], self.x[right])
            if base_covariance is None
            else base_covariance
        )
        if self.source is not None:
            matches = torch.as_tensor(
                np.asarray(left)[:, None] == np.asarray(right)[None, :], device=result.device
            )
            added = torch.einsum(
                "ij,iab->iajb", matches.to(result.dtype), self.source["blocks"][left]
            )
            result = result + self.model.source_covariance(
                added.reshape(2 * len(left), 2 * len(right))
            )
            if "low_rank" in self.source:
                u = self.source["low_rank"]
                result = result + self.model.source_covariance(
                    u[left].flatten(0, 1) @ u[right].flatten(0, 1).T
                )
        return result

    def blocks(self, indices):
        result = self.model.point_covariance(len(indices))
        if self.source is not None:
            result = result + self.model.source_covariance(self.source["blocks"][indices])
            if "low_rank" in self.source:
                u = self.source["low_rank"][indices]
                result = result + self.model.source_covariance(u @ u.transpose(-1, -2))
        return result


def source_tensors(sources, model):
    if sources is None:
        return []
    count = len(sources["names"])
    if np.shape(sources["means"])[0] != count or np.shape(sources["blocks"]) != (
        *np.shape(sources["means"]),
        2,
    ):
        raise ValueError("Joint source means and 2x2 blocks must align")
    keys = ("means", "blocks", "low_rank") if "low_rank" in sources else ("means", "blocks")
    return [
        {
            k: torch.as_tensor(sources[k][i], device=model.mean.device, dtype=torch.float64)
            for k in keys
        }
        for i in range(count)
    ]


def predict_components(
    target,
    physical,
    x,
    sources,
    reference,
    paid,
    labels,
    margins,
    *,
    batch_size=128,
    indices=None,
    encoded_target=None,
    encoded_physical=None,
):
    """Factor each observed prior once; stream prediction in candidate batches."""
    if batch_size < 1 or len(paid) != len(labels) or len(set(paid)) != len(paid):
        raise ValueError("Positive batch size and unique aligned paid observations are required")
    if indices is None:
        indices = list(range(len(x)))
    priors = [JointPrior(target, x, reference, s) for s in sources]
    priors.append(JointPrior(target, x, reference))
    if physical is not None:
        priors.append(JointPrior(physical, x, reference))
    encoded_target = target.features(x) if encoded_target is None else encoded_target
    if physical is not None and physical is not target and encoded_physical is None:
        encoded_physical = physical.features(x)
    observed_target = target.kernel_from_features(encoded_target[paid])
    observed_physical = (
        physical.kernel_from_features(encoded_physical[paid])
        if physical is not None and physical is not target
        else observed_target
    )
    states = []
    for prior in priors:
        observed = observed_physical if prior.model is physical else observed_target
        states.append(
            JointPosterior(
                prior.mean(paid),
                prior.covariance(paid, paid, base_covariance=observed),
                labels,
                margins,
            )
        )
    outputs = [[] for _ in priors]
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        target_cross = target.kernel_from_features(encoded_target[batch], encoded_target[paid])
        physical_cross = (
            physical.kernel_from_features(encoded_physical[batch], encoded_physical[paid])
            if physical is not None and physical is not target
            else target_cross
        )
        for values, prior, state in zip(outputs, priors, states):
            cross = physical_cross if prior.model is physical else target_cross
            mean, blocks = state.predict(
                prior.mean(batch),
                prior.covariance(batch, paid, base_covariance=cross),
                prior.blocks(batch),
            )
            values.append(torch_moments(mean[:, 0], blocks[:, 0, 0].clamp_min(1e-10))[0])
    return torch.stack([torch.cat(values) for values in outputs]), [state.info for state in states]


def initial_weights(sources, reference):
    if sources is None:
        return reference.new_ones(1)
    weights = torch.as_tensor(sources["weights"], dtype=reference.dtype, device=reference.device)
    return torch.cat((0.75 * weights, weights.new_tensor([0.25])))


def mixture_weights(scores, *, raw_only=False):
    if raw_only:
        return scores.new_ones(1)
    return torch.cat((0.9 * scores.softmax(0), scores.new_tensor([0.1])))


def update_weights(scores, probabilities, collision):
    likelihood = (probabilities if collision else 1 - probabilities).clamp_min(1e-15).log()
    return torch.log_softmax(scores + likelihood, dim=0), likelihood


def sequential_probability(
    sources,
    states,
    reference,
    x,
    labels,
    margins,
    count,
    *,
    encoded_target=None,
    encoded_physical=None,
):
    """Predict record count after replaying only the preceding paid responses."""
    if not 0 <= count < len(x) or len(states) <= count or len(labels) < count:
        raise ValueError("Sequential prediction needs a state for every preceding prefix")
    scores = initial_weights(sources, reference).log()
    tensors = source_tensors(sources, states[0][0])
    if encoded_target is None:
        encoded_target = states[0][0].features(x)
    if encoded_physical is None:
        encoded_physical = states[0][1].features(x)
    for position in range(count + 1):
        target, physical = states[position]
        components, _ = predict_components(
            target,
            physical,
            x,
            tensors,
            reference,
            list(range(position)),
            labels[:position],
            None if margins is None else margins[:position],
            indices=[position],
            encoded_target=encoded_target,
            encoded_physical=encoded_physical,
        )
        effective = mixture_weights(scores)
        if position == count:
            return (effective * components[:, 0]).sum()
        scores, _ = update_weights(scores, components[:-1, 0], labels[position])
