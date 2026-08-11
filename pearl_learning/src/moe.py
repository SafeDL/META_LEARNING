"""Posterior-routed residual mixture-of-experts actor components."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


DESCRIPTOR_SCHEMA = "merge_physical_task_descriptor_v1"
DESCRIPTOR_FIELDS = (
    "adversary_lane_count",
    "sut_lane_count",
    "merge_length_m",
    "conflict_radius_m",
    "adversary_route_curvature",
    "sut_route_curvature",
)
ROUTER_POSTERIOR_FIELDS = ("posterior_mean", "posterior_log_variance")
ROUTER_INPUT_FIELDS = {
    "static": DESCRIPTOR_FIELDS,
    "posterior_mean": ("posterior_mean",),
    "static_posterior_mean": (*DESCRIPTOR_FIELDS, "posterior_mean"),
    "static_posterior_mean_logvar": (*DESCRIPTOR_FIELDS, *ROUTER_POSTERIOR_FIELDS),
}
FORBIDDEN_DESCRIPTOR_TOKENS = (
    "task_id",
    "geometry_id",
    "logical_type",
    "split",
    "priority",
    "rule",
    "route_remaining",
)


def _content_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")


@dataclass(frozen=True)
class PhysicalTaskDescriptor:
    schema: str
    fields: tuple[str, ...]
    raw_values: tuple[float, ...]
    normalized_values: tuple[float, ...]
    normalization_scales: tuple[float, ...]
    content_hash: str

    def tensor(self, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(self.normalized_values, dtype=torch.float32, device=device)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fields": list(self.fields),
            "raw_values": dict(zip(self.fields, self.raw_values)),
            "normalized_values": dict(zip(self.fields, self.normalized_values)),
            "normalization_scales": dict(zip(self.fields, self.normalization_scales)),
            "content_hash": self.content_hash,
        }


def physical_task_descriptor(
    topology: Mapping[str, Any],
    *,
    schema: str,
    normalization: Mapping[str, Any],
) -> PhysicalTaskDescriptor:
    """Freeze the allowlisted physical fields from an initialized map."""
    if schema != DESCRIPTOR_SCHEMA:
        raise ValueError(f"unsupported physical task descriptor schema: {schema!r}")
    if tuple(normalization) != DESCRIPTOR_FIELDS:
        missing = sorted(set(DESCRIPTOR_FIELDS) - set(normalization))
        extra = sorted(set(normalization) - set(DESCRIPTOR_FIELDS))
        raise ValueError(f"descriptor normalization fields mismatch; missing={missing}, extra={extra}")
    if any(any(token in field for token in FORBIDDEN_DESCRIPTOR_TOKENS) for field in DESCRIPTOR_FIELDS):
        raise RuntimeError("descriptor allowlist contains a forbidden leakage field")
    try:
        raw = tuple(float(topology[field]) for field in DESCRIPTOR_FIELDS)
        scales = tuple(float(normalization[field]) for field in DESCRIPTOR_FIELDS)
    except KeyError as error:
        raise ValueError(f"topology lacks descriptor field {error.args[0]!r}") from error
    if not np.isfinite(raw).all():
        raise ValueError("physical task descriptor contains a non-finite value")
    if not np.isfinite(scales).all() or any(scale <= 0.0 for scale in scales):
        raise ValueError("descriptor normalization scales must be finite and positive")
    normalized = tuple(value / scale for value, scale in zip(raw, scales))
    payload = {
        "schema": schema,
        "fields": list(DESCRIPTOR_FIELDS),
        "raw_values": list(raw),
        "normalized_values": list(normalized),
        "normalization_scales": list(scales),
    }
    return PhysicalTaskDescriptor(
        schema,
        DESCRIPTOR_FIELDS,
        raw,
        normalized,
        scales,
        _content_hash(payload),
    )


@dataclass(frozen=True)
class RoutingOutput:
    logits: torch.Tensor
    soft_weights: torch.Tensor
    top_k_mask: torch.Tensor
    weights: torch.Tensor
    entropy: torch.Tensor
    top_k_indexes: torch.Tensor


@dataclass(frozen=True)
class RouteContext:
    descriptor: PhysicalTaskDescriptor
    posterior_version: int
    posterior_mean: tuple[float, ...]
    posterior_log_variance: tuple[float, ...]
    logits: tuple[float, ...]
    weights: tuple[float, ...]
    top_k_indexes: tuple[int, ...]
    entropy: float
    route_hash: str
    gradient_enabled: bool
    query_free: bool = True
    intervention: str = "none"
    source_route_hash: str | None = None

    def weight_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(self.weights, dtype=torch.float32, device=device).unsqueeze(0)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "posterior_version": self.posterior_version,
            "posterior_mean": list(self.posterior_mean),
            "posterior_log_variance": list(self.posterior_log_variance),
            "descriptor": self.descriptor.audit_dict(),
            "router_logits": list(self.logits),
            "routing_weights": list(self.weights),
            "top_k_indexes": list(self.top_k_indexes),
            "entropy": self.entropy,
            "route_hash": self.route_hash,
            "gradient_enabled": self.gradient_enabled,
            "query_free": self.query_free,
            "intervention": self.intervention,
            "source_route_hash": self.source_route_hash,
        }


class PosteriorRouter(nn.Module):
    def __init__(
        self,
        descriptor_dim: int,
        latent_dim: int,
        num_experts: int,
        top_k: int,
        routing: str,
        hidden_sizes: Sequence[int],
        input_mode: str = "static_posterior_mean_logvar",
    ):
        super().__init__()
        if descriptor_dim <= 0 or latent_dim <= 0 or num_experts < 2:
            raise ValueError("router dimensions must be positive and num_experts must be at least two")
        if routing not in {"soft", "top_k"}:
            raise ValueError("routing must be 'soft' or 'top_k'")
        if not 1 <= int(top_k) <= int(num_experts):
            raise ValueError("top_k must lie in [1, num_experts]")
        if routing == "soft" and int(top_k) != int(num_experts):
            raise ValueError("soft routing requires top_k == num_experts")
        widths = [int(width) for width in hidden_sizes]
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("router_hidden_sizes must contain positive widths")
        if input_mode not in ROUTER_INPUT_FIELDS:
            raise ValueError(f"unsupported router input_mode: {input_mode!r}")
        layers: list[nn.Module] = []
        last = (
            (descriptor_dim if "static" in input_mode else 0)
            + (latent_dim if "posterior_mean" in input_mode else 0)
            + (latent_dim if input_mode.endswith("logvar") else 0)
        )
        for width in widths:
            layers.extend((nn.Linear(last, width), nn.ReLU()))
            last = width
        final = nn.Linear(last, num_experts)
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.model = nn.Sequential(*layers)
        self.descriptor_dim = int(descriptor_dim)
        self.latent_dim = int(latent_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.routing = routing
        self.input_mode = input_mode
        self.input_fields = ROUTER_INPUT_FIELDS[input_mode]

    def forward(
        self,
        descriptor: torch.Tensor,
        posterior_mean: torch.Tensor,
        posterior_log_variance: torch.Tensor,
    ) -> RoutingOutput:
        for value, name, width in (
            (descriptor, "descriptor", self.descriptor_dim),
            (posterior_mean, "posterior_mean", self.latent_dim),
            (posterior_log_variance, "posterior_log_variance", self.latent_dim),
        ):
            if value.ndim != 2 or value.shape[-1] != width:
                raise ValueError(f"{name} must have shape [batch, {width}]")
            _finite_tensor(value, name)
        if not (len(descriptor) == len(posterior_mean) == len(posterior_log_variance)):
            raise ValueError("router inputs must share a batch dimension")
        inputs = []
        if "static" in self.input_mode:
            inputs.append(descriptor)
        if "posterior_mean" in self.input_mode:
            inputs.append(posterior_mean.detach())
        if self.input_mode.endswith("logvar"):
            inputs.append(posterior_log_variance.detach())
        features = torch.cat(inputs, dim=-1)
        logits = self.model(features)
        _finite_tensor(logits, "router logits")
        soft_weights = torch.softmax(logits, dim=-1)
        _, top_indexes = torch.topk(soft_weights, self.top_k, dim=-1)
        mask = torch.zeros_like(soft_weights, dtype=torch.bool).scatter(-1, top_indexes, True)
        weights = soft_weights if self.routing == "soft" else torch.where(mask, soft_weights, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        _finite_tensor(weights, "routing weights")
        if (weights < 0).any() or not torch.allclose(
            weights.sum(dim=-1), torch.ones(len(weights), device=weights.device), atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("routing weights must be non-negative and sum to one")
        if self.routing == "top_k" and not torch.equal(weights == 0.0, ~mask):
            raise RuntimeError("non-top-k experts must have exactly zero weight")
        entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
        return RoutingOutput(logits, soft_weights, mask, weights, entropy, top_indexes)


def load_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """Squared coefficient of variation of mean expert load."""
    _finite_tensor(weights, "routing weights")
    if weights.ndim != 2 or weights.shape[-1] < 2:
        raise ValueError("routing weights must have shape [batch, experts>=2]")
    load = weights.mean(dim=0)
    return load.var(unbiased=False) / load.mean().square().clamp_min(1e-12)


class PosteriorRoutedMoEActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        num_experts: int,
        expert_hidden_size: int,
    ):
        super().__init__()
        widths = [int(width) for width in hidden_sizes]
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("actor_hidden_sizes must contain positive widths")
        if int(expert_hidden_size) <= 0:
            raise ValueError("expert_hidden_size must be positive")
        input_dim = int(observation_dim) + int(latent_dim)
        feature_dim = widths[-1]
        shared_layers: list[nn.Module] = []
        last = input_dim
        for width in widths:
            shared_layers.extend((nn.Linear(last, width), nn.ReLU()))
            last = width
        self.shared_trunk = nn.Sequential(*shared_layers)
        self.residual_experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, int(expert_hidden_size)),
                nn.ReLU(),
                nn.Linear(int(expert_hidden_size), feature_dim),
            )
            for _ in range(int(num_experts))
        )
        self.gaussian_head = nn.Linear(feature_dim, int(action_dim) * 2)
        self.action_dim = int(action_dim)
        self.num_experts = int(num_experts)

    def forward(
        self,
        observation: torch.Tensor,
        latent: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or latent.ndim != 2:
            raise ValueError("actor observation and latent must be matrices")
        if route_weights.shape != (len(observation), self.num_experts):
            raise ValueError(
                f"route_weights must have shape [{len(observation)}, {self.num_experts}]"
            )
        _finite_tensor(route_weights, "route_weights")
        if (route_weights < 0).any() or not torch.allclose(
            route_weights.sum(dim=-1),
            torch.ones(len(route_weights), device=route_weights.device),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("route_weights must be non-negative and sum to one")
        actor_input = torch.cat((observation, latent), dim=-1)
        residuals = torch.stack([expert(actor_input) for expert in self.residual_experts], dim=1)
        features = self.shared_trunk(actor_input) + (route_weights.unsqueeze(-1) * residuals).sum(dim=1)
        mean, log_std = self.gaussian_head(features).chunk(2, dim=-1)
        return mean, log_std.clamp(-20, 2)

    def sample(
        self,
        observation: torch.Tensor,
        latent: torch.Tensor,
        route_weights: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observation, latent, route_weights)
        normal = Normal(mean, log_std.exp())
        raw_action = mean if deterministic else normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action.square() + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def expert_action_means(
        self,
        observation: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Return each anonymous expert's deterministic action at shared inputs."""
        actor_input = torch.cat((observation, latent), dim=-1)
        shared = self.shared_trunk(actor_input)
        actions = []
        for expert in self.residual_experts:
            mean, _ = self.gaussian_head(shared + expert(actor_input)).chunk(2, dim=-1)
            actions.append(torch.tanh(mean))
        return torch.stack(actions, dim=1)


def route_context(
    descriptor: PhysicalTaskDescriptor,
    posterior_version: int,
    posterior_mean: torch.Tensor,
    posterior_log_variance: torch.Tensor,
    output: RoutingOutput,
    *,
    gradient_enabled: bool,
) -> RouteContext:
    if posterior_version < 0:
        raise ValueError("posterior_version must be non-negative")
    if len(output.weights) != 1:
        raise ValueError("a collection route context must contain exactly one task")
    mean = tuple(float(value) for value in posterior_mean.detach().cpu().reshape(-1))
    log_variance = tuple(float(value) for value in posterior_log_variance.detach().cpu().reshape(-1))
    logits = tuple(float(value) for value in output.logits.detach().cpu().reshape(-1))
    weights = tuple(float(value) for value in output.weights.detach().cpu().reshape(-1))
    indexes = tuple(int(value) for value in output.top_k_indexes.detach().cpu().reshape(-1))
    entropy = float(output.entropy.detach().cpu().item())
    digest = _content_hash({
        "descriptor_hash": descriptor.content_hash,
        "posterior_version": int(posterior_version),
        "posterior_mean": mean,
        "posterior_log_variance": log_variance,
        "logits": logits,
        "weights": weights,
        "top_k_indexes": indexes,
    })
    return RouteContext(
        descriptor,
        int(posterior_version),
        mean,
        log_variance,
        logits,
        weights,
        indexes,
        entropy,
        digest,
        bool(gradient_enabled),
    )


def intervene_route(
    source: RouteContext,
    weights: Sequence[float],
    *,
    posterior_version: int,
    intervention: str,
) -> RouteContext:
    """Create an auditable route intervention without touching model tensors."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(source.weights):
        raise ValueError("intervention weights must match the source expert count")
    if not np.isfinite(values).all() or np.any(values < 0.0) or float(values.sum()) <= 0.0:
        raise ValueError("intervention weights must be finite, non-negative, and non-empty")
    values = values / values.sum()
    entropy = float(-np.sum(values * np.log(np.maximum(values, 1e-12))))
    indexes = tuple(int(index) for index in np.argsort(-values))
    digest = _content_hash({
        "source_route_hash": source.route_hash,
        "posterior_version": int(posterior_version),
        "intervention": intervention,
        "weights": values.tolist(),
    })
    return RouteContext(
        source.descriptor,
        int(posterior_version),
        source.posterior_mean,
        source.posterior_log_variance,
        source.logits,
        tuple(float(value) for value in values),
        indexes,
        entropy,
        digest,
        False,
        source.query_free,
        intervention,
        source.route_hash,
    )
