"""Episode-level hybrid PPO policy for scene configuration and intent."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Categorical, Normal


@dataclass
class SceneAction:
    candidate_index: torch.Tensor
    continuous: torch.Tensor
    option_index: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


class HybridScenePolicy(nn.Module):
    def __init__(self, input_dim: int, candidate_count: int, continuous_dim: int, option_count: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.candidates, self.options = nn.Linear(256, candidate_count), nn.Linear(256, option_count)
        self.mean, self.log_std = nn.Linear(256, continuous_dim), nn.Parameter(torch.zeros(continuous_dim))
        self.value_head = nn.Linear(256, 1)

    def _distributions(self, inputs: torch.Tensor) -> tuple[Categorical, Normal, Categorical, torch.Tensor]:
        body = self.body(inputs)
        return Categorical(logits=self.candidates(body)), Normal(self.mean(body), self.log_std.clamp(-5.0, 2.0).exp()), Categorical(logits=self.options(body)), self.value_head(body).squeeze(-1)

    def sample(self, inputs: torch.Tensor, deterministic: bool = False) -> SceneAction:
        candidates, continuous, options, value = self._distributions(inputs)
        candidate = candidates.probs.argmax(-1) if deterministic else candidates.sample()
        option = options.probs.argmax(-1) if deterministic else options.sample()
        raw = continuous.mean if deterministic else continuous.rsample()
        controls = raw.tanh()
        log_prob = candidates.log_prob(candidate) + options.log_prob(option) + continuous.log_prob(raw).sum(-1) - torch.log(1 - controls.square() + 1e-6).sum(-1)
        return SceneAction(candidate, controls, option, log_prob, value)

    def evaluate(self, inputs: torch.Tensor, candidate: torch.Tensor, controls: torch.Tensor, option: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates, continuous, options, value = self._distributions(inputs)
        raw = torch.atanh(controls.clamp(-0.999999, 0.999999))
        logprob = candidates.log_prob(candidate) + options.log_prob(option) + continuous.log_prob(raw).sum(-1) - torch.log(1 - controls.square() + 1e-6).sum(-1)
        entropy = candidates.entropy() + options.entropy() + continuous.entropy().sum(-1)
        return logprob, entropy, value
