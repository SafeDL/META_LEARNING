"""Fixed configuration for the highway-env DIVA-Mine MVP."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """The deliberately small experiment described in the MVP design."""

    num_anchors: int = 128
    prior_rank: int = 2
    support_budget: int = 4
    total_budget: int = 20
    random_support_repeats: int = 20
    seed: int = 20260912
    differential_weight: float = 1.0
    shared_weight: float = 0.10
    uncertainty_weight: float = 0.25
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0

    @property
    def mining_budget(self) -> int:
        return self.total_budget - self.support_budget

    def validate(self) -> None:
        """Validate settings shared by legacy and differential experiments."""
        if self.prior_rank < 1:
            raise ValueError("prior_rank must be positive")
        if self.total_budget > self.num_anchors:
            raise ValueError("total_budget cannot exceed num_anchors")
        if self.differential_weight < 0:
            raise ValueError("differential_weight must be non-negative")
        if self.shared_weight < 0 or self.uncertainty_weight < 0:
            raise ValueError("acquisition weights must be non-negative")
        if self.beta_prior_alpha <= 0 or self.beta_prior_beta <= 0:
            raise ValueError("Beta smoothing parameters must be positive")

    def validate_legacy_mining(self) -> None:
        """Validate the K-shot setting still required by legacy baselines."""
        self.validate()
        if not 0 < self.support_budget < self.total_budget:
            raise ValueError("support_budget must be in (0, total_budget)")


@dataclass(frozen=True)
class RegressionExperimentConfig:
    """Minimal fixed configuration for the active PR-BRVT experiment."""

    num_anchors: int = 128
    prior_rank: int = 2
    total_budget: int = 20
    random_support_repeats: int = 20
    seed: int = 20260912
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0
    critical_threshold: float = 0.75

    def validate(self) -> None:
        """Validate PR-BRVT settings without legacy acquisition parameters."""
        if self.prior_rank < 1:
            raise ValueError("prior_rank must be positive")
        if not 0 < self.total_budget <= self.num_anchors:
            raise ValueError("total_budget must be in [1, num_anchors]")
        if self.random_support_repeats < 1:
            raise ValueError("random_support_repeats must be positive")
        if self.beta_prior_alpha <= 0 or self.beta_prior_beta <= 0:
            raise ValueError("Beta smoothing parameters must be positive")
        if not 0.0 < self.critical_threshold < 1.0:
            raise ValueError("critical_threshold must be in (0, 1)")


@dataclass(frozen=True)
class VersionRegressionConfig:
    """Frozen settings for chronological version-regression replay."""

    num_anchors: int = 128
    calibration_anchors: int = 32
    prior_rank: int = 2
    total_budget: int = 20
    random_repeats: int = 20
    evaluation_seed: int = 2026091303
    calibration_seed: int = 2026091301
    critical_threshold: float = 0.75
    observation_noise: float = 0.03
    target_orders: tuple[int, ...] = (4, 5, 6)

    def validate(self) -> None:
        """Reject settings that would violate the frozen protocol."""
        if self.num_anchors < 1 or self.calibration_anchors < 1:
            raise ValueError("anchor counts must be positive")
        if not 0 < self.total_budget <= self.num_anchors:
            raise ValueError("total_budget must be in [1, num_anchors]")
        if self.prior_rank < 1:
            raise ValueError("prior_rank must be positive")
        if self.random_repeats < 1:
            raise ValueError("random_repeats must be positive")
        if not 0.0 < self.critical_threshold < 1.0:
            raise ValueError("critical_threshold must be in (0, 1)")
        if self.observation_noise <= 0:
            raise ValueError("observation_noise must be positive")
        if not self.target_orders or min(self.target_orders) < 2:
            raise ValueError("target_orders must contain version orders from 2 onward")
