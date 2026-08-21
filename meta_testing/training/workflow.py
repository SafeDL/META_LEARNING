"""Stage-aware module freezing for the prescribed MVR training order."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from torch import nn

from .stages import TrainingStage, trainable_components


@dataclass
class StagedWorkflow:
    components: Mapping[str, nn.Module]
    stage: TrainingStage | None = None

    def activate(self, stage: TrainingStage) -> frozenset[str]:
        enabled = trainable_components(stage)
        unknown = enabled - set(self.components)
        if unknown:
            raise ValueError(f"workflow is missing required components: {sorted(unknown)}")
        for name, module in self.components.items():
            active = name in enabled
            for parameter in module.parameters():
                parameter.requires_grad_(active)
        self.stage = stage
        return enabled

    def assert_ready(self, expected: TrainingStage) -> None:
        if self.stage != expected:
            raise RuntimeError(f"expected active stage {expected.value}, got {self.stage}")
