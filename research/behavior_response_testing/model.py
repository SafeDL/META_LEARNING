"""Nonlinear scene responses conditioned on an offline behavioral descriptor."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BEHAVIOR_BOUNDS = np.array([[3.5, 9.0], [6.5, 9.5], [0.025, 0.5]])


def behavior_coordinates(profile):
    physical = np.array([
        profile["max_brake"], profile["desired_gap"],
        profile["perception_delay_s"]
    ])
    normalized = (physical - BEHAVIOR_BOUNDS[:, 0]) / np.diff(BEHAVIOR_BOUNDS,
                                                              axis=1)[:, 0]
    return np.r_[normalized,
                 profile["controller"] == "FVDM"].astype(np.float32)


def response_inputs(scene_coordinates, behavior):
    return np.column_stack(
        (scene_coordinates[:, :4],
         np.broadcast_to(behavior,
                         (len(scene_coordinates), 4)))).astype(np.float32)


class BehaviorResponseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 128), nn.SiLU(),
                                    nn.Linear(128, 256), nn.SiLU(),
                                    nn.Linear(256, 128), nn.SiLU(),
                                    nn.Linear(128, 2))

    def forward(self, inputs):
        output = self.layers(inputs)
        return output[:, 0].sigmoid(), output[:, 1]


def response_loss(prediction, collision_logits, risk, collision):
    return (
        ((prediction - risk).square() * (1 + 4 * risk)).mean() +
        0.1 * F.binary_cross_entropy_with_logits(collision_logits, collision))
