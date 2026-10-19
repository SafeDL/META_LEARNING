"""Frozen behavior-conditioned neural inference."""
from torch import nn


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
