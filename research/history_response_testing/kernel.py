"""Similarity of full historical responses, with a physical residual channel."""
import numpy as np
import torch

from methods.history_guided_testing.kernel import matern52

from .config import (
    GP_AMPLITUDE,
    PHYSICAL_LENGTH,
    PHYSICAL_WEIGHT,
    RESPONSE_LENGTH,
)


def covariance(x, risk_predictions, device="cuda"):
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    responses = torch.as_tensor(risk_predictions,
                                dtype=torch.float64,
                                device=device)
    # RMS distances keep the response dimension comparable when a group is excluded.
    response_distance = torch.cdist(responses, responses) / np.sqrt(
        responses.shape[1])
    physical_distance = torch.cdist(x[:, :4], x[:, :4])
    response = matern52(response_distance, RESPONSE_LENGTH)
    physical = matern52(physical_distance, PHYSICAL_LENGTH)
    return GP_AMPLITUDE**2 * (
        (1 - PHYSICAL_WEIGHT) * response +
        PHYSICAL_WEIGHT * physical) * (x[:, 4:5] == x[:, 4:5].T)
