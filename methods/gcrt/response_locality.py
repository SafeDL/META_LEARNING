"""Fixed historical-response geometry for local source reliability."""

import torch


def squared_distance(public_x, source_probabilities, response_fraction):
    if not 0 <= response_fraction <= 1:
        raise ValueError('The response-distance fraction must lie in [0, 1]')
    if (public_x.ndim != 2 or source_probabilities.ndim != 2
            or source_probabilities.shape[1] != len(public_x)
            or not public_x.shape[1] or not source_probabilities.shape[0]):
        raise ValueError('Locality needs public features and source-by-candidate probabilities')
    physical = torch.square(public_x[:, None]-public_x[None]).mean(-1)
    response = torch.square(source_probabilities.T[:, None]-source_probabilities.T[None]).mean(-1)
    return (1-response_fraction)*physical+response_fraction*response


def influence_matrix(distance, bandwidth):
    if bandwidth is None:
        return torch.ones_like(distance)
    if bandwidth <= 0:
        raise ValueError('Local reliability bandwidth must be positive')
    return (-distance/(2*bandwidth**2)).exp()
