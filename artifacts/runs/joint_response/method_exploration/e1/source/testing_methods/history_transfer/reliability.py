"""Location-dependent source reliability from paid prequery collision likelihoods."""

import torch

from .inference import mixture_weights


def local_weights(scores, locations, paid_locations, log_likelihoods, bandwidth, *, batch_size=128):
    """Shrink local mean likelihood towards its global mean; keep raw mass fixed."""
    count = len(paid_locations)
    if len(log_likelihoods) != count:
        raise ValueError("Every reliability observation must correspond to a paid location")
    global_weights = mixture_weights(scores)
    if not count:
        return global_weights[:, None].expand(-1, len(locations))
    likelihood = torch.stack(log_likelihoods)
    global_mean = likelihood.mean(0)
    outputs = []
    for start in range(0, len(locations), batch_size):
        batch = locations[start : start + batch_size]
        distance = torch.cdist(batch, paid_locations) / bandwidth
        kernel = (-0.5 * distance.square()).exp()
        mass = kernel.sum(1)
        local_mean = kernel @ likelihood / mass.clamp_min(torch.finfo(mass.dtype).tiny)[:, None]
        strength = mass / (mass + 20)
        adjustment = strength[:, None] * count * (local_mean - global_mean)
        adjusted = scores[:, None] + adjustment.T
        weights = torch.cat((0.9 * adjusted.softmax(0), mass.new_full((1, len(batch)), 0.1)))
        outputs.append(weights)
    return torch.cat(outputs, dim=1)
