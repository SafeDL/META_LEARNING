"""Pure virtual conditioning and whole-pool continuation integration."""

import numpy as np
from scipy.special import stdtr
import torch

from methods.baselines.dart.planning import top_sum

from .config import FANTASIES


def conditional_probabilities(model, index, values, indices):
    output = np.broadcast_to(
        model.probabilities()[indices], (len(values), len(indices))
    ).copy()
    same = model.family[indices] == model.family[index]
    block = model.blocks[int(model.family[index])][1]
    mean, variance, alpha, beta, weight = block.conditional(
        model.positions[index], values, model.positions[indices[same]]
    )
    z = (mean - 0.5) / np.sqrt(variance[None] * beta[:, None] / alpha)
    output[:, same] = (stdtr(2 * alpha, z) * np.exp(weight[:, None])).sum(-1)
    return output


def acquisition_values(
    model, candidates, remaining, horizon, seed, *, draws=None, gpu=None
):
    probability = model.probabilities()
    if horizon == 1:
        return probability[candidates].copy()
    if draws is None:
        draws = model.draws(candidates, FANTASIES, np.random.default_rng(seed))
    available = np.flatnonzero(remaining)
    count = horizon - 1
    gpu = torch.cuda.is_available() if gpu is None else gpu
    if not gpu:
        values = []
        for column, index in enumerate(candidates):
            future = available[available != index]
            conditional = conditional_probabilities(
                model, index, draws[:, column], future
            )
            values.append(probability[index] + top_sum(conditional, count).mean())
        return np.asarray(values)
    import cupy as cp
    from methods.common.student import student_cdf

    output = probability[candidates].copy()
    for family, (_, block) in model.blocks.items():
        positions = np.flatnonzero(model.family[candidates] == family)
        same = np.flatnonzero(model.family[available] == family)
        local_indices = model.positions[available[same]]
        for start in range(0, len(positions), 16):
            columns = positions[start : start + 16]
            parameters = [
                block.conditional(
                    model.positions[candidates[c]], draws[:, c], local_indices
                )
                for c in columns
            ]
            mean = cp.asarray(np.stack([p[0] for p in parameters]))
            variance = cp.asarray(np.stack([p[1] for p in parameters]))
            beta = cp.asarray(np.stack([p[3] for p in parameters]))
            weight = cp.exp(cp.asarray(np.stack([p[4] for p in parameters])))
            alpha = block.alpha + 0.5
            z = (mean - 0.5) / cp.sqrt(variance[:, None] * beta[:, :, None] / alpha)
            updated = (student_cdf(2 * alpha, z) * weight[:, :, None]).sum(-1)
            future = cp.broadcast_to(
                cp.asarray(probability[available]),
                (len(columns), len(draws), len(available)),
            ).copy()
            future[:, :, cp.asarray(same)] = updated
            exclude = cp.asarray(available[None] == candidates[columns, None])
            future = cp.where(exclude[:, None], -cp.inf, future)
            # Small whole-pool batches are faster to partition on the CPU.
            output[columns] += top_sum(cp.asnumpy(future), count).mean(-1)
    return output
