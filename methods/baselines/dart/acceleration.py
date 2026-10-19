"""Double-precision Student CDF on CUDA, using the regularized incomplete beta."""

import numpy as np
import cupy as cp
from cupyx.scipy.special import betainc

from .fantasy import conditional_parameters


def student_cdf(df, z):
    squared = z * z
    central = squared < df
    argument = cp.where(central, squared / (df + squared), df / (df + squared))
    a = cp.where(central, 0.5, df / 2)
    b = cp.where(central, df / 2, 0.5)
    integral = betainc(a, b, argument)
    return cp.where(
        central,
        0.5 + cp.sign(z) * 0.5 * integral,
        cp.where(z > 0, 1 - 0.5 * integral, 0.5 * integral),
    )


def continuation_values(model, candidates, draws, available, count, spatial):
    output = []
    for start in range(0, len(candidates), 8):
        chunk = candidates[start : start + 8]
        parameters = [
            conditional_parameters(
                model, int(index), draws[:, start + i], available, spatial
            )
            for i, index in enumerate(chunk)
        ]
        mean = cp.asarray(np.stack([p[0] for p in parameters]))
        variance = cp.asarray(np.stack([p[1] for p in parameters]))
        beta = cp.asarray(np.stack([p[3] for p in parameters]))
        mixture = cp.exp(cp.asarray(np.stack([p[4] for p in parameters])))
        alpha = model.alpha + 0.5
        z = (mean - 0.5) / cp.sqrt(
            variance[:, None, :, None] * beta[:, :, None] / alpha
        )
        probability = (student_cdf(2 * alpha, z) * mixture[:, :, None]).sum(-1)
        exclude = cp.asarray(available[None] == chunk[:, None])
        probability = cp.where(exclude[:, None], -cp.inf, probability)
        value = (
            cp.partition(probability, probability.shape[-1] - count, axis=-1)[
                ..., -count:
            ]
            .sum(-1)
            .mean(-1)
        )
        output.extend(cp.asnumpy(value).tolist())
    return np.asarray(output)
