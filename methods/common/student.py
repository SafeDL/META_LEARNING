"""Exact integer-degree Student CDF via a finite cosine-integral recurrence."""

from functools import lru_cache

import cupy as cp
import numpy as np

from methods.baselines.dart.acceleration import (
    student_cdf as general_student_cdf,
)


CDF = cp.ElementwiseKernel(
    "float64 z, int32 df, raw float64 coefficients",
    "float64 output",
    r"""
    if (!isfinite(z)) {
        output = isnan(z) ? z : (z > 0 ? 1.0 : 0.0);
    } else {
        double norm = hypot(sqrt((double)df), z);
        double y = z / norm;
        double s = ((double)df / norm) / norm;
        double polynomial = 0.0;
        for (int j = df / 2 - 1; j >= 0; --j) {
            polynomial = fma(polynomial, s, coefficients[j]);
        }
        output = (df % 2)
          ? 0.5 + (atan(z / sqrt((double)df)) + y * sqrt(s) * polynomial)
                    * 0.318309886183790671537767526745
          : 0.5 + 0.5 * y * polynomial;
        output = fmin(1.0, fmax(0.0, output));
    }
    """,
    "selective_integer_student_cdf",
)


@lru_cache(maxsize=256)
def coefficients(df, device):
    values = np.ones(df // 2, dtype=np.float64)
    for j in range(1, len(values)):
        ratio = j / (j + 0.5) if df % 2 else (j - 0.5) / j
        values[j] = values[j - 1] * ratio
    with cp.cuda.Device(device):
        return cp.asarray(values)


def student_cdf(df, z):
    if not float(df).is_integer() or df < 1:
        return general_student_cdf(df, z)
    degree = int(df)
    return CDF(z, np.int32(degree), coefficients(degree, cp.cuda.runtime.getDevice()))
