# SPDX-License-Identifier: Apache-2.0
"""Small numeric helpers the estimators share, kept free of scipy.

``_norm_ppf`` is the inverse standard-normal CDF (Acklam's rational
approximation, accurate to about 1e-9), used for normal confidence intervals.
``_sigmoid`` is the stable logistic function used by the native logistic learner.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF for ``0 < p < 1``."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"_norm_ppf needs 0 < p < 1, got {p}")
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p <= 1.0 - plow:
        q = p - 0.5
        r = q * q
        return (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
            * q
            / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
        (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    )


def _sigmoid(eta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Logistic function, clipped to avoid overflow."""
    clipped = np.clip(eta, -30.0, 30.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)
