# SPDX-License-Identifier: Apache-2.0
"""Numeric helpers for the discovery tests, kept free of scipy.

``_norm_cdf`` is the standard normal CDF (through ``math.erf``). ``_chi2_sf``
is the chi-square survival function, computed from the regularized incomplete
gamma function by series or continued fraction (the classic split at s + 1).
"""

from __future__ import annotations

import math

_EPS = 1e-14
_MAX_ITER = 500


def _norm_cdf(x: float) -> float:
    """The standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _lower_series(s: float, x: float) -> float:
    """Regularized lower incomplete gamma P(s, x) by series; needs x < s + 1."""
    if x <= 0.0:
        return 0.0
    ap = s
    term = 1.0 / s
    total = term
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + s * math.log(x) - math.lgamma(s))


def _upper_fraction(s: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(s, x) by continued fraction; needs x >= s + 1."""
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + s * math.log(x) - math.lgamma(s)) * h


def _chi2_sf(x: float, df: float) -> float:
    """The chi-square survival function P(X > x) for df > 0."""
    if df <= 0.0:
        raise ValueError(f"chi-square needs df > 0, got {df}")
    if x <= 0.0:
        return 1.0
    s = df / 2.0
    half = x / 2.0
    p = 1.0 - _lower_series(s, half) if half < s + 1.0 else _upper_fraction(s, half)
    return min(1.0, max(0.0, p))
