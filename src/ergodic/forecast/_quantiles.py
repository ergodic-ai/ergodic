# SPDX-License-Identifier: Apache-2.0
"""Quantile forecasts as draws: inverse-CDF sampling over a few quantile levels.

A quantile model (Chronos-2 is one) answers each forecast step with a handful of
quantiles rather than sample paths. Every consumer in this pillar reads ``(S, H)``
draws: the metrics, the backtester, reconciliation, the coherence check. So
:func:`quantile_draws` turns quantiles into draws once, at the boundary, and
everything downstream stays one code path.

The method, step by step:

1. Sort each step's quantiles (monotone rearrangement), so a crossing pair in the
   model's output can never make the quantile function run backwards.
2. Between the outermost levels the quantile function is the piecewise-linear
   interpolant through the ``(level, value)`` knots.
3. Below the lowest level ``p_1`` and above the highest ``p_K`` the tails are
   exponential: ``q_1 + b_lo * log(u / p_1)`` and ``q_K - b_hi * log((1 - u) /
   (1 - p_K))``, with ``b_lo`` and ``b_hi`` set so the density is continuous at
   the outer knots. The tails are unbounded but light, and a flat outer segment
   gives a flat tail.
4. Draw ``S`` independent uniforms per step and push them through.

The draws match the model's marginal at every step and nothing more: each step
is sampled on its own, so a path carries **no dependence across steps** and two
series carry none between them. Anything that reads one step at a time is exact
up to Monte Carlo error (the median, an interval, coverage, the per-step CRPS).
Anything that adds steps or series up (a total over a window, a bottom-up
aggregate) treats them as independent, which understates the spread of a sum
when the real errors move together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = ["quantile_draws"]

# Uniforms are kept this far from 0 and 1, so the exponential tails stay finite.
_EDGE = 1e-12


def _levels(levels: Sequence[float]) -> npt.NDArray[np.float64]:
    out = np.asarray(list(levels), dtype=np.float64)
    if out.ndim != 1 or out.size < 2:
        raise ValueError(f"need at least two quantile levels, got {list(levels)}")
    if not bool(np.all(np.isfinite(out))) or out[0] <= 0.0 or out[-1] >= 1.0:
        raise ValueError(f"quantile levels must lie strictly inside (0, 1), got {list(levels)}")
    if not bool(np.all(np.diff(out) > 0.0)):
        raise ValueError(f"quantile levels must be strictly increasing, got {list(levels)}")
    return out


def quantile_draws(
    levels: Sequence[float],
    quantiles: npt.NDArray[Any] | Any,
    *,
    draws: int = 1000,
    seed: int | np.random.Generator = 0,
) -> npt.NDArray[np.float64]:
    """Sample draws from per-step quantiles by inverse-CDF sampling.

    Parameters
    ----------
    levels
        The quantile levels, strictly increasing inside ``(0, 1)``, at least two.
        More levels pin the shape down better: with only ``[0.1, 0.5, 0.9]`` a
        fifth of the mass lives in the assumed exponential tails.
    quantiles
        The quantile values, shape ``(H, K)``: one row per forecast step, one
        column per level. Rows that are not sorted are sorted first.
    draws
        How many draws per step, ``S``.
    seed
        An integer seed or a ``numpy.random.Generator``. The same seed gives the
        same draws.

    Returns
    -------
    numpy.ndarray
        Shape ``(S, H)``, ready for a :class:`~ergodic.forecast.ForecastResult`.
        Steps are sampled independently (see the module notes).

    Raises
    ------
    ValueError
        Bad levels, a shape that does not match them, a non-finite quantile, or
        ``draws < 1``.
    """
    p = _levels(levels)
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != p.size:
        raise ValueError(f"quantiles must have shape (H, {p.size}), got {q.shape}")
    if q.shape[0] < 1:
        raise ValueError("quantiles need at least one step")
    if not bool(np.all(np.isfinite(q))):
        raise ValueError("quantiles must be finite")
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws < 1:
        raise ValueError(f"draws must be a positive integer, got {draws!r}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    q = np.sort(q, axis=1)
    horizon = q.shape[0]
    u = np.clip(rng.random((int(draws), horizon)), _EDGE, 1.0 - _EDGE)
    out = np.empty_like(u)
    # Density-matching slopes of the two exponential tails, one per step.
    b_lo = p[0] * (q[:, 1] - q[:, 0]) / (p[1] - p[0])
    b_hi = (1.0 - p[-1]) * (q[:, -1] - q[:, -2]) / (p[-1] - p[-2])
    for step in range(horizon):
        column = u[:, step]
        values = np.interp(column, p, q[step])
        low = column < p[0]
        high = column > p[-1]
        values[low] = q[step, 0] + b_lo[step] * np.log(column[low] / p[0])
        values[high] = q[step, -1] - b_hi[step] * np.log((1.0 - column[high]) / (1.0 - p[-1]))
        out[:, step] = values
    return out
