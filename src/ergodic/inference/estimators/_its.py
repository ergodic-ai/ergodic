# SPDX-License-Identifier: Apache-2.0
"""Interrupted time series: segmented regression around one known break.

One series, one break the analyst names. Fit a level and trend plus a
post-break jump and slope change by OLS, optionally with Fourier seasonality,
and read the effect as the average post-period gap to the projected pre-trend.
Standard errors are Newey-West (HAC) with Bartlett weights, since one series's
residuals autocorrelate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._design import Design
from .._result import EffectEstimate
from .._target import EffectTarget

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...data import TimeSeriesData

__all__ = ["ITSEffect", "interrupted_time_series"]


@dataclass(frozen=True, repr=False)
class ITSEffect(EffectEstimate):
    """A segmented-regression fit around one break.

    ``level_change`` is the jump at the break and ``slope_change`` the change
    in trend per period, each with a Newey-West standard error. ``value`` is
    the ATT: the average post-period gap between ``fitted`` and
    ``counterfactual``, where ``counterfactual`` projects the pre-period trend
    over every period with the break terms zeroed and the seasonal terms kept.
    """

    design: Design | None = None
    at: Any = None
    level_change: float = float("nan")
    level_change_se: float = float("nan")
    slope_change: float = float("nan")
    slope_change_se: float = float("nan")
    times: tuple[Any, ...] = ()
    observed: tuple[float, ...] = ()
    counterfactual: tuple[float, ...] = ()
    fitted: tuple[float, ...] = ()


def _hac_cov(
    x: npt.NDArray[np.float64], resid: npt.NDArray[np.float64], lags: int
) -> npt.NDArray[np.float64]:
    """Newey-West coefficient covariance with Bartlett weights over ``lags`` lags."""
    scores = x * resid[:, None]
    meat = np.asarray(scores.T @ scores, dtype=np.float64)
    for lag in range(1, lags + 1):
        gamma = np.asarray(scores[lag:].T @ scores[:-lag], dtype=np.float64)
        meat = meat + (1.0 - lag / (lags + 1.0)) * (gamma + gamma.T)
    bread = np.linalg.pinv(x.T @ x)
    return np.asarray(bread @ meat @ bread, dtype=np.float64)


def interrupted_time_series(
    data: TimeSeriesData,
    outcome: str,
    *,
    at: Any,
    seasonality: int | None = None,
    hac_lags: int | None = None,
) -> ITSEffect:
    """Fit an interrupted time series for ``outcome`` with a break at ``at``.

    The break index is the first row whose time value is at least ``at``; both
    sides need at least 3 points. The design matrix over row positions
    ``t = 0..T-1`` is ``[1, t, post, post * (t - t0)]``, plus, with
    ``seasonality=m`` (between 2 and ``T // 2``), Fourier pairs
    ``sin/cos(2 pi k t / m)`` for ``k = 1..min(3, m // 2)``. OLS fits it, and
    the coefficient covariance is Newey-West with Bartlett weights over
    ``hac_lags`` lags, default ``floor(0.75 * T ** (1/3))``.

    ``level_change`` is the post coefficient and ``slope_change`` the
    interaction. The headline value is the ATT, the mean post-period gap
    between the fit and the pre-trend projection, which equals
    ``level_change + slope_change * mean(t - t0 | post)``; its standard error
    comes from the matching linear combination of the HAC covariance. The
    ``counterfactual`` path zeroes the break terms but keeps the trend and
    seasonal terms, all at the fitted coefficients.

    A NaN outcome raises, as does a break too close to either end.
    """
    if outcome not in data.variables:
        raise ValueError(
            f"outcome {outcome!r} is not a variable of the series; have {sorted(data.variables)}"
        )
    n = data.n_rows
    y = data.matrix([outcome])[:, 0]
    if np.isnan(y).any():
        raise ValueError(
            f"outcome {outcome!r} has missing values; interrupted time series "
            "needs a complete series"
        )
    if seasonality is not None and not 2 <= seasonality <= n // 2:
        raise ValueError(
            f"seasonality must be between 2 and half the series length ({n // 2}), "
            f"got {seasonality}"
        )
    if hac_lags is not None and hac_lags < 0:
        raise ValueError(f"hac_lags must be nonnegative, got {hac_lags}")

    times_list = list(data.times.tolist())
    try:
        post_flags = [bool(value >= at) for value in times_list]
    except TypeError as exc:
        raise ValueError(f"at={at!r} is not comparable to the time column {data.time!r}") from exc
    t0 = post_flags.index(True) if any(post_flags) else n
    n_pre = t0
    n_post = n - t0
    if n_pre < 3 or n_post < 3:
        raise ValueError(
            "the break needs at least 3 points on each side, "
            f"got {n_pre} pre and {n_post} post for at={at!r}"
        )

    t = np.arange(n, dtype=np.float64)
    post = (t >= t0).astype(np.float64)
    cols = [np.ones(n, dtype=np.float64), t, post, post * (t - float(t0))]
    if seasonality is not None:
        for k in range(1, min(3, seasonality // 2) + 1):
            angle = 2.0 * np.pi * k * t / float(seasonality)
            cols.append(np.sin(angle))
            cols.append(np.cos(angle))
    x = np.asarray(np.column_stack(cols), dtype=np.float64)

    beta = np.asarray(np.linalg.lstsq(x, y, rcond=None)[0], dtype=np.float64)
    resid = np.asarray(y - x @ beta, dtype=np.float64)
    lags = hac_lags if hac_lags is not None else int(np.floor(0.75 * float(n) ** (1.0 / 3.0)))
    cov = _hac_cov(x, resid, lags)

    level_change = float(beta[2])
    slope_change = float(beta[3])
    level_change_se = float(np.sqrt(max(float(cov[2, 2]), 0.0)))
    slope_change_se = float(np.sqrt(max(float(cov[3, 3]), 0.0)))
    contrast = np.zeros(x.shape[1], dtype=np.float64)
    contrast[2] = 1.0
    contrast[3] = float(np.mean(t[t0:] - float(t0)))
    value = float(contrast @ beta)
    std_error = float(np.sqrt(max(float(contrast @ cov @ contrast), 0.0)))

    fitted = np.asarray(x @ beta, dtype=np.float64)
    x_counter = x.copy()
    x_counter[:, 2] = 0.0
    x_counter[:, 3] = 0.0
    counterfactual = np.asarray(x_counter @ beta, dtype=np.float64)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    design = Design(
        strategy="interrupted_time_series",
        outcome=outcome,
        treatment=None,
        assumptions=(
            "the pre-period trend would have continued without the break",
            "no co-timed shock at the break",
        ),
        estimand="average post-period gap to the projected pre-trend",
    )
    return ITSEffect(
        target=EffectTarget.ATT,
        estimator="interrupted_time_series",
        learners={},
        n=n,
        kind="analytic",
        diagnostics={
            "n_pre": float(n_pre),
            "n_post": float(n_post),
            "hac_lags": float(lags),
            "r2": r2,
        },
        _point=value,
        _std_error=std_error,
        design=design,
        at=at,
        level_change=level_change,
        level_change_se=level_change_se,
        slope_change=slope_change,
        slope_change_se=slope_change_se,
        times=tuple(times_list),
        observed=tuple(float(v) for v in y),
        counterfactual=tuple(float(v) for v in counterfactual),
        fitted=tuple(float(v) for v in fitted),
    )
