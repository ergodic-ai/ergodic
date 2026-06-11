# SPDX-License-Identifier: Apache-2.0
"""Granger causality as a table of predictive p-values, not a graph.

Granger answers a predictive question: do the lags of ``X`` improve the
forecast of ``Y`` once ``Y``'s own lags and every other variable's lags are
in the regression? That is not a structural question. The test is blind to
confounding (a common driver makes ``X`` "Granger-cause" ``Y`` without any
causal link) and to contemporaneous effects. That is why the result here is
a table of ordered pairs and never a graph; for structure, use
:class:`~ergodic.discovery.algorithms._pcmci.PCMCI`.

Each ordered pair gets a likelihood-ratio statistic,
``n * (log RSS_restricted - log RSS_full)``, asymptotically chi-squared with
``max_lag`` degrees of freedom. We use the likelihood-ratio form because the
codebase carries a chi-squared survival function and no F distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..data import TimeSeriesData
from ._lagged import lagged_node, lagged_tabular
from ._stats import _chi2_sf

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

__all__ = ["GrangerResult", "granger"]

_RSS_FLOOR = 1e-300


@dataclass(frozen=True)
class GrangerResult:
    """Granger tests for every ordered pair: a predictive table, not a graph.

    ``pvalues`` maps ``(cause, effect)`` to the likelihood-ratio p-value;
    ``significant`` holds the pairs with ``p <= alpha``. ``n`` counts the
    stacked window rows the regressions used.
    """

    pvalues: Mapping[tuple[str, str], float]
    significant: frozenset[tuple[str, str]]
    max_lag: int
    alpha: float
    n: int


def _rss(design: npt.NDArray[np.float64], response: npt.NDArray[np.float64]) -> float:
    """The residual sum of squares of an ordinary least-squares fit."""
    coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coef
    return float(residual @ residual)


def granger(data: TimeSeriesData, *, max_lag: int = 1, alpha: float = 0.05) -> GrangerResult:
    """Granger-test every ordered pair of variables in ``data``.

    For each effect ``Y``, the full regression puts ``Y[t]`` on an intercept
    plus every variable's lags ``1..max_lag``; the restricted one drops the
    candidate cause's lags. All variables must be continuous. See the module
    docstring for what this does and does not mean.
    """
    if not isinstance(data, TimeSeriesData):
        raise TypeError(
            f"granger needs TimeSeriesData, got {type(data).__name__}; "
            "build one with ergodic.data.timeseries"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")
    offenders = [name for name in data.variables if not data.schema[name].is_continuous]
    if offenders:
        raise ValueError(f"granger needs continuous variables; not continuous: {offenders}")
    frame = lagged_tabular(data, max_lag).dropna()
    variables = sorted(data.variables)
    lag_owners = [v for v in variables for _ in range(1, max_lag + 1)]
    lag_columns = [lagged_node(v, k) for v in variables for k in range(1, max_lag + 1)]
    n = frame.n_rows
    n_params = 1 + len(lag_columns)
    if n < n_params + 2:
        raise ValueError(
            f"need at least {n_params + 2} complete rows for {n_params} regressors, got {n}"
        )
    full = np.hstack([np.ones((n, 1), dtype=np.float64), frame.matrix(lag_columns)])
    pvalues: dict[tuple[str, str], float] = {}
    for effect in variables:
        response = frame.matrix([lagged_node(effect, 0)])[:, 0]
        rss_full = math.log(max(_rss(full, response), _RSS_FLOOR))
        for cause in variables:
            if cause == effect:
                continue
            keep = [0] + [i + 1 for i, owner in enumerate(lag_owners) if owner != cause]
            rss_restricted = math.log(max(_rss(full[:, keep], response), _RSS_FLOOR))
            statistic = max(0.0, n * (rss_restricted - rss_full))
            pvalues[(cause, effect)] = _chi2_sf(statistic, float(max_lag))
    significant = frozenset(pair for pair, p in pvalues.items() if p <= alpha)
    return GrangerResult(
        pvalues=pvalues, significant=significant, max_lag=max_lag, alpha=alpha, n=n
    )
