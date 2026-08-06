# SPDX-License-Identifier: Apache-2.0
"""The temporal-model protocol: fit on a prefix, forecast a horizon.

Anything that satisfies :class:`ForecastModel` can be backtested. It is a
structural protocol on purpose, so a naive baseline, the Bayesian forecaster the
next sprints build, and a wrapper around somebody else's model all qualify
without inheriting anything. Two rules make the harness honest: ``fit`` reads
only the data it is handed, and both calls take an explicit seed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..data import TimeSeriesData
    from ._result import ForecastResult

__all__ = ["ForecastModel"]


class ForecastModel(Protocol):
    """A model that fits a time series and forecasts forward from its end."""

    def fit(self, data: TimeSeriesData, *, seed: int = 0) -> ForecastModel:
        """Fit on ``data`` and return the fitted model (a new object, not ``self``)."""
        ...

    def forecast(self, horizon: int, *, seed: int = 0) -> ForecastResult:
        """Forecast ``horizon`` steps past the end of the fitted data."""
        ...
