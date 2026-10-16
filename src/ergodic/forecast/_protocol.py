# SPDX-License-Identifier: Apache-2.0
"""The temporal-model protocol: fit on a prefix, forecast a horizon.

Anything that satisfies :class:`ForecastModel` can be backtested. It is a
structural protocol on purpose, so a naive baseline, the Bayesian forecaster the
next sprints build, and a wrapper around somebody else's model all qualify
without inheriting anything. Two rules make the harness honest: ``fit`` reads
only the data it is handed, and both calls take an explicit seed.

A model that reads covariates whose future is known (a promotion calendar, a
holiday flag, a price list) also satisfies :class:`CovariateForecastModel`: it
names those columns in ``future_covariates`` and its ``forecast`` takes their
values over the horizon as ``future=``. :func:`~ergodic.forecast.backtest` reads
the attribute, and at each origin hands the model the true covariate values of
the window it forecasts, those columns only, so the targets never leak. A model
without the attribute is called exactly as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..data import TimeSeriesData
    from ._result import ForecastResult

__all__ = ["CovariateForecastModel", "ForecastModel"]


class ForecastModel(Protocol):
    """A model that fits a time series and forecasts forward from its end."""

    def fit(self, data: TimeSeriesData, *, seed: int = 0) -> ForecastModel:
        """Fit on ``data`` and return the fitted model (a new object, not ``self``)."""
        ...

    def forecast(self, horizon: int, *, seed: int = 0) -> ForecastResult:
        """Forecast ``horizon`` steps past the end of the fitted data."""
        ...


class CovariateForecastModel(ForecastModel, Protocol):
    """A :class:`ForecastModel` that reads covariates whose future values are known.

    ``future_covariates`` names the columns. ``forecast`` takes their values over
    the horizon as ``future``: a :class:`~ergodic.data.TimeSeriesData` holding the
    time index and those columns, ``horizon`` rows. The backtester builds it from
    the held-out window, so it never carries a target.
    """

    @property
    def future_covariates(self) -> tuple[str, ...]:
        """The covariate columns whose values over the horizon the model reads."""
        ...

    def forecast(
        self, horizon: int, *, seed: int = 0, future: TimeSeriesData | None = None
    ) -> ForecastResult:
        """Forecast ``horizon`` steps, reading the known covariates from ``future``."""
        ...
