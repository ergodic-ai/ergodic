# SPDX-License-Identifier: Apache-2.0
"""Naive baselines: the floor every forecast table needs.

Three models, each a textbook one-liner: :class:`Naive` repeats the last value,
:class:`SeasonalNaive` repeats the last season, and :class:`Drift` extends the
line from the first observation to the last. They exist for two reasons. A
backtest without them cannot say whether a model earned its complexity, and they
make the harness testable before any sampler exists.

Each returns draws rather than a point path, so the same result object and the
same metrics serve them and the Bayesian forecasters alike. The draws are normal
around the point path with the textbook forecast variance: the scale is the
standard deviation of the model's own one-step in-sample residuals, widening
with the square root of the horizon for :class:`Naive` and :class:`Drift` and
stepping once per completed season for :class:`SeasonalNaive`.

The models are immutable, like everything else here: :meth:`fit` returns a new
fitted instance and leaves the original alone, which is what lets the backtester
hand the same unfitted object to every fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ._result import ForecastResult

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    import numpy.typing as npt

    from ..data import TimeSeriesData

__all__ = ["Drift", "Naive", "SeasonalNaive"]


def _numeric_series(data: TimeSeriesData) -> dict[str, npt.NDArray[np.float64]]:
    """The forecastable columns of ``data`` as float arrays, in schema order."""
    out: dict[str, npt.NDArray[np.float64]] = {}
    refused: list[str] = []
    for name in data.variables:
        column = data.column(name)
        if column.dtype.kind not in "iuf":
            refused.append(name)
            continue
        out[name] = np.asarray(column, dtype=np.float64)
    if refused:
        raise TypeError(
            f"the naive baselines forecast numeric series; {refused} are not numeric. "
            "Exclude them from the dataset or model them with a mechanism."
        )
    if not out:
        raise ValueError("no numeric series to forecast")
    return out


def _next_label(data: TimeSeriesData) -> Any | None:
    """The time label one step past the end, when the index is evenly spaced."""
    times = data.times
    if times.size < 2 or times.dtype.kind not in "Mmiuf":
        return None
    steps = np.diff(times)
    if not bool(np.all(steps == steps[0])):
        return None
    return times[-1] + steps[0]


def _sd(residuals: npt.NDArray[np.float64]) -> float:
    """The standard deviation of the finite residuals, zero when there are too few."""
    finite = residuals[np.isfinite(residuals)]
    if finite.size < 2:
        return 0.0
    return float(np.std(finite, ddof=1))


@dataclass(frozen=True)
class _Baseline:
    """Shared fit and forecast machinery for the three baselines."""

    n_draws: int = field(default=500, kw_only=True)
    _history: Mapping[str, npt.NDArray[np.float64]] | None = field(
        default=None, repr=False, kw_only=True
    )
    _start: Any | None = field(default=None, repr=False, kw_only=True)

    # -- the protocol --------------------------------------------------------

    def fit(self, data: TimeSeriesData, *, seed: int = 0) -> Self:
        """Read the series off ``data``; the fitted copy comes back, ``self`` is untouched."""
        history = _numeric_series(data)
        for name, values in history.items():
            self._check(name, values)
        return replace(self, _history=history, _start=_next_label(data))

    def forecast(self, horizon: int, *, seed: int = 0) -> ForecastResult:
        """Draw ``n_draws`` sample paths ``horizon`` steps past the fitted data."""
        if self._history is None:
            raise ValueError(f"{type(self).__name__} is not fitted; call fit(data) first")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        rng = np.random.default_rng(seed)
        steps = np.arange(1, horizon + 1)
        series: dict[str, npt.NDArray[np.float64]] = {}
        for name, values in self._history.items():
            path = self._path(values, horizon)
            scale = self._scale(values, steps)
            noise = rng.standard_normal((self.n_draws, horizon))
            series[name] = path[None, :] + noise * scale[None, :]
        return ForecastResult(series=series, horizon=horizon, kind="draws", start=self._start)

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._history is not None

    # -- what each baseline defines ------------------------------------------

    def _check(self, name: str, values: npt.NDArray[np.float64]) -> None:
        """Raise when a series is too short for this model."""
        raise NotImplementedError

    def _path(self, values: npt.NDArray[np.float64], horizon: int) -> npt.NDArray[np.float64]:
        """The point forecast, shape ``(horizon,)``."""
        raise NotImplementedError

    def _scale(
        self, values: npt.NDArray[np.float64], steps: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """The forecast standard deviation at each step, shape ``(horizon,)``."""
        raise NotImplementedError


@dataclass(frozen=True)
class Naive(_Baseline):
    """Repeat the last observation. The random-walk forecast, and the honest floor.

    The interval widens with ``sqrt(h)``: a random walk's uncertainty grows with
    the number of steps taken.
    """

    def _check(self, name: str, values: npt.NDArray[np.float64]) -> None:
        if values.size < 2:
            raise ValueError(f"series {name!r} has {values.size} observations; Naive needs 2")

    def _path(self, values: npt.NDArray[np.float64], horizon: int) -> npt.NDArray[np.float64]:
        return np.full(horizon, float(values[-1]), dtype=np.float64)

    def _scale(
        self, values: npt.NDArray[np.float64], steps: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        sigma = _sd(np.diff(values))
        return np.asarray(sigma * np.sqrt(steps), dtype=np.float64)


@dataclass(frozen=True)
class SeasonalNaive(_Baseline):
    """Repeat the last full season. The floor for anything with a calendar in it.

    ``period`` is the season length in rows (12 for monthly data with a yearly
    cycle). The interval is flat within a season and steps up by ``sqrt(k + 1)``
    once the forecast has to reuse a season it has already reused ``k`` times.
    """

    period: int

    def __post_init__(self) -> None:
        if self.period < 1:
            raise ValueError(f"period must be >= 1, got {self.period}")

    def _check(self, name: str, values: npt.NDArray[np.float64]) -> None:
        if values.size < self.period + 1:
            raise ValueError(
                f"series {name!r} has {values.size} observations; "
                f"SeasonalNaive(period={self.period}) needs {self.period + 1}"
            )

    def _path(self, values: npt.NDArray[np.float64], horizon: int) -> npt.NDArray[np.float64]:
        m = self.period
        season = values[-m:]
        reps = int(np.ceil(horizon / m))
        return np.asarray(np.tile(season, reps)[:horizon], dtype=np.float64)

    def _scale(
        self, values: npt.NDArray[np.float64], steps: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        m = self.period
        sigma = _sd(values[m:] - values[:-m])
        seasons = (steps - 1) // m
        return np.asarray(sigma * np.sqrt(seasons + 1), dtype=np.float64)


@dataclass(frozen=True)
class Drift(_Baseline):
    """Extend the line from the first observation to the last.

    The slope is ``(y_T - y_1) / (T - 1)``, which is what a random walk with
    drift estimates. The interval widens with ``sqrt(h (1 + h / (T - 1)))``: the
    random-walk term plus the uncertainty in the slope itself.
    """

    def _check(self, name: str, values: npt.NDArray[np.float64]) -> None:
        if values.size < 3:
            raise ValueError(f"series {name!r} has {values.size} observations; Drift needs 3")

    @staticmethod
    def _slope(values: npt.NDArray[np.float64]) -> float:
        return float((values[-1] - values[0]) / (values.size - 1))

    def _path(self, values: npt.NDArray[np.float64], horizon: int) -> npt.NDArray[np.float64]:
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        return np.asarray(float(values[-1]) + steps * self._slope(values), dtype=np.float64)

    def _scale(
        self, values: npt.NDArray[np.float64], steps: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        length = float(values.size)
        sigma = _sd(np.diff(values) - self._slope(values))
        h = steps.astype(np.float64)
        return np.asarray(sigma * np.sqrt(h * (1.0 + h / (length - 1.0))), dtype=np.float64)
