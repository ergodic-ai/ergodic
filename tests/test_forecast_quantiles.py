# SPDX-License-Identifier: Apache-2.0
"""Quantiles to draws, and known future covariates through the backtester.

Pure numpy: no sampler, no network. The quantile model the conversion serves is
Welt's Chronos-2 (tests/test_welt_forecast.py); here it is held to its own
contract on synthetic quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ergodic import timeseries
from ergodic.data import TimeSeriesData
from ergodic.forecast import ForecastResult, Naive, backtest, quantile_draws

LEVELS = (0.01, *(round(k / 20, 2) for k in range(1, 20)), 0.99)


def _normal_quantiles(mean: np.ndarray, sd: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    z = np.array([NormalDist().inv_cdf(p) for p in levels])
    return np.asarray(mean[:, None] + sd[:, None] * z[None, :], dtype=np.float64)


def test_draws_recover_the_quantiles_they_came_from() -> None:
    mean = np.array([10.0, 12.0, 15.0])
    sd = np.array([1.0, 2.0, 4.0])
    q = _normal_quantiles(mean, sd, LEVELS)
    draws = quantile_draws(LEVELS, q, draws=200_000, seed=3)
    assert draws.shape == (200_000, 3)
    assert np.all(np.isfinite(draws))
    empirical = np.quantile(draws, LEVELS, axis=0).T
    np.testing.assert_allclose(empirical, q, atol=0.03 * sd.max())
    # The interior is linear between knots, so the median is the native median.
    np.testing.assert_allclose(np.median(draws, axis=0), mean, atol=0.02 * sd.max())


def test_the_tails_hold_the_mass_beyond_the_outer_levels_exponentially() -> None:
    q = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    levels = (0.05, 0.25, 0.5, 0.75, 0.95)
    draws = quantile_draws(levels, q, draws=400_000, seed=1)[:, 0]
    assert np.mean(draws < 0.0) == pytest.approx(0.05, abs=0.002)
    assert np.mean(draws > 4.0) == pytest.approx(0.05, abs=0.002)
    # Density-matched exponential tail: scale p1 * (q2 - q1) / (p2 - p1) = 0.25.
    below = draws[draws < 0.0]
    assert np.mean(0.0 - below) == pytest.approx(0.25, rel=0.03)
    above = draws[draws > 4.0]
    assert np.mean(above - 4.0) == pytest.approx(0.25, rel=0.03)


def test_a_flat_outer_segment_makes_a_flat_tail() -> None:
    q = np.array([[5.0, 5.0, 6.0, 7.0, 7.0]])
    draws = quantile_draws((0.1, 0.3, 0.5, 0.7, 0.9), q, draws=10_000, seed=0)
    assert draws.min() == 5.0 and draws.max() == 7.0


def test_crossing_quantiles_are_sorted_before_sampling() -> None:
    crossed = np.array([[1.0, 3.0, 2.0], [0.0, 1.0, 2.0]])
    ordered = np.sort(crossed, axis=1)
    a = quantile_draws((0.1, 0.5, 0.9), crossed, draws=500, seed=7)
    b = quantile_draws((0.1, 0.5, 0.9), ordered, draws=500, seed=7)
    np.testing.assert_array_equal(a, b)


def test_the_quantile_function_is_monotone_in_the_uniform() -> None:
    """Same uniforms, sorted: the draws come out sorted, step by step."""
    q = _normal_quantiles(np.array([0.0, 5.0]), np.array([1.0, 3.0]), LEVELS)
    rng = np.random.default_rng(0)
    draws = quantile_draws(LEVELS, q, draws=5_000, seed=rng)
    replay = np.random.default_rng(0).random((5_000, 2))
    for step in range(2):
        by_u = draws[np.argsort(replay[:, step]), step]
        assert np.all(np.diff(by_u) >= 0.0)


def test_the_seed_reproduces_the_draws() -> None:
    q = _normal_quantiles(np.zeros(4), np.ones(4), LEVELS)
    a = quantile_draws(LEVELS, q, draws=300, seed=11)
    b = quantile_draws(LEVELS, q, draws=300, seed=11)
    c = quantile_draws(LEVELS, q, draws=300, seed=12)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_steps_are_sampled_independently() -> None:
    """Per-step marginals only: two steps with the same quantiles are uncorrelated."""
    q = _normal_quantiles(np.zeros(2), np.ones(2), LEVELS)
    draws = quantile_draws(LEVELS, q, draws=50_000, seed=5)
    assert abs(np.corrcoef(draws[:, 0], draws[:, 1])[0, 1]) < 0.02


@pytest.mark.parametrize(
    ("levels", "quantiles", "draws", "match"),
    [
        ((0.5,), [[1.0]], 10, "at least two"),
        ((0.0, 0.5), [[1.0, 2.0]], 10, "strictly inside"),
        ((0.5, 0.1), [[1.0, 2.0]], 10, "strictly increasing"),
        ((0.1, 0.9), [[1.0, 2.0, 3.0]], 10, "shape"),
        ((0.1, 0.9), [[1.0, np.nan]], 10, "finite"),
        ((0.1, 0.9), [[1.0, 2.0]], 0, "positive integer"),
    ],
)
def test_bad_inputs_are_refused(
    levels: tuple[float, ...], quantiles: list[list[float]], draws: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        quantile_draws(levels, np.asarray(quantiles), draws=draws)


# -- known future covariates through the backtester ----------------------------


def _sales(n: int = 40) -> TimeSeriesData:
    rng = np.random.default_rng(0)
    day = pd.date_range("2026-01-01", periods=n, freq="D")
    promo = (np.arange(n) % 7 == 0).astype(float)
    sales = 50 + 10 * promo + rng.normal(0, 1, n)
    frame = pd.DataFrame({"day": day, "sales": sales, "promo": promo})
    return timeseries(frame, time="day", freq="D", types={"promo": "continuous"})


@dataclass(frozen=True)
class _Spy:
    """Forecasts ``sales`` as its last value plus ten times the promo it is handed."""

    seen: list[dict[str, Any]] = field(default_factory=list)
    future_covariates: tuple[str, ...] = ("promo",)
    train: TimeSeriesData | None = None

    def fit(self, data: TimeSeriesData, *, seed: int = 0) -> _Spy:
        return _Spy(seen=self.seen, future_covariates=self.future_covariates, train=data)

    def forecast(
        self, horizon: int, *, seed: int = 0, future: TimeSeriesData | None = None
    ) -> ForecastResult:
        assert self.train is not None and future is not None
        self.seen.append({"train_end": self.train.times[-1], "future": future})
        path = self.train.column("sales")[-1] + 10 * np.asarray(future.column("promo"))
        return ForecastResult(series={"sales": np.tile(path, (5, 1))}, horizon=horizon)

    def __deepcopy__(self, memo: dict[int, Any]) -> _Spy:
        return self  # share the record of calls across folds


def test_backtest_hands_each_fold_the_true_future_covariates_and_no_target() -> None:
    data = _sales()
    spy = _Spy()
    report = backtest(spy, data, test_size=7, n_splits=3)
    assert report.future_covariates == ("promo",)
    assert len(spy.seen) == 3
    times = data.times
    for fold, call in zip(report.folds, spy.seen, strict=True):
        future = call["future"]
        # Only the time index and the declared covariate: the target never leaks.
        assert isinstance(future, TimeSeriesData)
        assert future.variables == ("promo",)
        assert "sales" not in future.schema
        assert future.n_rows == 7
        np.testing.assert_array_equal(future.times, times[fold.cutoff : fold.cutoff + 7])
        np.testing.assert_array_equal(
            future.column("promo"), data.column("promo")[fold.cutoff : fold.cutoff + 7]
        )
        # The model trained on everything before the cutoff and nothing after.
        assert call["train_end"] == times[fold.cutoff - 1]
    # The known promo makes the spike forecastable, so the error is the noise.
    assert max(report.values("sales", "mae")) < 2.0


def test_a_model_without_future_covariates_is_called_as_before() -> None:
    report = backtest(Naive(n_draws=20), _sales(), test_size=7, n_splits=2)
    assert report.future_covariates == ()
    assert report.nodes == ("sales", "promo")


def test_a_future_covariate_that_is_not_a_column_is_refused() -> None:
    spy = _Spy(future_covariates=("price",))
    with pytest.raises(ValueError, match=r"\['price'\]"):
        backtest(spy, _sales(), test_size=7)


def test_a_forecast_node_declared_as_a_future_covariate_is_refused() -> None:
    spy = _Spy(future_covariates=("promo", "sales"))
    with pytest.raises(ValueError, match="both forecast and declared future covariates"):
        backtest(spy, _sales(), test_size=7)
