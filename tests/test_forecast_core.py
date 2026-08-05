# SPDX-License-Identifier: Apache-2.0
"""Tests for the forecast result, the naive baselines, and the scalers.

The baselines run on seeded worlds with planted structure at realistic scales:
a monthly seasonal series with a small noise floor, and a linear trend with a
noisier one. Twenty seeds each, seventy-two months of history, twelve held out.
The claims are the ones a forecaster has to earn honestly: the baseline that
matches the world's structure wins on the world it matches, and the intervals
the textbook formulas produce actually cover.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ergodic.data import timeseries
from ergodic.forecast import (
    Drift,
    ForecastResult,
    IdentityScaler,
    LogScaler,
    Naive,
    RatioScaler,
    SeasonalNaive,
    ZScoreScaler,
    coverage,
    mae,
    mase,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TimeSeriesData

N_SEEDS = 20
T = 72
HOLDOUT = 12
PERIOD = 12


# -- worlds -------------------------------------------------------------------


def _seasonal_world(seed: int, n: int = T) -> npt.NDArray[np.float64]:
    """y = 10 + 3 sin(2 pi t / 12) + eps(0.3): a clean yearly cycle."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.asarray(
        10.0 + 3.0 * np.sin(2.0 * np.pi * t / PERIOD) + rng.normal(0.0, 0.3, n),
        dtype=np.float64,
    )


def _trend_world(seed: int, n: int = T) -> npt.NDArray[np.float64]:
    """y = 0.5 t + eps(1.0): a steady climb with real noise on it."""
    rng = np.random.default_rng(10_000 + seed)
    t = np.arange(n)
    return np.asarray(0.5 * t + rng.normal(0.0, 1.0, n), dtype=np.float64)


def _series(values: npt.NDArray[np.float64]) -> TimeSeriesData:
    return timeseries({"t": np.arange(values.size), "y": values}, time="t")


def _split(values: npt.NDArray[np.float64]) -> tuple[TimeSeriesData, npt.NDArray[np.float64]]:
    """Train on everything but the last twelve points, hold those out."""
    cut = values.size - HOLDOUT
    return _series(values[:cut]), values[cut:]


# -- the result object --------------------------------------------------------


def test_result_accessors_read_the_cloud() -> None:
    draws = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = ForecastResult(series={"y": draws}, horizon=2)

    assert result.nodes == ("y",)
    assert result.n_draws == 3
    assert result.median().tolist() == [3.0, 4.0]
    assert result.mean().tolist() == [3.0, 4.0]
    assert result.draws("y").shape == (3, 2)
    lo, hi = result.interval(level=0.5)
    assert lo.tolist() == [2.0, 3.0]
    assert hi.tolist() == [4.0, 5.0]
    frame = result.to_frame()
    assert list(frame.columns) == ["node", "step", "median", "lo", "hi"]
    assert frame["step"].tolist() == [1, 2]


def test_result_needs_a_node_when_it_holds_several() -> None:
    two = ForecastResult(series={"a": np.ones((4, 3)), "b": np.zeros((4, 3))}, horizon=3)
    assert two.nodes == ("a", "b")
    with pytest.raises(ValueError, match="node= is required"):
        two.median()
    with pytest.raises(KeyError, match="no series"):
        two.median("c")
    assert two.median("b").tolist() == [0.0, 0.0, 0.0]


def test_result_validates_its_arrays() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        ForecastResult(series={"y": np.ones(3)}, horizon=3)
    with pytest.raises(ValueError, match="expected horizon"):
        ForecastResult(series={"y": np.ones((4, 2))}, horizon=3)
    with pytest.raises(ValueError, match="every node shares one draw axis"):
        ForecastResult(series={"a": np.ones((4, 3)), "b": np.ones((5, 3))}, horizon=3)
    with pytest.raises(ValueError, match="at least one series"):
        ForecastResult(series={}, horizon=3)
    # NaN draws are permitted on purpose: a masked window can leave one.
    nan_draws = np.array([[1.0, np.nan], [2.0, np.nan]])
    assert np.isnan(ForecastResult(series={"y": nan_draws}, horizon=2).median()[1])


def test_result_is_frozen_and_its_draws_are_read_only() -> None:
    result = ForecastResult(series={"y": np.ones((2, 2))}, horizon=2)
    with pytest.raises(AttributeError):
        result.horizon = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        result.draws()[0, 0] = 5.0


# -- the baselines ------------------------------------------------------------


def test_baselines_forecast_the_shape_they_promise() -> None:
    data = _series(_seasonal_world(0))
    for model in (Naive(), SeasonalNaive(PERIOD), Drift()):
        fitted = model.fit(data)
        assert not model.fitted, "fit must not touch the model it was called on"
        assert fitted.fitted
        result = fitted.forecast(HOLDOUT, seed=1)
        assert result.kind == "draws"
        assert result.draws("y").shape == (500, HOLDOUT)
        assert result.start == T  # the next step of the integer time index


def test_baseline_point_paths_are_the_textbook_ones() -> None:
    values = np.arange(24.0) + np.array([0.0, 0.5] * 12)
    data = _series(values)

    naive = Naive().fit(data).forecast(3, seed=0)
    assert naive.median().tolist() == pytest.approx([values[-1]] * 3, abs=0.5)

    seasonal = SeasonalNaive(4).fit(data).forecast(6, seed=0)
    assert seasonal.median().tolist() == pytest.approx(
        list(values[-4:]) + list(values[-4:-2]), abs=0.5
    )

    drift = Drift().fit(data).forecast(3, seed=0)
    slope = (values[-1] - values[0]) / (values.size - 1)
    assert drift.median().tolist() == pytest.approx(
        [values[-1] + slope, values[-1] + 2 * slope, values[-1] + 3 * slope], abs=0.5
    )


def test_baseline_intervals_widen_with_the_horizon() -> None:
    """Naive and Drift widen with sqrt(h); SeasonalNaive is flat inside a season."""
    data = _series(_seasonal_world(1))
    naive = Naive().fit(data).forecast(24, seed=2)
    lo, hi = naive.interval(level=0.9)
    width = hi - lo
    assert width[0] < width[5] < width[23]
    assert width[3] / width[0] == pytest.approx(2.0, rel=0.15)  # sqrt(4)

    seasonal = SeasonalNaive(PERIOD).fit(data).forecast(24, seed=2)
    slo, shi = seasonal.interval(level=0.9)
    swidth = shi - slo
    assert swidth[0] == pytest.approx(swidth[11], rel=0.15)
    assert swidth[12] / swidth[0] == pytest.approx(np.sqrt(2.0), rel=0.15)


def test_baselines_are_deterministic_given_a_seed() -> None:
    data = _series(_trend_world(3))
    first = Drift().fit(data).forecast(6, seed=7).draws("y")
    second = Drift().fit(data).forecast(6, seed=7).draws("y")
    third = Drift().fit(data).forecast(6, seed=8).draws("y")
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_seasonal_naive_wins_on_the_seasonal_world() -> None:
    """Criterion 5: the model that matches the world beats both alternatives on MASE."""
    scores: dict[str, list[float]] = {"seasonal": [], "naive": [], "drift": []}
    for seed in range(N_SEEDS):
        values = _seasonal_world(seed)
        data, actual = _split(values)
        history = data.column("y")
        for name, model in (
            ("seasonal", SeasonalNaive(PERIOD)),
            ("naive", Naive()),
            ("drift", Drift()),
        ):
            result = model.fit(data).forecast(HOLDOUT, seed=100 + seed)
            scores[name].append(mase(actual, result.draws("y"), history, m=1))
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    assert means["seasonal"] < means["naive"], means
    assert means["seasonal"] < means["drift"], means


def test_drift_wins_on_the_trend_world() -> None:
    """Criterion 6: on a steady climb, extending the line beats repeating a value."""
    scores: dict[str, list[float]] = {"drift": [], "naive": [], "seasonal": []}
    for seed in range(N_SEEDS):
        values = _trend_world(seed)
        data, actual = _split(values)
        for name, model in (
            ("drift", Drift()),
            ("naive", Naive()),
            ("seasonal", SeasonalNaive(PERIOD)),
        ):
            result = model.fit(data).forecast(HOLDOUT, seed=200 + seed)
            scores[name].append(mae(actual, result.draws("y")))
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    assert means["drift"] < means["naive"], means
    assert means["drift"] < means["seasonal"], means


def test_baseline_intervals_cover_across_both_worlds() -> None:
    """Criterion 7: pooled over 40 worlds and 480 held-out points, the 90% band covers."""
    hits: list[float] = []
    for seed in range(N_SEEDS):
        data, actual = _split(_seasonal_world(seed))
        result = SeasonalNaive(PERIOD).fit(data).forecast(HOLDOUT, seed=300 + seed)
        hits.append(coverage(actual, result.draws("y"), level=0.9))
    for seed in range(N_SEEDS):
        data, actual = _split(_trend_world(seed))
        result = Drift().fit(data).forecast(HOLDOUT, seed=400 + seed)
        hits.append(coverage(actual, result.draws("y"), level=0.9))
    pooled = float(np.mean(hits))  # every world contributes the same 12 points
    assert 0.75 <= pooled <= 0.99, pooled


def test_baselines_refuse_what_they_cannot_read() -> None:
    short = _series(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="Drift needs 3"):
        Drift().fit(short)
    with pytest.raises(ValueError, match="needs 13"):
        SeasonalNaive(PERIOD).fit(_series(np.arange(12.0)))
    with pytest.raises(ValueError, match="period must be"):
        SeasonalNaive(0)
    with pytest.raises(ValueError, match="not fitted"):
        Naive().forecast(3)
    with pytest.raises(ValueError, match="horizon must be"):
        Naive().fit(_series(np.arange(10.0))).forecast(0)
    categorical = timeseries(
        {"t": np.arange(10), "label": ["a", "b"] * 5, "y": np.arange(10.0)}, time="t"
    )
    with pytest.raises(TypeError, match="not numeric"):
        Naive().fit(categorical)


def test_baselines_forecast_every_numeric_series() -> None:
    rng = np.random.default_rng(5)
    data = timeseries(
        {"t": np.arange(40), "a": rng.normal(size=40), "b": rng.normal(size=40) + 10.0},
        time="t",
    )
    result = Naive().fit(data).forecast(4, seed=1)
    assert result.nodes == ("a", "b")
    assert not np.allclose(result.median("a"), result.median("b"))


# -- the scalers --------------------------------------------------------------


def test_every_scaler_round_trips() -> None:
    """Criterion 17: transform then inverse returns the data, to 1e-12."""
    rng = np.random.default_rng(11)
    positive = rng.uniform(5.0, 50.0, 200)
    signed = rng.normal(3.0, 7.0, 200)

    for scaler, values in (
        (RatioScaler(), positive),
        (ZScoreScaler(), signed),
        (IdentityScaler(), signed),
        (LogScaler(), positive),
    ):
        fitted = scaler.fit(values)
        back = fitted.inverse(fitted.transform(values))
        assert np.allclose(back, values, rtol=1e-12, atol=0.0), type(scaler).__name__


def test_fit_returns_a_new_scaler_and_leaves_the_old_one_alone() -> None:
    raw = RatioScaler()
    fitted = raw.fit(np.array([10.0, 20.0, 30.0]))
    assert raw.median == 1.0
    assert fitted.median == 20.0
    assert fitted is not raw


def test_ratio_scaler_ignores_non_positives_when_it_takes_the_median() -> None:
    values = np.array([0.0, 0.0, 0.0, 2.0, 4.0, 6.0, -1.0])
    assert RatioScaler().fit(values).median == 4.0  # median of (2, 4, 6), not of all seven
    assert RatioScaler().fit(np.zeros(5)).median == 1.0  # nothing positive: stay put
    floored = RatioScaler(floor=0.5).fit(values).transform(np.array([0.1, 4.0]))
    assert floored.tolist() == [0.5, 1.0]


def test_zscore_scaler_standardizes_and_survives_a_constant_series() -> None:
    values = np.array([2.0, 4.0, 6.0, 8.0])
    fitted = ZScoreScaler().fit(values)
    assert fitted.mean == pytest.approx(5.0)
    assert float(np.mean(fitted.transform(values))) == pytest.approx(0.0)
    assert ZScoreScaler().fit(np.full(5, 3.0)).std == 1.0


def test_log_scaler_refuses_values_below_its_offset() -> None:
    scaler = LogScaler(offset=1.0)
    assert scaler.transform(np.array([0.0])).tolist() == [0.0]
    with pytest.raises(ValueError, match=r"needs values above -1\.0"):
        scaler.transform(np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="needs values above"):
        scaler.transform(np.array([-2.0]))
    assert LogScaler(offset=5.0).transform(np.array([-4.0])).tolist() == [0.0]


def test_unfitted_scalers_are_the_identity() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert RatioScaler().transform(values).tolist() == values.tolist()
    assert ZScoreScaler().transform(values).tolist() == values.tolist()
    assert IdentityScaler().fit(values).transform(values).tolist() == values.tolist()
