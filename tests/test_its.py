# SPDX-License-Identifier: Apache-2.0
"""Tests for interrupted time series: planted breaks, seasonality, and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic import EffectTarget
from ergodic.data import timeseries
from ergodic.inference.estimators._its import ITSEffect, interrupted_time_series

# -- a known-truth break -------------------------------------------------------

T = 120
BREAK = 80
LEVEL = 4.0
SLOPE = 0.05
# mean(t - t0) over post points 80..119 is 19.5, so the planted ATT is:
TRUTH = LEVEL + SLOPE * 19.5


def _planted(seed: int = 0, seasonal_amp: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    y = (
        10.0
        + 0.1 * t
        + (t >= BREAK) * (LEVEL + SLOPE * (t - BREAK))
        + seasonal_amp * np.sin(2 * np.pi * t / 12.0)
        + rng.normal(size=T)
    )
    return pd.DataFrame({"t": t, "y": y})


# -- recovery ------------------------------------------------------------------


def test_recovers_the_planted_break() -> None:
    data = timeseries(_planted(), time="t")
    est = interrupted_time_series(data, "y", at=BREAK)
    assert isinstance(est, ITSEffect)
    assert abs(est.level_change - LEVEL) < 2 * est.level_change_se
    assert abs(est.slope_change - SLOPE) < 2 * est.slope_change_se
    expected = est.level_change + est.slope_change * 19.5
    assert abs(est.value - expected) < 1e-9
    lo, hi = est.interval()
    assert lo <= TRUTH <= hi


def test_result_record_shape() -> None:
    data = timeseries(_planted(), time="t")
    est = interrupted_time_series(data, "y", at=BREAK)
    assert est.target is EffectTarget.ATT
    assert est.estimator == "interrupted_time_series"
    assert est.kind == "analytic"
    assert est.std_error is not None and est.std_error > 0.0
    assert est.learners == {}
    assert est.n == T
    assert est.at == BREAK
    assert len(est.times) == len(est.observed) == T
    assert len(est.fitted) == len(est.counterfactual) == T
    # before the break the counterfactual is the fit itself; after it they split
    assert np.allclose(est.fitted[:BREAK], est.counterfactual[:BREAK])
    post_gap = np.asarray(est.fitted[BREAK:]) - np.asarray(est.counterfactual[BREAK:])
    assert abs(float(np.mean(post_gap)) - est.value) < 1e-9
    assert est.diagnostics["n_pre"] == BREAK
    assert est.diagnostics["n_post"] == T - BREAK
    assert est.diagnostics["hac_lags"] == 3
    assert est.diagnostics["r2"] > 0.9
    assert est.design is not None
    assert est.design.strategy == "interrupted_time_series"
    assert est.design.treatment is None


def test_seasonality_covers_the_truth_and_tightens_the_error() -> None:
    data = timeseries(_planted(seasonal_amp=3.0), time="t")
    seasonal = interrupted_time_series(data, "y", at=BREAK, seasonality=12)
    plain = interrupted_time_series(data, "y", at=BREAK)
    lo, hi = seasonal.interval()
    assert lo <= TRUTH <= hi
    assert seasonal.std_error is not None and plain.std_error is not None
    assert seasonal.std_error < plain.std_error


def test_datetime_time_column() -> None:
    dates = pd.date_range("2024-01-01", periods=T, freq="D")
    frame = pd.DataFrame({"date": dates, "y": _planted()["y"].to_numpy()})
    data = timeseries(frame, time="date")
    at = pd.Timestamp("2024-03-21")  # 80 days after the start
    est = interrupted_time_series(data, "y", at=at)
    assert est.at == at
    assert est.diagnostics["n_pre"] == BREAK
    assert abs(est.level_change - LEVEL) < 2 * est.level_change_se


# -- validation ------------------------------------------------------------------


def test_break_too_close_to_either_end_raises() -> None:
    data = timeseries(_planted(), time="t")
    with pytest.raises(ValueError, match="at least 3 points"):
        interrupted_time_series(data, "y", at=2)
    with pytest.raises(ValueError, match="at least 3 points"):
        interrupted_time_series(data, "y", at=118)
    with pytest.raises(ValueError, match="at least 3 points"):
        interrupted_time_series(data, "y", at=500)


def test_bad_seasonality_raises() -> None:
    data = timeseries(_planted(), time="t")
    with pytest.raises(ValueError, match="seasonality"):
        interrupted_time_series(data, "y", at=BREAK, seasonality=1)
    with pytest.raises(ValueError, match="seasonality"):
        interrupted_time_series(data, "y", at=BREAK, seasonality=T)


def test_nan_outcome_raises() -> None:
    frame = _planted()
    frame.loc[7, "y"] = np.nan
    data = timeseries(frame, time="t")
    with pytest.raises(ValueError, match="missing values"):
        interrupted_time_series(data, "y", at=BREAK)


def test_unknown_outcome_raises() -> None:
    data = timeseries(_planted(), time="t")
    with pytest.raises(ValueError, match="not a variable"):
        interrupted_time_series(data, "z", at=BREAK)


# -- determinism -------------------------------------------------------------------


def test_repeated_calls_are_identical() -> None:
    data = timeseries(_planted(), time="t")
    first = interrupted_time_series(data, "y", at=BREAK)
    second = interrupted_time_series(data, "y", at=BREAK)
    assert first.value == second.value
    assert first.std_error == second.std_error
    assert first.level_change == second.level_change
    assert first.slope_change == second.slope_change
    assert first.counterfactual == second.counterfactual
    assert first.fitted == second.fitted
