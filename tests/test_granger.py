# SPDX-License-Identifier: Apache-2.0
"""Tests for the Granger table: a planted direction, validation, determinism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ergodic.data import tabular, timeseries
from ergodic.discovery._granger import granger

if TYPE_CHECKING:
    from ergodic.data import TimeSeriesData


def _granger_world(seed: int = 0, n: int = 2000) -> TimeSeriesData:
    """X is AR(1); Y[t] = 0.8 X[t-1] + 0.5 Y[t-1] + e: X drives Y, not back."""
    rng = np.random.default_rng(seed)
    ex, ey = rng.normal(size=(2, n))
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = ex[0], ey[0]
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.8 * x[t - 1] + 0.5 * y[t - 1] + ey[t]
    return timeseries({"t": np.arange(n), "X": x, "Y": y}, time="t")


def test_granger_finds_the_planted_direction() -> None:
    result = granger(_granger_world(), max_lag=1, alpha=0.01)
    assert set(result.pvalues) == {("X", "Y"), ("Y", "X")}  # all ordered pairs
    assert ("X", "Y") in result.significant
    assert ("Y", "X") not in result.significant
    assert result.pvalues[("X", "Y")] < 1e-6
    assert result.max_lag == 1
    assert result.alpha == 0.01
    assert result.n == 1999  # one row lost to the lag


def test_granger_rejects_non_timeseries() -> None:
    data = tabular({"X": np.arange(20.0), "Y": np.arange(20.0)})
    with pytest.raises(TypeError, match="TimeSeriesData"):
        granger(data)  # type: ignore[arg-type]


def test_granger_rejects_categorical_variables() -> None:
    n = 40
    rng = np.random.default_rng(1)
    data = timeseries(
        {"t": np.arange(n), "X": rng.normal(size=n), "C": np.arange(n) % 3},
        time="t",
        types={"C": "categorical"},
    )
    with pytest.raises(ValueError, match="C"):
        granger(data)


def test_granger_validates_parameters() -> None:
    data = _granger_world(seed=2, n=60)
    with pytest.raises(ValueError, match="alpha"):
        granger(data, alpha=1.0)
    with pytest.raises(ValueError, match="max_lag"):
        granger(data, max_lag=0)


def test_granger_is_deterministic() -> None:
    data = _granger_world(seed=1)
    first = granger(data, max_lag=2, alpha=0.05)
    second = granger(data, max_lag=2, alpha=0.05)
    assert first.pvalues == second.pvalues
    assert first.significant == second.significant
