# SPDX-License-Identifier: Apache-2.0
"""Tests for the forecast metrics.

Four anchors. Hand fixtures pin the arithmetic where the right answer is
obvious (a perfect forecast, a constant bias). CRPS is checked against the
brute-force double loop it optimizes away, on random cases. MASE is checked
against the definition that gives it meaning: a naive forecast scores one.
Coverage is checked on a quantile grid built so the answer is exact rather
than approximate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ergodic.data import timeseries
from ergodic.forecast import (
    Naive,
    accuracy,
    accuracy_full,
    compute_metrics,
    coverage,
    crps,
    mae,
    mape,
    mape_full,
    mase,
    rmse,
    rolling_accuracy_3,
    rolling_accuracy_6,
    rolling_mape_3,
    rolling_mape_6,
    smape,
)

if TYPE_CHECKING:
    import numpy.typing as npt


def _degenerate(path: npt.NDArray[np.float64], n_draws: int = 64) -> npt.NDArray[np.float64]:
    """A cloud with no spread at all: every draw is the same path."""
    return np.tile(np.asarray(path, dtype=np.float64), (n_draws, 1))


def _random_walk(seed: int, n: int, sigma: float = 1.0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return np.asarray(np.cumsum(rng.normal(0.0, sigma, n)), dtype=np.float64)


# -- criterion 1: hand fixtures -----------------------------------------------


def test_perfect_forecast_scores_perfectly() -> None:
    """A forecast that is exactly right scores zero error and full coverage."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    samples = _degenerate(actual)

    assert mae(actual, samples) == 0.0
    assert rmse(actual, samples) == 0.0
    assert mape(actual, samples) == 0.0
    assert smape(actual, samples) == 0.0
    assert crps(actual, samples) == 0.0
    assert coverage(actual, samples, level=0.9) == 1.0
    assert accuracy(actual, samples) == 1.0
    assert rolling_mape_3(actual, samples) == 0.0
    assert mape_full(actual, samples) == 0.0
    assert rolling_accuracy_3(actual, samples) == 1.0
    assert accuracy_full(actual, samples) == 1.0
    # Six-step windows do not fit in a four-step horizon.
    assert np.isnan(rolling_mape_6(actual, samples))
    assert np.isnan(rolling_accuracy_6(actual, samples))


def test_constant_bias_scores_by_hand() -> None:
    """Every step off by exactly one: mae = rmse = 1 and smape is the hand value."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    samples = _degenerate(actual + 1.0)

    assert mae(actual, samples) == pytest.approx(1.0)
    assert rmse(actual, samples) == pytest.approx(1.0)
    # 2 |a - (a+1)| / (|a| + |a+1|) = 2 / (2a + 1), as a percentage.
    expected = float(np.mean([200.0 / (2 * a + 1) for a in (1, 2, 3, 4)]))
    assert smape(actual, samples) == pytest.approx(expected)
    assert expected == pytest.approx(39.365, abs=1e-3)
    # mape averages 1/a; accuracy is one minus that, clipped.
    assert mape(actual, samples) == pytest.approx(100.0 * np.mean([1 / 1, 1 / 2, 1 / 3, 1 / 4]))
    assert accuracy(actual, samples) == pytest.approx(1.0 - np.mean([1.0, 0.5, 1 / 3, 0.25]))
    # The bias never cancels, so the error on the total is 4 out of 10.
    assert mape_full(actual, samples) == pytest.approx(40.0)


def test_mape_masks_zero_actuals_and_is_nan_when_all_zero() -> None:
    actual = np.array([0.0, 2.0, 0.0, 4.0])
    samples = _degenerate(np.array([1.0, 3.0, 1.0, 5.0]))
    assert mape(actual, samples) == pytest.approx(100.0 * np.mean([0.5, 0.25]))
    assert np.isnan(mape(np.zeros(4), samples))
    assert np.isnan(accuracy(np.zeros(4), samples))


def test_compute_metrics_carries_every_key_and_mase_only_with_train() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    samples = _degenerate(actual + 1.0)
    table = compute_metrics(actual, samples)
    assert set(table) == {
        "mae",
        "rmse",
        "mape",
        "smape",
        "rolling_mape_3",
        "rolling_mape_6",
        "mape_full",
        "accuracy",
        "rolling_accuracy_3",
        "rolling_accuracy_6",
        "accuracy_full",
        "coverage",
        "crps",
    }
    with_train = compute_metrics(actual, samples, train=np.arange(10.0), m=1)
    assert with_train["mase"] == pytest.approx(1.0)


# -- criterion 2: CRPS against the brute-force energy form ---------------------


def _crps_brute(actual: npt.NDArray[np.float64], samples: npt.NDArray[np.float64]) -> float:
    """CRPS by the definition: E|X - y| - 0.5 E|X - X'|, both by double loop."""
    scores = []
    for t in range(actual.shape[0]):
        x = samples[:, t]
        first = float(np.mean(np.abs(x - actual[t])))
        second = float(np.mean(np.abs(x[:, None] - x[None, :])))
        scores.append(first - 0.5 * second)
    return float(np.mean(scores))


def test_crps_matches_the_brute_force_energy_form() -> None:
    """Fifty random clouds: the sorted form and the double loop agree to 1e-10."""
    rng = np.random.default_rng(20260805)
    for _ in range(50):
        n_draws = int(rng.integers(2, 201))
        horizon = int(rng.integers(1, 9))
        centre = rng.normal(0.0, 10.0, horizon)
        spread = float(rng.uniform(0.1, 5.0))
        samples = centre[None, :] + rng.normal(0.0, spread, (n_draws, horizon))
        actual = centre + rng.normal(0.0, spread, horizon)
        assert crps(actual, samples) == pytest.approx(_crps_brute(actual, samples), rel=1e-10)


def test_crps_of_a_point_forecast_is_the_absolute_error() -> None:
    actual = np.array([1.0, 2.0])
    samples = _degenerate(np.array([1.5, 2.5]))
    assert crps(actual, samples) == pytest.approx(0.5)


# -- criterion 3: the MASE identity -------------------------------------------


def test_mase_of_the_naive_forecast_sits_at_one() -> None:
    """On a random walk, the one-step naive path scores about one, by construction.

    MASE divides by the in-sample naive error, so a forecast that is itself
    naive has to land near one. Twenty seeds, a 300-step test window each.
    """
    scores = []
    for seed in range(20):
        walk = _random_walk(seed, 600)
        train, actual = walk[:300], walk[300:]
        one_step = walk[299:599]  # the naive prediction: yesterday's value
        scores.append(mase(actual, one_step, train, m=1))
    assert all(0.6 <= s <= 1.6 for s in scores), scores
    assert 0.85 <= float(np.mean(scores)) <= 1.25


def test_the_naive_model_predicts_the_last_value() -> None:
    """The path the identity above assumes is the one the model actually forecasts."""
    walk = _random_walk(0, 60)
    data = timeseries({"t": np.arange(60), "y": walk}, time="t")
    result = Naive().fit(data).forecast(1, seed=3)
    assert result.median("y")[0] == pytest.approx(float(walk[-1]), abs=0.1)


def test_mase_is_nan_when_the_scale_cannot_be_formed() -> None:
    actual = np.array([1.0, 2.0])
    samples = _degenerate(actual)
    assert np.isnan(mase(actual, samples, np.array([3.0]), m=1))  # too short
    assert np.isnan(mase(actual, samples, np.full(10, 7.0), m=1))  # flat, scale zero
    with pytest.raises(ValueError, match="m must be"):
        mase(actual, samples, np.arange(10.0), m=0)


def test_mase_scales_with_the_seasonal_period() -> None:
    """A seasonal series has a small seasonal-naive scale and a big one-step scale."""
    t = np.arange(48)
    season = np.sin(2 * np.pi * t / 12)
    actual = np.zeros(4)
    samples = _degenerate(np.ones(4))
    assert mase(actual, samples, season, m=12) > mase(actual, samples, season, m=1)


# -- criterion 4: coverage is exact on a quantile grid ------------------------


def test_coverage_is_exactly_the_constructed_fraction() -> None:
    """A uniform draw grid puts the 5th and 95th percentiles exactly on 0.05 and 0.95."""
    horizon = 10
    grid = np.linspace(0.0, 1.0, 101)
    samples = np.tile(grid[:, None], (1, horizon))
    lo, hi = np.percentile(samples[:, 0], 5.0), np.percentile(samples[:, 0], 95.0)
    assert (lo, hi) == pytest.approx((0.05, 0.95))

    for inside in range(horizon + 1):
        actual = np.concatenate([np.full(inside, 0.5), np.full(horizon - inside, 2.0)])
        assert coverage(actual, samples, level=0.9) == inside / horizon
    # The boundary counts as covered, and a value just outside does not.
    edge = np.full(horizon, 0.95)
    assert coverage(edge, samples, level=0.9) == 1.0
    assert coverage(np.full(horizon, 0.9500001), samples, level=0.9) == 0.0


def test_coverage_rejects_a_level_outside_the_unit_interval() -> None:
    samples = _degenerate(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="level must be"):
        coverage(np.array([1.0, 2.0]), samples, level=1.0)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="steps"):
        mae(np.array([1.0, 2.0]), _degenerate(np.array([1.0, 2.0, 3.0])))
    with pytest.raises(ValueError, match="steps"):
        crps(np.array([1.0, 2.0]), _degenerate(np.array([1.0, 2.0, 3.0])))
    with pytest.raises(ValueError, match="1-D"):
        mae(np.ones((2, 2)), _degenerate(np.array([1.0, 2.0])))
