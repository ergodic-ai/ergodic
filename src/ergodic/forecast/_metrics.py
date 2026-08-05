# SPDX-License-Identifier: Apache-2.0
"""Forecast metrics: point scores on the median path, distribution scores on the cloud.

Every function takes the actuals ``(H,)`` and the forecast draws ``(S, H)``. A
point metric reduces the draws to their per-step median first, so one call site
serves a posterior, a bootstrap, and a plain point path alike (pass a 1-D array
and it is read as the path itself). Distribution metrics read the whole cloud:
:func:`coverage` asks how often the truth fell inside the band, :func:`crps`
scores the shape of the forecast against the single observation.

Conventions worth knowing, all carried over from the source these came from:
:func:`mape` masks zero actuals and is NaN when every actual is zero;
:func:`smape` runs on the 0 to 200 scale; the rolling metrics score window sums,
which is how a planner reads a quarter rather than a month; and the accuracy
family is ``1 - clip(relative error, 0, 1)``, so it lands in ``[0, 1]`` and reads
as a share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "accuracy",
    "accuracy_full",
    "compute_metrics",
    "coverage",
    "crps",
    "mae",
    "mape",
    "mape_full",
    "mase",
    "rmse",
    "rolling_accuracy",
    "rolling_accuracy_3",
    "rolling_accuracy_6",
    "rolling_mape",
    "rolling_mape_3",
    "rolling_mape_6",
    "smape",
]


def _actual(actual: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
    """The truth as a 1-D float array."""
    arr = np.asarray(actual, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"actual must be 1-D (H,), got shape {arr.shape}")
    return arr


def _point(samples: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
    """The point path a point metric scores: the per-step median of the draws.

    A 1-D array is already a path and comes back unchanged.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"samples must be 2-D (S, H) or 1-D (H,), got shape {arr.shape}")
    return np.asarray(np.median(arr, axis=0), dtype=np.float64)


def _cloud(samples: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
    """The draws a distribution metric scores, as a 2-D ``(S, H)`` array."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"samples must be 2-D (S, H) or 1-D (H,), got shape {arr.shape}")
    return arr


def _aligned(
    actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The truth and the point path, checked to be the same length."""
    truth = _actual(actual)
    predicted = _point(samples)
    if truth.shape != predicted.shape:
        raise ValueError(
            f"actual has {truth.shape[0]} steps, the forecast has {predicted.shape[0]}"
        )
    return truth, predicted


# -- point metrics ------------------------------------------------------------


def mae(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """Mean absolute error of the median path."""
    truth, predicted = _aligned(actual, samples)
    return float(np.mean(np.abs(truth - predicted)))


def rmse(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """Root mean squared error of the median path."""
    truth, predicted = _aligned(actual, samples)
    return float(np.sqrt(np.mean((truth - predicted) ** 2)))


def mape(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """Mean absolute percentage error, in percent, over the non-zero actuals.

    NaN when every actual is zero, since there is nothing to take a percentage of.
    """
    truth, predicted = _aligned(actual, samples)
    mask = truth != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((truth[mask] - predicted[mask]) / truth[mask])) * 100.0)


def smape(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """Symmetric mean absolute percentage error on the 0 to 200 scale."""
    truth, predicted = _aligned(actual, samples)
    denom = np.abs(truth) + np.abs(predicted)
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(truth[mask] - predicted[mask]) / denom[mask]) * 100.0)


def rolling_mape(
    actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any, window: int
) -> float:
    """MAPE of rolling window sums, in percent.

    Each window position sums the actuals and the forecast over ``window`` steps
    and scores the two sums; the result averages those. NaN when the horizon is
    shorter than the window or every window of actuals sums to zero.
    """
    truth, predicted = _aligned(actual, samples)
    horizon = truth.shape[0]
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if horizon < window:
        return float("nan")
    scores: list[float] = []
    for start in range(horizon - window + 1):
        end = start + window
        sum_a = float(truth[start:end].sum())
        sum_p = float(predicted[start:end].sum())
        if sum_a != 0:
            scores.append(abs(sum_p - sum_a) / abs(sum_a) * 100.0)
    return float(np.mean(scores)) if scores else float("nan")


def rolling_mape_3(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """:func:`rolling_mape` over three-step windows."""
    return rolling_mape(actual, samples, 3)


def rolling_mape_6(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """:func:`rolling_mape` over six-step windows."""
    return rolling_mape(actual, samples, 6)


def mape_full(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """:func:`rolling_mape` over the whole horizon: the error on the total."""
    truth = _actual(actual)
    return rolling_mape(actual, samples, int(truth.shape[0]))


def accuracy(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """``1 - clip(relative error, 0, 1)``, averaged over the non-zero actuals.

    One is a perfect path, zero is every step off by a full 100 percent or more.
    NaN when every actual is zero.
    """
    truth, predicted = _aligned(actual, samples)
    mask = truth != 0
    if not mask.any():
        return float("nan")
    relative = np.abs((predicted[mask] - truth[mask]) / truth[mask])
    return float(np.mean(1.0 - np.clip(relative, 0.0, 1.0)))


def rolling_accuracy(
    actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any, window: int
) -> float:
    """:func:`accuracy` on rolling window sums. NaN when the horizon is too short."""
    truth, predicted = _aligned(actual, samples)
    horizon = truth.shape[0]
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if horizon < window:
        return float("nan")
    scores: list[float] = []
    for start in range(horizon - window + 1):
        end = start + window
        sum_a = float(truth[start:end].sum())
        sum_p = float(predicted[start:end].sum())
        if sum_a != 0:
            scores.append(1.0 - min(abs(sum_p - sum_a) / abs(sum_a), 1.0))
    return float(np.mean(scores)) if scores else float("nan")


def rolling_accuracy_3(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """:func:`rolling_accuracy` over three-step windows."""
    return rolling_accuracy(actual, samples, 3)


def rolling_accuracy_6(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """:func:`rolling_accuracy` over six-step windows."""
    return rolling_accuracy(actual, samples, 6)


def accuracy_full(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """:func:`rolling_accuracy` over the whole horizon: accuracy on the total."""
    truth = _actual(actual)
    return rolling_accuracy(actual, samples, int(truth.shape[0]))


def mase(
    actual: npt.NDArray[Any] | Any,
    samples: npt.NDArray[Any] | Any,
    train: npt.NDArray[Any] | Any,
    m: int = 1,
) -> float:
    """Mean absolute scaled error against the in-sample seasonal naive.

    The scale is the mean absolute error a seasonal naive would have made on the
    training data at period ``m`` (``m=1`` is the plain naive, the right choice
    with no seasonality). One means the forecast matched that floor, below one
    beat it. NaN when the training series is too short for one seasonal
    difference, or when that difference is flat, since the scale would be zero.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    history = np.asarray(train, dtype=np.float64).ravel()
    if history.size <= m:
        return float("nan")
    scale = float(np.nanmean(np.abs(history[m:] - history[:-m])))
    if not np.isfinite(scale) or scale == 0.0:
        return float("nan")
    return mae(actual, samples) / scale


# -- distribution metrics -----------------------------------------------------


def coverage(
    actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any, level: float = 0.9
) -> float:
    """The share of actuals inside the central ``level`` interval of the draws.

    NaN when the band cannot be formed anywhere it is needed, which is what the
    error metrics already do with a NaN in the same place. Returning zero there
    would read as "the forecast missed every step" and quietly drag an average
    down, when the truth is that there was nothing to score.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    truth = _actual(actual)
    draws = _cloud(samples)
    if draws.shape[1] != truth.shape[0]:
        raise ValueError(f"actual has {truth.shape[0]} steps, the forecast has {draws.shape[1]}")
    tail = (1.0 - level) / 2.0
    lo = np.percentile(draws, tail * 100.0, axis=0)
    hi = np.percentile(draws, (1.0 - tail) * 100.0, axis=0)
    if not (
        bool(np.all(np.isfinite(lo)))
        and bool(np.all(np.isfinite(hi)))
        and bool(np.all(np.isfinite(truth)))
    ):
        return float("nan")
    return float(np.mean((truth >= lo) & (truth <= hi)))


def crps(actual: npt.NDArray[Any] | Any, samples: npt.NDArray[Any] | Any) -> float:
    """Continuous ranked probability score, averaged over the horizon.

    The sample energy form ``E|X - y| - 0.5 E|X - X'|`` per step, with the second
    term read off the sorted draws so the cost is ``S log S`` rather than
    ``S^2``. Lower is better, and it collapses to the absolute error when the
    forecast is a point.
    """
    truth = _actual(actual)
    draws = _cloud(samples)
    if draws.shape[1] != truth.shape[0]:
        raise ValueError(f"actual has {truth.shape[0]} steps, the forecast has {draws.shape[1]}")
    n = int(draws.shape[0])
    weights = 2.0 * np.arange(1, n + 1) - n - 1
    scores: list[float] = []
    for t in range(truth.shape[0]):
        sorted_draws = np.sort(draws[:, t])
        spread = float(np.sum(weights * sorted_draws) / (n * n))
        error = float(np.mean(np.abs(sorted_draws - truth[t])))
        scores.append(error - spread)
    return float(np.mean(scores))


# -- the table ----------------------------------------------------------------


def compute_metrics(
    actual: npt.NDArray[Any] | Any,
    samples: npt.NDArray[Any] | Any,
    *,
    coverage_level: float = 0.9,
    train: npt.NDArray[Any] | Any | None = None,
    m: int = 1,
) -> dict[str, float]:
    """Every metric for one node and one fold, keyed by name.

    ``train`` is the in-sample history the fold was fitted on; give it and the
    table carries ``mase`` at period ``m``, leave it out and the key is absent.
    """
    truth = _actual(actual)
    point = _point(samples)
    out = {
        "mae": mae(truth, point),
        "rmse": rmse(truth, point),
        "mape": mape(truth, point),
        "smape": smape(truth, point),
        "rolling_mape_3": rolling_mape_3(truth, point),
        "rolling_mape_6": rolling_mape_6(truth, point),
        "mape_full": mape_full(truth, point),
        "accuracy": accuracy(truth, point),
        "rolling_accuracy_3": rolling_accuracy_3(truth, point),
        "rolling_accuracy_6": rolling_accuracy_6(truth, point),
        "accuracy_full": accuracy_full(truth, point),
        "coverage": coverage(truth, samples, level=coverage_level),
        "crps": crps(truth, samples),
    }
    if train is not None:
        out["mase"] = mase(truth, point, train, m)
    return out
