# SPDX-License-Identifier: Apache-2.0
"""The backtester: rolling-origin evaluation for anything that forecasts.

:func:`backtest` takes an unfitted model, cuts the history at one or more
origins, refits from scratch on each prefix, forecasts the window that follows,
and scores it. Refitting per fold is the whole point. Reusing a posterior across
folds would leak the future into the past, exactly the way reusing a fitted
nuisance model across cross-fitting folds does.

The model is deep-copied before each fit, so the object the caller passes in is
never fitted, never mutated, and cannot carry state from one fold to the next.

Cutoff arithmetic, in ``expanding`` mode: the last origin is ``T - test_size``,
the first is ``max(min_train_size, last - (n_splits - 1) * test_size)``, and the
origins in between are evenly spaced. ``single`` mode keeps the last origin
only. Training is always the whole prefix ``[0, cutoff)``, so later folds see
more history, never a sliding window.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._metrics import compute_metrics

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt
    import pandas as pd

    from ..data import TimeSeriesData
    from ._protocol import ForecastModel

__all__ = ["BacktestFold", "BacktestReport", "backtest"]

# Metrics where a bigger number is a better forecast. Everything else is an
# error measure, so lower is better, except coverage, which is judged by how
# close it lands to its nominal level.
_HIGHER_IS_BETTER = frozenset(
    {"accuracy", "rolling_accuracy_3", "rolling_accuracy_6", "accuracy_full"}
)


def _cutoffs(
    n_rows: int, mode: str, test_size: int, n_splits: int, min_train_size: int
) -> list[int]:
    """The training-set end positions, one per fold, in ascending order."""
    if mode not in ("single", "expanding"):
        raise ValueError(f"mode must be 'single' or 'expanding', got {mode!r}")
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1, got {test_size}")
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if min_train_size < 1:
        raise ValueError(f"min_train_size must be >= 1, got {min_train_size}")

    last = n_rows - test_size
    if mode == "single":
        if last < min_train_size:
            raise ValueError(
                f"not enough data: T={n_rows}, test_size={test_size}, "
                f"min_train_size={min_train_size}"
            )
        return [last]

    first = max(min_train_size, last - (n_splits - 1) * test_size)
    if first > last:
        raise ValueError(
            f"not enough data for {n_splits} expanding splits: T={n_rows}, "
            f"test_size={test_size}, min_train_size={min_train_size}"
        )
    step = max(1, (last - first) // max(n_splits - 1, 1))
    cutoffs = list(range(first, last + 1, step))
    if cutoffs[-1] != last:
        cutoffs.append(last)
    return cutoffs[:n_splits]


@dataclass(frozen=True)
class BacktestFold:
    """One train-test split: where it was cut, what was forecast, how it scored."""

    index: int
    cutoff: int
    horizon: int
    metrics: Mapping[str, Mapping[str, float]]
    forecasts: Mapping[str, npt.NDArray[np.float64]] = field(repr=False)
    actuals: Mapping[str, npt.NDArray[np.float64]] = field(repr=False)
    fit_seed: int = 0
    forecast_seed: int = 0

    @property
    def nodes(self) -> tuple[str, ...]:
        """The scored series, in forecast order."""
        return tuple(self.metrics)

    def __repr__(self) -> str:
        return (
            f"BacktestFold({self.index}: cutoff={self.cutoff}, horizon={self.horizon}, "
            f"{len(self.metrics)} nodes)"
        )


@dataclass(frozen=True)
class BacktestReport:
    """Every fold of one backtest, plus the tables that read across them."""

    model: str
    mode: str
    test_size: int
    level: float
    folds: tuple[BacktestFold, ...]
    nodes: tuple[str, ...]
    metrics: tuple[str, ...]

    @property
    def cutoffs(self) -> tuple[int, ...]:
        """The training-set end position of each fold."""
        return tuple(fold.cutoff for fold in self.folds)

    def values(self, node: str, metric: str) -> tuple[float, ...]:
        """One metric for one node, fold by fold."""
        return tuple(
            float(fold.metrics[node][metric]) for fold in self.folds if node in fold.metrics
        )

    def _mean(self, node: str, metric: str) -> float:
        """The across-fold mean, ignoring folds where the metric was undefined."""
        finite = [v for v in self.values(node, metric) if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    def summary(self) -> pd.DataFrame:
        """Mean metric per node across folds: nodes down, metrics across."""
        import pandas as pd

        rows = {
            node: {metric: self._mean(node, metric) for metric in self.metrics}
            for node in self.nodes
        }
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(self.metrics))
        frame.index.name = "node"
        return frame

    def to_frame(self) -> pd.DataFrame:
        """A tidy frame: one row per fold, node, and metric."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for fold in self.folds:
            for node, scores in fold.metrics.items():
                for metric, value in scores.items():
                    rows.append(
                        {
                            "fold": fold.index,
                            "cutoff": fold.cutoff,
                            "node": node,
                            "metric": metric,
                            "value": float(value),
                        }
                    )
        return pd.DataFrame(rows)

    def compare(self, other: BacktestReport) -> pd.DataFrame:
        """This report against another, metric by metric, with the direction applied.

        ``delta`` is this report's mean minus ``other``'s. ``improved`` reads the
        direction of each metric: lower is better for the error measures, higher
        for the accuracy family, and closer to the nominal level for coverage.
        Comparing folds cut at different origins is on the caller; run both
        models through the same split to keep it fair.
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        metrics = list(self.metrics) + [m for m in other.metrics if m not in self.metrics]
        for node in sorted(set(self.nodes) | set(other.nodes)):
            for metric in metrics:
                mine = self._mean(node, metric) if node in self.nodes else float("nan")
                theirs = other._mean(node, metric) if node in other.nodes else float("nan")
                delta = mine - theirs
                if metric == "coverage":
                    improved = bool(abs(mine - self.level) < abs(theirs - other.level))
                elif metric in _HIGHER_IS_BETTER:
                    improved = bool(delta > 0)
                else:
                    improved = bool(delta < 0)
                rows.append(
                    {
                        "node": node,
                        "metric": metric,
                        "value": mine,
                        "other": theirs,
                        "delta": delta,
                        "improved": improved,
                    }
                )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"BacktestReport({self.model}, {self.mode}: {len(self.folds)} folds "
            f"at {list(self.cutoffs)}, horizon={self.test_size}, "
            f"{len(self.nodes)} nodes)"
        )


def backtest(
    model: ForecastModel,
    data: TimeSeriesData,
    *,
    mode: str = "expanding",
    test_size: int,
    n_splits: int = 3,
    min_train_size: int | None = None,
    seed: int = 0,
    level: float = 0.9,
    m: int = 1,
) -> BacktestReport:
    """Score ``model`` on ``data`` by refitting at each rolling origin.

    Parameters
    ----------
    model
        An unfitted model satisfying :class:`~ergodic.forecast.ForecastModel`.
        It is deep-copied per fold, so it comes back untouched.
    data
        The history. Rows are already in time order.
    mode
        ``"expanding"`` for several folds, each training on everything before
        its origin, or ``"single"`` for one split at the end.
    test_size
        The held-out window, which is also the forecast horizon.
    n_splits
        How many folds in expanding mode.
    min_train_size
        The shortest training prefix allowed, ``2 * test_size`` by default.
    seed
        The base seed. Fold ``i`` fits with ``seed + i`` and forecasts with
        ``seed + 1000 + i``, so no two folds share a random state.
    level
        The nominal level for the coverage metric and the interval it reads.
    m
        The season length for the MASE scale. One is the plain naive scale,
        which is the right choice when the series carries no seasonality.

    Returns
    -------
    BacktestReport
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if min_train_size is None:
        min_train_size = 2 * test_size
    cutoffs = _cutoffs(data.n_rows, mode, test_size, n_splits, min_train_size)

    columns = set(data.variables)
    folds: list[BacktestFold] = []
    for index, cutoff in enumerate(cutoffs):
        fit_seed = seed + index
        forecast_seed = seed + 1000 + index
        train = data.take(np.arange(cutoff, dtype=np.int64))
        fitted = copy.deepcopy(model).fit(train, seed=fit_seed)
        result = fitted.forecast(test_size, seed=forecast_seed)

        scored = [node for node in result.nodes if node in columns]
        if not scored:
            raise ValueError(
                f"none of the forecast series {list(result.nodes)} is a column of the data "
                f"{sorted(columns)}; there is nothing to score against"
            )
        metrics: dict[str, dict[str, float]] = {}
        forecasts: dict[str, npt.NDArray[np.float64]] = {}
        actuals: dict[str, npt.NDArray[np.float64]] = {}
        for node in scored:
            draws = result.draws(node)
            actual = np.asarray(data.column(node)[cutoff : cutoff + test_size], dtype=np.float64)
            history = np.asarray(train.column(node), dtype=np.float64)
            metrics[node] = compute_metrics(actual, draws, coverage_level=level, train=history, m=m)
            forecasts[node] = draws
            actuals[node] = actual
        folds.append(
            BacktestFold(
                index=index,
                cutoff=cutoff,
                horizon=test_size,
                metrics=metrics,
                forecasts=forecasts,
                actuals=actuals,
                fit_seed=fit_seed,
                forecast_seed=forecast_seed,
            )
        )

    nodes = tuple(folds[0].metrics)
    names: Sequence[str] = tuple(next(iter(folds[0].metrics.values())))
    return BacktestReport(
        model=type(model).__name__,
        mode=mode,
        test_size=test_size,
        level=level,
        folds=tuple(folds),
        nodes=nodes,
        metrics=tuple(names),
    )
