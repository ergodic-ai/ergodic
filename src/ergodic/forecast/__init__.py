# SPDX-License-Identifier: Apache-2.0
"""Forecasting: Bayesian trajectories with causal structure in them.

The pillar is built in layers, and this is the layer that needs no sampler: the
result object every model returns (:class:`ForecastResult`), the metrics
(:func:`compute_metrics` and the functions behind it), the naive baselines
(:class:`Naive`, :class:`SeasonalNaive`, :class:`Drift`), the rolling-origin
harness (:func:`backtest`), the window-graph builder (:func:`window_graph`), the
staged-fitting plan (:class:`TrainingSchedule`), and the per-node scalers.

Two ideas hold it together. Predictive structure is a plain DAG over lagged node
names, which is exactly what temporal discovery already returns, so a discovered
graph feeds a forecast with no translation. And any object that fits a series and
forecasts a horizon (:class:`ForecastModel`) can be backtested, so a baseline,
the Bayesian forecaster, and somebody else's model all land in one table.
"""

from __future__ import annotations

from ._backtest import BacktestFold, BacktestReport, backtest
from ._baselines import Drift, Naive, SeasonalNaive
from ._metrics import (
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
    rolling_accuracy,
    rolling_accuracy_3,
    rolling_accuracy_6,
    rolling_mape,
    rolling_mape_3,
    rolling_mape_6,
    smape,
)
from ._protocol import ForecastModel
from ._result import ForecastResult
from ._scalers import IdentityScaler, LogScaler, RatioScaler, Scaler, ZScoreScaler
from ._schedule import NodeRole, Stage, StageStrategy, TrainingSchedule
from ._structure import bases, max_lag, split_lag, validate_window_graph, window_graph

__all__ = [
    "BacktestFold",
    "BacktestReport",
    "Drift",
    "ForecastModel",
    "ForecastResult",
    "IdentityScaler",
    "LogScaler",
    "Naive",
    "NodeRole",
    "RatioScaler",
    "Scaler",
    "SeasonalNaive",
    "Stage",
    "StageStrategy",
    "TrainingSchedule",
    "ZScoreScaler",
    "accuracy",
    "accuracy_full",
    "backtest",
    "bases",
    "compute_metrics",
    "coverage",
    "crps",
    "mae",
    "mape",
    "mape_full",
    "mase",
    "max_lag",
    "rmse",
    "rolling_accuracy",
    "rolling_accuracy_3",
    "rolling_accuracy_6",
    "rolling_mape",
    "rolling_mape_3",
    "rolling_mape_6",
    "smape",
    "split_lag",
    "validate_window_graph",
    "window_graph",
]
