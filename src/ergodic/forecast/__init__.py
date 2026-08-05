# SPDX-License-Identifier: Apache-2.0
"""Forecasting: Bayesian trajectories with causal structure in them.

:func:`forecast` is the entry point: a typed time series and a horizon in, a
:class:`ForecastResult` out, with a posterior behind every step rather than a
point and an error bar bolted on. Underneath it, a
:class:`TemporalMechanism` per series holds that series' structural equation.
The one built so far, :class:`StructuralMechanism`, is a damped local linear
trend with a season and a regression on parents, composed from those pieces so a
fitted forecast can be taken apart again.

The rest of the pillar needs no sampler: the result object every model returns,
the metrics, the naive baselines that give a backtest an honest floor, the
rolling-origin harness itself, the window-graph builder, the staged-fitting plan,
and the per-node scalers.

Two ideas hold it together. Predictive structure is a plain DAG over lagged node
names, which is exactly what temporal discovery already returns, so a discovered
graph feeds a forecast with no translation. And any object that fits a series and
forecasts a horizon (:class:`ForecastModel`) can be backtested, so a baseline,
the Bayesian forecaster, and somebody else's model all land in one table.

pymc is imported inside the methods that sample, so the bare install never loads
it; ``pip install ergodic[forecast]`` adds it. Sampling runs through PyMC's JAX
backend when jax and numpyro are installed, which the engine spike measured at
about four times the C backend on this model.
"""

from __future__ import annotations

from ._backtest import BacktestFold, BacktestReport, backtest
from ._baselines import Drift, Naive, SeasonalNaive
from ._facade import forecast
from ._fit import NodeDiagnostics, NodeFit, NodeForecast, fit_node, forecast_node
from ._mechanism import (
    ScenarioMechanism,
    TemporalMechanism,
    assign_mechanisms,
    auto_scaler,
    infer_period,
)
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
from .mechanisms import StructuralMechanism

__all__ = [
    "BacktestFold",
    "BacktestReport",
    "Drift",
    "ForecastModel",
    "ForecastResult",
    "IdentityScaler",
    "LogScaler",
    "Naive",
    "NodeDiagnostics",
    "NodeFit",
    "NodeForecast",
    "NodeRole",
    "RatioScaler",
    "Scaler",
    "ScenarioMechanism",
    "SeasonalNaive",
    "Stage",
    "StageStrategy",
    "StructuralMechanism",
    "TemporalMechanism",
    "TrainingSchedule",
    "ZScoreScaler",
    "accuracy",
    "accuracy_full",
    "assign_mechanisms",
    "auto_scaler",
    "backtest",
    "bases",
    "compute_metrics",
    "coverage",
    "crps",
    "fit_node",
    "forecast",
    "forecast_node",
    "infer_period",
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
