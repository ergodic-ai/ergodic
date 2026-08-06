# SPDX-License-Identifier: Apache-2.0
"""Forecasting: Bayesian trajectories with causal structure in them.

:func:`forecast` is the entry point: a typed time series and a horizon in, a
:class:`ForecastResult` out, with a posterior behind every step rather than a
point and an error bar bolted on. Underneath it, a
:class:`TemporalMechanism` per series holds that series' structural equation.
:class:`StructuralMechanism` is the workhorse, a damped local linear trend with a
season and a regression on parents, composed from those pieces so a fitted
forecast can be taken apart again; :class:`CausalOU` puts the drivers in the
long-run attractor instead, and :class:`Bass` is the S-curve for a launch with no
history to extrapolate.

Give :func:`forecast` a window graph and the series read each other. Staged
fitting runs one node at a time in the order :func:`schedule` lays out, handing
each node's whole forecast posterior down to its children rather than a path; a
small graph is fitted jointly instead, so the nodes keep their joint posterior.
A driver whose future is known arrives through ``scenario=``, and one with
neither a mechanism nor a scenario is refused rather than quietly frozen.

The rest of the pillar needs no sampler: the result object every model returns,
the metrics, the naive baselines that give a backtest an honest floor, the
rolling-origin harness itself, the window-graph builder, the staged-fitting plan,
the reconciliation menu, and the per-node scalers.

Two ideas hold it together. Predictive structure is a plain DAG over lagged node
names, which is exactly what temporal discovery already returns, so a discovered
graph feeds a forecast with no translation. And any object that fits a series and
forecasts a horizon (:class:`ForecastModel`) can be backtested, so a baseline,
the Bayesian forecaster, and somebody else's model all land in one table.

pymc is imported inside the methods that sample, so the bare install never loads
it; ``pip install ergodic[forecast]`` adds it. Sampling runs through PyMC's JAX
backend when jax and numpyro are installed, which the engine spike measured at
about four times the C backend on this model.

A note on the name, because this pillar is the one place in the package where a
name is shadowed. ``ergodic.forecast`` at the top level is the *function*:
``from ergodic import forecast`` gives something callable. This subpackage is
reached with ``from ergodic.forecast import backtest``, which goes through the
import system and is unaffected. But ``import ergodic.forecast as mod`` binds the
function rather than the module, because Python prefers the attribute when one
exists.
"""

from __future__ import annotations

from ._backtest import BacktestFold, BacktestReport, backtest
from ._baselines import Drift, Naive, SeasonalNaive
from ._facade import forecast, schedule
from ._fit import (
    FrozenParent,
    JointFit,
    JointNode,
    NodeDiagnostics,
    NodeFit,
    NodeForecast,
    ParentColumn,
    align_draws,
    fit_joint,
    fit_node,
    forecast_joint,
    forecast_node,
    in_sample_median,
    inject_futures,
    lag_align,
)
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
from ._reconcile import (
    PROPORTION_RULES,
    RECONCILE_METHODS,
    WEIGHT_RULES,
    constraint_matrix,
    historical_shares,
    mint_weights,
    reconcile,
)
from ._result import FitStrategy, ForecastResult
from ._scalers import IdentityScaler, LogScaler, RatioScaler, Scaler, ZScoreScaler
from ._schedule import NodeRole, Stage, StageStrategy, TrainingSchedule
from ._staged import (
    AUTO_JOINT_MAX_NODES,
    FIT_STRATEGIES,
    NodeState,
    PosteriorStore,
    design_keys,
    mcmc_node_count,
    node_roles,
    resolve_strategy,
)
from ._structure import bases, max_lag, split_lag, validate_window_graph, window_graph
from .mechanisms import Bass, CausalOU, StructuralMechanism

__all__ = [
    "AUTO_JOINT_MAX_NODES",
    "FIT_STRATEGIES",
    "PROPORTION_RULES",
    "RECONCILE_METHODS",
    "WEIGHT_RULES",
    "BacktestFold",
    "BacktestReport",
    "Bass",
    "CausalOU",
    "Drift",
    "FitStrategy",
    "ForecastModel",
    "ForecastResult",
    "FrozenParent",
    "IdentityScaler",
    "JointFit",
    "JointNode",
    "LogScaler",
    "Naive",
    "NodeDiagnostics",
    "NodeFit",
    "NodeForecast",
    "NodeRole",
    "NodeState",
    "ParentColumn",
    "PosteriorStore",
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
    "align_draws",
    "assign_mechanisms",
    "auto_scaler",
    "backtest",
    "bases",
    "compute_metrics",
    "constraint_matrix",
    "coverage",
    "crps",
    "design_keys",
    "fit_joint",
    "fit_node",
    "forecast",
    "forecast_joint",
    "forecast_node",
    "historical_shares",
    "in_sample_median",
    "infer_period",
    "inject_futures",
    "lag_align",
    "mae",
    "mape",
    "mape_full",
    "mase",
    "max_lag",
    "mcmc_node_count",
    "mint_weights",
    "node_roles",
    "reconcile",
    "resolve_strategy",
    "rmse",
    "rolling_accuracy",
    "rolling_accuracy_3",
    "rolling_accuracy_6",
    "rolling_mape",
    "rolling_mape_3",
    "rolling_mape_6",
    "schedule",
    "smape",
    "split_lag",
    "validate_window_graph",
    "window_graph",
]
