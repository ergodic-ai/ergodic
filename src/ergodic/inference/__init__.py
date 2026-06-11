# SPDX-License-Identifier: Apache-2.0
"""Causal inference: estimate causal effects from data and a causal model.

An estimator factors into independent pieces: the estimand family (from
:mod:`ergodic.identification`), the strategy that turns it into a number (a
:class:`CausalEstimator`), and the learner in each nuisance slot. ``estimate_effect``
runs identification then estimation and returns an :class:`EffectEstimate`.

    >>> from ergodic import dag, tabular
    >>> from ergodic.inference import estimate_effect
    >>> est = estimate_effect(graph, data, "T", "Y")          # doctest: +SKIP
    >>> est.value, est.interval()                              # doctest: +SKIP

The learner is structurally a sklearn estimator, so ``estimate_effect(..., learner="knn")``
or a slot mapping like ``learner={"outcome": GradientBoostingRegressor()}`` drops
any sklearn or xgboost model into the slots with no adapter. The result is
distribution-shaped: a point estimate with an analytic standard error, or
bootstrap draws when there is no closed form.

The quasi-experimental designs read panels and time series instead of a graph:
:func:`did` and :func:`event_study` over a :class:`~ergodic.data.PanelData`
(parallel trends), :func:`synthetic_control` (a convex donor combination), and
:func:`interrupted_time_series` over a :class:`~ergodic.data.TimeSeriesData`
(a projected pre-trend). Each states its identification in a :class:`Design`
record and returns the same :class:`EffectEstimate` surface.
"""

from __future__ import annotations

from ._design import Design
from ._estimate import estimate, estimate_effect
from ._estimator import CausalEstimator, Slot, Task
from ._learner import BayesianLearner, Learner
from ._registry import candidates
from ._result import EffectEstimate
from ._target import EffectTarget
from .estimators import (
    AIPW,
    DML,
    IPW,
    PLIV,
    DiDEffect,
    DRLearner,
    EventTimeEffect,
    GroupTimeEffect,
    ITSEffect,
    NullEffect,
    OutcomeRegression,
    RLearner,
    Stratification,
    SyntheticControlEffect,
    TwoStageLeastSquares,
    TwoStageRegression,
    Wald,
    XLearner,
    did,
    event_study,
    interrupted_time_series,
    synthetic_control,
)
from .learners import (
    OLS,
    KNNClassifier,
    KNNRegressor,
    LogisticRegression,
    PyMCClassifier,
    PyMCLearner,
)

__all__ = [
    "AIPW",
    "DML",
    "IPW",
    "OLS",
    "PLIV",
    "BayesianLearner",
    "CausalEstimator",
    "DRLearner",
    "Design",
    "DiDEffect",
    "EffectEstimate",
    "EffectTarget",
    "EventTimeEffect",
    "GroupTimeEffect",
    "ITSEffect",
    "KNNClassifier",
    "KNNRegressor",
    "Learner",
    "LogisticRegression",
    "NullEffect",
    "OutcomeRegression",
    "PyMCClassifier",
    "PyMCLearner",
    "RLearner",
    "Slot",
    "Stratification",
    "SyntheticControlEffect",
    "Task",
    "TwoStageLeastSquares",
    "TwoStageRegression",
    "Wald",
    "XLearner",
    "candidates",
    "did",
    "estimate",
    "estimate_effect",
    "event_study",
    "interrupted_time_series",
    "synthetic_control",
]
